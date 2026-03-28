import glob
import json
import os
import signal
import subprocess
import sys
import time
import threading
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── dashboard.py ───────────────────────────────────────────────────────────────
# MLOps dashboard for Knowledge Trainer.
# Run with: python src/dashboard.py
# Then open: http://localhost:8001
#
# Shows: version registry, score chart, loss chart, deploy buttons.
# Reads live from data/manifest.json and mlruns/.
# Deploy button promotes version AND restarts serve.py automatically.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

app = FastAPI(title="Knowledge Trainer Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Track the serve.py process
serve_process: subprocess.Popen | None = None
serve_status: dict = {"state": "unknown", "version": None, "started_at": None}


def find_serve_process() -> int | None:
    """Find PID of any running serve.py process."""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-Process python -ErrorAction SilentlyContinue | "
             "Where-Object { $_.CommandLine -like '*serve.py*' } | "
             "Select-Object -ExpandProperty Id"],
            capture_output=True, text=True
        )
        pids = [int(p.strip()) for p in result.stdout.strip().splitlines() if p.strip().isdigit()]
        return pids[0] if pids else None
    except Exception:
        return None


def kill_serve():
    """Kill any running serve.py process."""
    global serve_process
    # Kill our tracked process
    if serve_process and serve_process.poll() is None:
        serve_process.terminate()
        try:
            serve_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            serve_process.kill()
        serve_process = None

    # Also kill any other serve.py on port 8000
    try:
        subprocess.run(
            ["powershell", "-Command",
             "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | "
             "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"],
            capture_output=True
        )
    except Exception:
        pass
    time.sleep(1)


def start_serve(version: str):
    """Start serve.py as a background process."""
    global serve_process, serve_status
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = open(os.path.join(logs_dir, "serve.log"), "a", encoding="utf-8")
    log_file.write(f"\n\n{'='*60}\nStarted at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
    log_file.flush()

    serve_py = os.path.join(SRC_DIR, "serve.py")
    serve_process = subprocess.Popen(
        [sys.executable, serve_py],
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=log_file,
    )
    serve_status = {
        "state":      "running",
        "version":    version,
        "started_at": time.strftime("%H:%M:%S"),
        "pid":        serve_process.pid,
    }


def restart_serve_async(version: str):
    """Kill and restart serve.py in a background thread."""
    def _restart():
        serve_status["state"] = "restarting"
        kill_serve()
        time.sleep(2)
        start_serve(version)
    threading.Thread(target=_restart, daemon=True).start()


def load_manifest() -> dict:
    path = os.path.join(PROJECT_ROOT, "data", "manifest.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"current_version": None, "production_version": None, "versions": {}}


def load_mlflow_losses() -> dict:
    """Read training loss values from mlruns/ metric files."""
    losses = {}
    pattern = os.path.join(PROJECT_ROOT, "mlruns", "*", "*", "metrics", "train_loss")
    for metric_file in glob.glob(pattern):
        try:
            # The params folder is a sibling of metrics
            params_dir = os.path.join(os.path.dirname(os.path.dirname(metric_file)), "params")
            round_file = os.path.join(params_dir, "round")
            if not os.path.exists(round_file):
                continue
            with open(round_file) as f:
                round_num = int(f.read().strip())
            with open(metric_file) as f:
                last_line = f.readlines()[-1]
                loss = float(last_line.split()[1])
            losses[round_num] = loss
        except Exception:
            continue
    return losses


@app.get("/api/data")
def get_data():
    manifest = load_manifest()
    losses   = load_mlflow_losses()
    versions = manifest.get("versions", {})

    # Enrich versions with loss data
    for ver, info in versions.items():
        round_num = info.get("round", 0)
        info["train_loss"] = losses.get(round_num)

    return {
        "manifest":           manifest,
        "production_version": manifest.get("production_version"),
        "current_version":    manifest.get("current_version"),
    }


@app.get("/api/serve-status")
def get_serve_status():
    """Return current status of the serve.py process."""
    global serve_process, serve_status
    if serve_process and serve_process.poll() is None:
        serve_status["state"] = "running"
    elif serve_status["state"] == "running":
        serve_status["state"] = "stopped"
    return serve_status


@app.post("/api/deploy/{version}")
def deploy_version(version: str):
    """Promote version to production AND restart serve.py."""
    try:
        from versioning import promote_to_production, load_manifest as lm
        m    = lm(PROJECT_ROOT)
        info = m.get("versions", {}).get(version)
        if not info:
            return {"ok": False, "error": f"Version {version} not found"}
        score = info.get("score", 0.0) or 0.0
        promote_to_production(PROJECT_ROOT, version, score)
        restart_serve_async(version)
        return {"ok": True, "message": f"v{version} promoted — restarting serve.py"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/restart-serve")
def restart_serve_endpoint():
    """Restart serve.py without changing the production version."""
    try:
        manifest = load_manifest()
        prod_ver = manifest.get("production_version", "unknown")
        restart_serve_async(prod_ver)
        return {"ok": True, "message": "Restarting serve.py"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Knowledge Trainer — Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0d0f12;
    --surface:  #13161b;
    --surface2: #1a1e25;
    --border:   rgba(255,255,255,0.07);
    --border2:  rgba(255,255,255,0.12);
    --text:     #e8eaf0;
    --muted:    #6b7280;
    --green:    #34d399;
    --blue:     #60a5fa;
    --amber:    #fbbf24;
    --red:      #f87171;
    --font:     'DM Sans', sans-serif;
    --mono:     'DM Mono', monospace;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 14px;
    min-height: 100vh;
    padding: 2rem;
  }

  header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }

  header h1 {
    font-size: 18px;
    font-weight: 500;
    letter-spacing: -0.3px;
  }

  header .subtitle {
    font-size: 13px;
    color: var(--muted);
    font-family: var(--mono);
  }

  .header-actions { margin-left: auto; display: flex; align-items: center; gap: 10px; }
  .serve-status {
    font-size: 12px; font-family: var(--mono); color: var(--muted);
    display: flex; align-items: center; gap: 6px;
    padding: 6px 14px; border: 1px solid var(--border); border-radius: 6px;
  }
  .serve-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); flex-shrink: 0; }
  .serve-dot.running    { background: var(--green); animation: pulse 2s infinite; }
  .serve-dot.restarting { background: var(--amber); animation: pulse 0.6s infinite; }
  .serve-dot.stopped    { background: var(--red); }
  .icon-btn {
    background: transparent; border: 1px solid var(--border2); color: var(--muted);
    padding: 6px 14px; border-radius: 6px; cursor: pointer;
    font-size: 12px; font-family: var(--font); transition: all 0.15s;
  }
  .icon-btn:hover { border-color: var(--blue); color: var(--blue); }

  .metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 1.5rem;
  }

  .metric {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
  }

  .metric-label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
    font-family: var(--mono);
  }

  .metric-value {
    font-size: 28px;
    font-weight: 300;
    letter-spacing: -1px;
    font-family: var(--mono);
  }

  .metric-value.green { color: var(--green); }
  .metric-value.blue  { color: var(--blue);  }

  .charts {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 1.5rem;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem;
  }

  .card-title {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-family: var(--mono);
    margin-bottom: 1rem;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }

  th {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-family: var(--mono);
    font-weight: 400;
    text-align: left;
    padding: 0 16px 10px 0;
    border-bottom: 1px solid var(--border);
  }

  td {
    padding: 12px 16px 12px 0;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }

  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,0.02); }

  .version-tag {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 500;
  }

  .badge {
    display: inline-block;
    font-size: 11px;
    font-family: var(--mono);
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 500;
  }

  .badge-production { background: rgba(52,211,153,0.12); color: var(--green); border: 1px solid rgba(52,211,153,0.25); }
  .badge-staging    { background: rgba(251,191,36,0.12);  color: var(--amber); border: 1px solid rgba(251,191,36,0.25); }
  .badge-retired    { background: rgba(107,114,128,0.12); color: var(--muted); border: 1px solid rgba(107,114,128,0.2); }

  .score-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .score-bar-bg {
    width: 70px;
    height: 4px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px;
    overflow: hidden;
  }

  .score-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: var(--green);
    transition: width 0.6s ease;
  }

  .score-text {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--green);
    min-width: 36px;
  }

  .deploy-btn {
    font-size: 11px;
    font-family: var(--mono);
    padding: 5px 14px;
    border-radius: 6px;
    border: 1px solid var(--border2);
    background: transparent;
    color: var(--text);
    cursor: pointer;
    transition: all 0.15s;
  }

  .deploy-btn:hover { border-color: var(--green); color: var(--green); }
  .deploy-btn:disabled { opacity: 0.3; cursor: default; }

  .live-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    background: var(--green);
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
  }

  .pages-list {
    color: var(--muted);
    font-size: 12px;
  }

  .empty-state {
    text-align: center;
    padding: 3rem;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 13px;
  }

  .toast {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 8px;
    padding: 12px 20px;
    font-size: 13px;
    font-family: var(--mono);
    opacity: 0;
    transform: translateY(10px);
    transition: all 0.3s;
    pointer-events: none;
    z-index: 100;
  }

  .toast.show { opacity: 1; transform: translateY(0); }
  .toast.success { border-color: rgba(52,211,153,0.4); color: var(--green); }
  .toast.error   { border-color: rgba(248,113,113,0.4); color: var(--red); }
</style>
</head>
<body>

<header>
  <h1>Knowledge Trainer</h1>
  <span class="subtitle">mlops dashboard</span>
  <div class="header-actions">
    <div class="serve-status">
      <span class="serve-dot" id="serve-dot"></span>
      <span id="serve-label">checking serve.py...</span>
    </div>
    <button class="icon-btn" onclick="restartServe()">↺ restart serve</button>
    <button class="icon-btn" onclick="loadData()">↻ refresh</button>
  </div>
</header>

<div class="metrics">
  <div class="metric">
    <div class="metric-label">Production</div>
    <div class="metric-value blue" id="m-prod">—</div>
  </div>
  <div class="metric">
    <div class="metric-label">Best score</div>
    <div class="metric-value green" id="m-score">—</div>
  </div>
  <div class="metric">
    <div class="metric-label">Versions</div>
    <div class="metric-value" id="m-versions">—</div>
  </div>
  <div class="metric">
    <div class="metric-label">Pages ingested</div>
    <div class="metric-value" id="m-pages">—</div>
  </div>
</div>

<div class="charts">
  <div class="card">
    <div class="card-title">Score by version</div>
    <div style="position:relative; height:180px;">
      <canvas id="scoreChart"></canvas>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Training loss by version</div>
    <div style="position:relative; height:180px;">
      <canvas id="lossChart"></canvas>
    </div>
  </div>
</div>

<div class="card">
  <div class="card-title">Version registry</div>
  <table>
    <thead>
      <tr>
        <th>Version</th>
        <th>Status</th>
        <th>Score</th>
        <th>Loss</th>
        <th>Pages</th>
        <th>Round</th>
        <th>Created</th>
        <th>Action</th>
      </tr>
    </thead>
    <tbody id="vtable"></tbody>
  </table>
</div>

<div class="toast" id="toast"></div>

<script>
let scoreChart, lossChart;

function showToast(msg, type='success') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = `toast show ${type}`;
  setTimeout(() => t.className = 'toast', 3000);
}

async function loadData() {
  try {
    const res  = await fetch('/api/data');
    const data = await res.json();
    render(data);
  } catch(e) {
    showToast('Failed to load data', 'error');
  }
}

function render(data) {
  const versions = Object.values(data.manifest.versions || {});
  const sorted   = versions.sort((a,b) => a.version.localeCompare(b.version));
  const prod     = versions.find(v => v.status === 'production');
  const allPages = [...new Set(versions.flatMap(v => v.pages || []))];
  const bestScore = Math.max(...versions.filter(v => v.score != null).map(v => v.score), 0);

  document.getElementById('m-prod').textContent     = prod ? `v${prod.version}` : '—';
  document.getElementById('m-score').textContent    = bestScore > 0 ? Math.round(bestScore * 100) + '%' : '—';
  document.getElementById('m-versions').textContent = versions.length;
  document.getElementById('m-pages').textContent    = allPages.length;

  renderCharts(sorted);
  renderTable(sorted);
}

function renderCharts(sorted) {
  const labels = sorted.map(v => `v${v.version}`);
  const scores = sorted.map(v => v.score != null ? Math.round(v.score * 100) : null);
  const losses = sorted.map(v => v.train_loss != null ? parseFloat(v.train_loss.toFixed(3)) : null);

  const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#6b7280', font: { family: 'DM Mono', size: 11 } } },
      y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#6b7280', font: { family: 'DM Mono', size: 11 } } }
    }
  };

  if (scoreChart) scoreChart.destroy();
  if (lossChart)  lossChart.destroy();

  scoreChart = new Chart(document.getElementById('scoreChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: scores,
        borderColor: '#34d399',
        backgroundColor: 'rgba(52,211,153,0.06)',
        borderWidth: 2,
        pointBackgroundColor: '#34d399',
        pointRadius: 5,
        tension: 0.35,
        fill: true,
        spanGaps: true,
      }]
    },
    options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, min: 0, max: 100, ticks: { ...chartDefaults.scales.y.ticks, callback: v => v + '%' } } } }
  });

  lossChart = new Chart(document.getElementById('lossChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: losses,
        borderColor: '#60a5fa',
        backgroundColor: 'rgba(96,165,250,0.06)',
        borderWidth: 2,
        pointBackgroundColor: '#60a5fa',
        pointRadius: 5,
        tension: 0.35,
        fill: true,
        spanGaps: true,
      }]
    },
    options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, ticks: { ...chartDefaults.scales.y.ticks, callback: v => v.toFixed(2) } } } }
  });
}

function renderTable(sorted) {
  const tbody = document.getElementById('vtable');

  if (!sorted.length) {
    tbody.innerHTML = `<tr><td colspan="8"><div class="empty-state">no versions registered yet — run the pipeline to get started</div></td></tr>`;
    return;
  }

  tbody.innerHTML = sorted.slice().reverse().map(v => {
    const scoreHtml = v.score != null
      ? `<div class="score-wrap"><span class="score-text">${Math.round(v.score*100)}%</span><div class="score-bar-bg"><div class="score-bar-fill" style="width:${Math.round(v.score*100)}%"></div></div></div>`
      : `<span style="color:var(--muted)">—</span>`;

    const lossHtml = v.train_loss != null
      ? `<span style="font-family:var(--mono); font-size:12px; color:var(--blue)">${parseFloat(v.train_loss).toFixed(3)}</span>`
      : `<span style="color:var(--muted)">—</span>`;

    const pages = (v.pages || []).length ? v.pages.join(', ') : '<span style="color:var(--muted)">none (baseline)</span>';
    const created = (v.created_at || '').slice(0, 10) || '—';

    let badge, action;
    if (v.status === 'production') {
      badge  = `<span class="badge badge-production"><span class="live-dot"></span>production</span>`;
      action = `<button class="deploy-btn" disabled>live</button>`;
    } else if (v.status === 'staging') {
      badge  = `<span class="badge badge-staging">staging</span>`;
      action = `<button class="deploy-btn" onclick="deploy('${v.version}')">deploy ↑</button>`;
    } else {
      badge  = `<span class="badge badge-retired">retired</span>`;
      action = `<button class="deploy-btn" disabled>retired</button>`;
    }

    return `<tr>
      <td><span class="version-tag">v${v.version}</span></td>
      <td>${badge}</td>
      <td>${scoreHtml}</td>
      <td>${lossHtml}</td>
      <td class="pages-list">${pages}</td>
      <td style="color:var(--muted); font-family:var(--mono)">round ${v.round}</td>
      <td style="color:var(--muted); font-family:var(--mono); font-size:12px">${created}</td>
      <td>${action}</td>
    </tr>`;
  }).join('');
}

async function deploy(version) {
  try {
    const res  = await fetch(`/api/deploy/${version}`, { method: 'POST' });
    const data = await res.json();
    if (data.ok) {
      showToast(`v${version} promoted to production`, 'success');
      loadData();
    } else {
      showToast(data.error || 'Deploy failed', 'error');
    }
  } catch(e) {
    showToast('Deploy failed', 'error');
  }
}

async function loadServeStatus() {
  try {
    const res  = await fetch('/api/serve-status');
    const data = await res.json();
    const dot   = document.getElementById('serve-dot');
    const label = document.getElementById('serve-label');
    dot.className = `serve-dot ${data.state || 'unknown'}`;
    if (data.state === 'running') {
      label.textContent = `serve.py running · v${data.version || '?'} · pid ${data.pid || '?'}`;
    } else if (data.state === 'restarting') {
      label.textContent = 'serve.py restarting...';
    } else {
      label.textContent = 'serve.py stopped';
    }
  } catch(e) {
    document.getElementById('serve-label').textContent = 'serve.py unknown';
  }
}

async function restartServe() {
  try {
    const res  = await fetch('/api/restart-serve', { method: 'POST' });
    const data = await res.json();
    if (data.ok) showToast('Restarting serve.py...', 'success');
    else showToast(data.error || 'Restart failed', 'error');
  } catch(e) {
    showToast('Restart failed', 'error');
  }
}

loadData();
loadServeStatus();
setInterval(loadData, 10000);
setInterval(loadServeStatus, 3000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("\n── Knowledge Trainer Dashboard ───────────────────────")
    print("   Open: http://localhost:8001")
    print("   Auto-refreshes every 10 seconds")
    print("──────────────────────────────────────────────────────\n")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")