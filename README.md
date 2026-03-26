# Knowledge Trainer

A local MLOps project that teaches a language model new knowledge through iterative fine-tuning,
and automatically tests what it learned using AI-generated questions.

---

## What this project does

Most machine learning demos train a model once and call it done. This project is different —
it models the real MLOps workflow where a model continuously improves over time as new data arrives.

The loop works like this:

1. You provide a Wikipedia page URL on any topic
2. A local LLM (the "trainer") reads the page and generates quiz questions from it
3. A student model (distilbert) is fine-tuned on the page text
4. The student is quizzed using the trainer's questions — you see a score
5. You add more pages, retrain, quiz again — the score climbs
6. When happy with the result, deploy to a local chat UI at http://localhost:8000/chat

Every training run is tracked in MLflow so you can compare rounds, see which pages helped most,
and roll back to any previous version of the model.

---

## Why we built it this way

### Two models with two jobs
The trainer model (Ollama / llama3.2:3b) is a general-purpose LLM that already knows how to
read and reason. We use it only to generate evaluation questions — not to answer them.

The student model (distilbert) starts knowing nothing about your chosen topic. Its only job is
to learn from the pages you feed it. This separation means the evaluation is always honest —
the trainer generates the questions before the student has seen the answers.

### Why distilbert
DistilBERT is a smaller, faster version of BERT — one of the most important language models ever
built. It runs on a normal laptop CPU, fine-tunes in minutes rather than hours, and is the
industry standard starting point for this kind of task. It already understands English grammar
and word relationships — we just teach it your specific domain on top.

### Why Ollama
Ollama is Docker for LLMs. You pull a model like a container image and run it locally with one
command. No API keys, no costs, no data leaving your machine. llama3.2:3b is small enough to
run on CPU but capable enough to generate meaningful questions from any text.

### Why MLflow
Every fine-tuning run is an experiment. MLflow tracks the parameters (which pages, how many
epochs, what batch size), the metrics (quiz score, training loss), and the model artifact for
every run. This means you can always answer: "which version of the model was trained on which
data, and how well did it perform?"

This is the core MLOps discipline — never lose track of what produced what.

### Why FastAPI for serving
FastAPI is the standard for ML model serving in Python — minimal boilerplate, automatic API
docs at /docs, and fast enough for local use. serve.py always loads the latest trained model
from the models/ directory, so retraining and restarting the server is all you need to upgrade.

---

## Architecture

```
You (Wikipedia URL)
        │
        ▼
   ingest.py          Fetches page text, cleans it, saves to data/pages/<title>.txt
        │
        ├─────────────────────────────┐
        ▼                             ▼
generate_quiz.py                 train.py
Sends text to Ollama             Fine-tunes distilbert on all pages in data/pages/
Ollama returns Q&A pairs         Tracked in MLflow (loss, epochs, pages, batch size)
Saved to data/quiz_bank.json     Model saved to models/round_N/
        │                             │
        └─────────────────────────────┘
                      │
                      ▼
               evaluate.py
        Loads model from models/round_N/
        Asks every question in quiz_bank.json
        Ollama judges if answers are correct
        Prints score: X / Y correct
        Logs score to MLflow
                      │
              ┌───────┴───────┐
              ▼               ▼
         deploy (y)        skip (n)
              │
              ▼
          serve.py
    FastAPI on :8000
    loads latest model
              │
              ▼
          chat.html
    http://localhost:8000/chat
    browser chat interface
```

---

## Project structure

```
knowledge-trainer/
├── data/
│   ├── pages/                  Raw Wikipedia text, one .txt file per page
│   ├── quiz_bank.json          Auto-generated Q&A pairs, grows with every new page
│   └── eval_round_N.json       Detailed evaluation results per round (auto-created)
├── models/
│   └── round_N/                Trained model saved here after each training run
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer files
├── src/
│   ├── ingest.py               Phase 1 — fetch a Wikipedia page, save to data/pages/
│   ├── generate_quiz.py        Phase 1 — send page to Ollama, get Q&A pairs back
│   ├── train.py                Phase 1 — fine-tune distilbert, track run in MLflow
│   ├── evaluate.py             Phase 1 — quiz the student model, print score
│   ├── serve.py                Phase 2 — FastAPI server, exposes /ask endpoint
│   ├── chat.html               Phase 2 — browser chat UI, no build step needed
│   └── pipeline.py             Phase 3 — full automated loop, one command to rule them all
├── mlruns/                     MLflow experiment tracking (auto-created on first run)
├── venv/                       Python virtual environment (never commit this)
├── .gitignore                  See contents below
├── requirements.txt            Locked dependency versions
└── README.md                   This file
```

### .gitignore contents
```
venv/
mlruns/
__pycache__/
*.pyc
.env
data/pages/
models/
data/eval_round_*.json
```

Note: `data/quiz_bank.json` IS committed — it represents accumulated knowledge about
what questions have been generated and should be preserved across machines.
`data/pages/` and `models/` are NOT committed — they are always reproducible by
running ingest.py and train.py again.

---

## Prerequisites

Install these before anything else.

### 1. Python 3.11+
Download from https://www.python.org/downloads/

On Windows — critical: check "Add python.exe to PATH" during install.

Verify:
```powershell
python --version
```

### 2. Ollama
Ollama runs the local trainer LLM (llama3.2:3b). It is NOT a Python package —
it is a standalone application that must be installed separately.

Download from https://ollama.com/download and run the Windows installer.

After install, pull the model (about 2GB — like docker pull):
```powershell
ollama pull llama3.2:3b
```

Verify:
```powershell
ollama list
# should show: llama3.2:3b
```

---

## Environment setup (do this once)

### 1. Clone the repo
```powershell
git clone https://github.com/WeissShaharIL/ml-01_knowledge-trainer.git
cd ml-01_knowledge-trainer
```

### 2. Create and activate virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you get a permissions error on Activate.ps1:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try activating again. You should see `(venv)` in your prompt.

### 3. Install Python dependencies
```powershell
pip install -r requirements.txt
```

Or install fresh and freeze:
```powershell
pip install transformers torch datasets requests mlflow fastapi uvicorn
pip freeze > requirements.txt
```

### 4. Create required directories
```powershell
mkdir data
mkdir data\pages
```

### 5. Verify everything works
```powershell
python -c "import transformers, mlflow, torch, datasets, fastapi, uvicorn; print('all good')"
ollama list
```

Both commands should succeed before running anything else.

---

## Running the project

There are two ways to run this project:

- **Manual** — run each script individually (good for learning and debugging)
- **Pipeline** — one command does everything (good for adding new pages quickly)

---

### Manual flow (Phase 1)

Run these four commands in order for each new Wikipedia page.

#### Step 1 — Ingest a Wikipedia page
```powershell
python src/ingest.py --url "https://en.wikipedia.org/wiki/Alan_Turing"
```
Fetches the page, cleans it, saves to `data/pages/Alan_Turing.txt`.

#### Step 2 — Generate quiz questions
```powershell
python src/generate_quiz.py
```
Ollama reads every new page in `data/pages/` and generates 5 questions per page.
Questions are saved to `data/quiz_bank.json`. Skips pages already in the quiz bank.

To regenerate questions for all pages:
```powershell
python src/generate_quiz.py --regenerate
```

#### Step 3 — Train the model
```powershell
python src/train.py --round 1
```
Fine-tunes distilbert on all pages in `data/pages/`. Saves model to `models/round_1/`.
Tracks the run in MLflow. Use `--round 2`, `--round 3` etc. for subsequent rounds.

Optional flags:
```powershell
python src/train.py --round 1 --epochs 5 --batch-size 4
```

#### Step 4 — Evaluate
```powershell
python src/evaluate.py --model-dir models/round_1 --round 1
```
Quizzes the model using all questions in `quiz_bank.json`.
Ollama judges whether each answer is correct.
Prints score and logs to MLflow.

---

### Manual flow (Phase 2) — start the chat server

After at least one training run:
```powershell
python src/serve.py
```

Then open in your browser:
```
http://localhost:8000/chat
```

The server loads the latest model from `models/` automatically.
When you retrain, just restart the server — it picks up the new model.

API docs available at:
```
http://localhost:8000/docs
```

Call the API directly:
```powershell
curl -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"question": "Who was Alan Turing?"}'
```

---

### Automated pipeline (Phase 3) — one command does everything

```powershell
python src/pipeline.py --url "https://en.wikipedia.org/wiki/Marie_Curie" --round 2
```

This runs all four Phase 1 steps automatically, shows you a score comparison
vs the previous round, then asks if you want to deploy:

```
── Results ───────────────────────────────────────────
Previous score:  3 / 5   (60%)
New score:       5 / 10  (50%)
Improvement:     -10%

Deploy new model to production? (y/n):
```

If you type `y` — the server restarts with the new model automatically.
If you type `n` — the model is saved in MLflow but the running server is unchanged.

Optional flags:
```powershell
python src/pipeline.py \
  --url "https://en.wikipedia.org/wiki/Marie_Curie" \
  --round 2 \
  --epochs 5 \
  --questions-per-page 10
```

---

### View experiment history in MLflow
```powershell
python -m mlflow ui
```
Open http://localhost:5000 — shows all training and evaluation runs side by side.

---

## What good progress looks like

| Round | Pages fed | Expected quiz score |
|-------|-----------|-------------------|
| 1 | 1 page | 2–4/10 — starting to pick up key facts |
| 2 | 3 pages | 4–6/10 — solid on core concepts |
| 3 | 5 pages | 6–8/10 — strong domain knowledge |
| 4 | 10+ pages | 8–10/10 — comprehensive coverage |

Scores vary depending on question difficulty and page coverage.
The trend across rounds matters more than any single score.

---

## Key concepts learned by building this

| Concept | Where you see it |
|---------|-----------------|
| Transfer learning | distilbert starts pretrained, we fine-tune on top |
| Synthetic evaluation | Ollama generates tests automatically from new data |
| Experiment tracking | MLflow logs every run with params, metrics, artifacts |
| Iterative improvement | Score climbs as you add more pages |
| Local LLM serving | Ollama runs llama3.2 entirely on your machine |
| Model serving | FastAPI exposes the model as a REST API |
| Human approval gate | pipeline.py asks before deploying a new model version |
| Registry-based deployment | serve.py loads by directory name, not hardcoded path |

---

## Decisions log

| Decision | Why |
|----------|-----|
| distilbert over larger models | Runs on CPU, fine-tunes in minutes, good enough for this task |
| Ollama over API | No cost, no key management, no data leaving the machine |
| llama3.2:3b over larger | Fast enough for question generation, small enough for CPU |
| MLflow over custom logging | Industry standard, free, runs locally, visual UI |
| One .txt file per page | Simple, auditable, easy to remove a bad page and retrain |
| quiz_bank.json grows over time | Questions accumulate — later rounds are harder and more comprehensive |
| FastAPI over Flask | Cleaner syntax, automatic /docs, better for ML APIs |
| subprocess in pipeline.py | Each script stays independently runnable and testable |
| models/ not committed to git | Always reproducible by rerunning train.py |
| data/pages/ not committed | Always reproducible by rerunning ingest.py |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `python` not recognized | Check PATH — reinstall Python with "Add to PATH" checked |
| `Activate.ps1` blocked | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `ollama` not recognized | Restart PowerShell after Ollama install so PATH updates |
| Ollama not responding | Make sure Ollama app is running (check system tray) |
| `No trained model found` | Run train.py before serve.py |
| MLflow UI blank | Run at least one training run first |
| Low quiz score | Expected on Round 1 — add more pages and retrain |