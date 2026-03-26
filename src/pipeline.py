import argparse
import json
import os
import subprocess
import sys

# ── pipeline.py ────────────────────────────────────────────────────────────────
# Phase 3 — Full retraining pipeline.
# One command triggers the complete loop:
#   ingest → generate questions → train → evaluate → show results → deploy?
#
# Why use subprocess to call the other scripts instead of importing them?
# Same reason a Makefile calls separate scripts rather than inlining everything.
# Each script is independently runnable and testable. The pipeline just
# orchestrates them in order — it doesn't own their logic.
# This also means if train.py fails, the pipeline stops cleanly at that step
# and you can rerun from there without starting over.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON       = sys.executable  # use the same python that's running this script

def run_step(step_name: str, cmd: list) -> bool:
    """Run a pipeline step. Returns True if successful, False if it failed."""
    print(f"\n{'─'*50}")
    print(f"  {step_name}")
    print(f"{'─'*50}")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"\nPipeline stopped — {step_name} failed.")
        print("Fix the error above and rerun.")
        return False
    return True


def get_current_score(round_num: int) -> float | None:
    """Load the score from the previous evaluation run if it exists."""
    path = os.path.join(PROJECT_ROOT, "data", f"eval_round_{round_num - 1}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    correct = sum(1 for r in results if r["is_correct"])
    return correct / len(results) if results else None


def get_new_score(round_num: int) -> tuple[int, int] | None:
    """Load the score from the current evaluation run."""
    path = os.path.join(PROJECT_ROOT, "data", f"eval_round_{round_num}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    correct = sum(1 for r in results if r["is_correct"])
    return correct, len(results)


def restart_server():
    """
    Restart serve.py so it picks up the newly trained model.
    Kills any existing serve.py process and starts a new one in the background.
    """
    print("\nRestarting model server...")

    # Kill existing server if running
    subprocess.run(
        ["powershell", "-Command",
         "Get-Process python | Where-Object {$_.MainWindowTitle -eq ''} | Stop-Process -Force"],
        capture_output=True
    )

    # Start new server in background
    serve_script = os.path.join(PROJECT_ROOT, "src", "serve.py")
    subprocess.Popen(
        [PYTHON, serve_script],
        cwd=PROJECT_ROOT,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    print("Server restarted. Open http://localhost:8000/chat in your browser.")


def main():
    parser = argparse.ArgumentParser(
        description="Full retraining pipeline — ingest → quiz → train → evaluate → deploy"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help='Wikipedia URL to learn from e.g. "https://en.wikipedia.org/wiki/Alan_Turing"'
    )
    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="Round number (1, 2, 3...)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs (default: 3)"
    )
    parser.add_argument(
        "--questions-per-page",
        type=int,
        default=5,
        help="Questions Ollama generates per page (default: 5)"
    )
    args = parser.parse_args()

    model_dir = f"models/round_{args.round}"

    print(f"\n{'='*50}")
    print(f"  Knowledge Trainer — Round {args.round}")
    print(f"  Page: {args.url}")
    print(f"{'='*50}")

    # ── Step 1: Ingest ─────────────────────────────────────────────────────────
    ok = run_step("Step 1/4  Ingesting Wikipedia page", [
        PYTHON, "src/ingest.py",
        "--url", args.url,
    ])
    if not ok: return

    # ── Step 2: Generate quiz questions ───────────────────────────────────────
    ok = run_step("Step 2/4  Generating quiz questions (Ollama)", [
        PYTHON, "src/generate_quiz.py",
        "--questions-per-page", str(args.questions_per_page),
    ])
    if not ok: return

    # ── Step 3: Train ──────────────────────────────────────────────────────────
    ok = run_step("Step 3/4  Fine-tuning student model", [
        PYTHON, "src/train.py",
        "--round",  str(args.round),
        "--epochs", str(args.epochs),
    ])
    if not ok: return

    # ── Step 4: Evaluate ───────────────────────────────────────────────────────
    ok = run_step("Step 4/4  Evaluating model", [
        PYTHON, "src/evaluate.py",
        "--model-dir", model_dir,
        "--round",     str(args.round),
    ])
    if not ok: return

    # ── Results summary ────────────────────────────────────────────────────────
    prev_score = get_current_score(args.round)
    new_result = get_new_score(args.round)

    print(f"\n{'='*50}")
    print(f"  Results — Round {args.round}")
    print(f"{'='*50}")

    if new_result:
        correct, total = new_result
        pct = correct / total
        print(f"  New score:  {correct}/{total}  ({pct:.0%})")

        if prev_score is not None:
            delta = pct - prev_score
            sign  = "+" if delta >= 0 else ""
            print(f"  Previous:   {prev_score:.0%}")
            print(f"  Change:     {sign}{delta:.0%}")

    print(f"{'='*50}\n")

    # ── Deploy decision ────────────────────────────────────────────────────────
    print("Deploy this model to the chat server? (y/n): ", end="", flush=True)
    choice = input().strip().lower()

    if choice == "y":
        restart_server()
        print("\nDeployed. Chat at http://localhost:8000/chat")
    else:
        print(f"\nNot deployed. Model saved at {model_dir}")
        print("You can deploy manually later with: python src/serve.py")

    print("\nDone. Run MLflow UI to see all rounds:")
    print("  python -m mlflow ui")
    print("  http://localhost:5000")


if __name__ == "__main__":
    main()
