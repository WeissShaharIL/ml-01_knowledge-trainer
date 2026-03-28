import argparse
import json
import os
import subprocess
import sys

# ── pipeline.py ────────────────────────────────────────────────────────────────
# Full automated pipeline: ingest → quiz → train → evaluate → version → deploy
#
# New in this version:
# - Semantic versioning (1.0.0, 1.1.0, 1.2.0 ...)
# - manifest.json tracks every version with pages, scores, and status
# - Evaluation gate: only promotes to production if score improved
# - If score is equal, asks the user to decide

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")

# Add src/ to path so we can import versioning
sys.path.insert(0, SRC_DIR)
from versioning import (
    load_manifest,
    get_next_version,
    register_version,
    promote_to_production,
    get_production_version,
    print_manifest_summary,
)


def run_step(step_name: str, cmd: list[str]) -> bool:
    """Run a subprocess step. Returns True if successful."""
    width = 50
    print(f"\n{'─' * width}")
    print(f"  {step_name}")
    print(f"{'─' * width}")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\nPipeline stopped — {step_name} failed.")
        print("Fix the error above and rerun.")
        return False
    return True


def read_score(round_num: int) -> float | None:
    """Read the evaluation score from the saved results JSON."""
    results_path = os.path.join(PROJECT_ROOT, "data", f"eval_round_{round_num}.json")
    if not os.path.exists(results_path):
        return None

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        return 0.0

    correct = sum(1 for r in results if r.get("is_correct"))
    return correct / len(results)


def read_ingest_log() -> dict:
    """Read the ingest log to get page metadata."""
    log_path = os.path.join(PROJECT_ROOT, "data", "ingest_log.json")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def restart_server():
    """Kill any running serve.py and remind user to restart."""
    print("\n── Deployment ────────────────────────────────────────────")
    print("  New production model is ready.")
    print("  Restart the chat server to use it:")
    print("  → Stop serve.py (Ctrl+C in its terminal)")
    print("  → Run: python src/serve.py")
    print("──────────────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Trainer Pipeline")
    parser.add_argument("--url",   type=str, required=True,  help="Wikipedia URL to ingest")
    parser.add_argument("--round", type=int, required=True,  help="Round number")
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--questions",   type=int, default=5)
    args = parser.parse_args()

    # ── Determine next version ─────────────────────────────────────────────────
    manifest     = load_manifest(PROJECT_ROOT)
    next_version = get_next_version(manifest)
    model_dir    = f"models/round_{args.round}"

    print(f"\n{'=' * 50}")
    print(f"  Knowledge Trainer — Round {args.round}")
    print(f"  Version: {next_version}")
    print(f"  Page: {args.url}")
    print(f"{'=' * 50}")

    # Show current registry state
    print_manifest_summary(PROJECT_ROOT)

    # ── Step 1: Ingest ─────────────────────────────────────────────────────────
    ok = run_step("Step 1/4  Ingesting Wikipedia page", [
        sys.executable, os.path.join(SRC_DIR, "ingest.py"),
        "--url", args.url,
    ])
    if not ok:
        return

    # ── Step 2: Generate quiz ──────────────────────────────────────────────────
    ok = run_step("Step 2/4  Generating quiz questions (Ollama)", [
        sys.executable, os.path.join(SRC_DIR, "generate_quiz.py"),
        "--questions-per-page", str(args.questions),
    ])
    if not ok:
        return

    # ── Step 3: Train ──────────────────────────────────────────────────────────
    ok = run_step("Step 3/4  Fine-tuning student model", [
        sys.executable, os.path.join(SRC_DIR, "train.py"),
        "--round",      str(args.round),
        "--epochs",     str(args.epochs),
        "--batch-size", str(args.batch_size),
    ])
    if not ok:
        return

    # ── Step 4: Evaluate ───────────────────────────────────────────────────────
    ok = run_step("Step 4/4  Evaluating student model", [
        sys.executable, os.path.join(SRC_DIR, "evaluate.py"),
        "--model-dir", model_dir,
        "--round",     str(args.round),
    ])
    if not ok:
        return

    # ── Read score ─────────────────────────────────────────────────────────────
    new_score = read_score(args.round)
    if new_score is None:
        print("Could not read evaluation score. Check data/eval_round_N.json")
        return

    # ── Read page metadata from ingest log ─────────────────────────────────────
    ingest_log  = read_ingest_log()
    page_titles = list(ingest_log.keys())
    page_urls   = [ingest_log[t]["url"] for t in page_titles]
    word_counts = {t: ingest_log[t]["word_count"] for t in page_titles}

    # ── Register new version as staging ───────────────────────────────────────
    register_version(
        project_root = PROJECT_ROOT,
        version      = next_version,
        round_num    = args.round,
        pages        = page_titles,
        urls         = page_urls,
        word_counts  = word_counts,
        model_dir    = model_dir,
        score        = new_score,
        status       = "staging",
    )

    # ── Results summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"  Results — v{next_version}")
    print(f"  New score:  {new_score:.0%}  ({int(new_score * 5)}/5)")
    print(f"{'=' * 50}")

    # ── Evaluation gate ────────────────────────────────────────────────────────
    production = get_production_version(PROJECT_ROOT)

    if production is None:
        # No production version yet — always promote first version
        print(f"\nFirst version — promoting v{next_version} to production automatically.")
        promote_to_production(PROJECT_ROOT, next_version, new_score)
        restart_server()

    else:
        old_score   = production.get("score", 0.0) or 0.0
        old_version = production.get("version")

        print(f"\n  Previous production: v{old_version}  score={old_score:.0%}")
        print(f"  New staging:         v{next_version}  score={new_score:.0%}")

        if new_score > old_score:
            print(f"\n✅ Score improved ({old_score:.0%} → {new_score:.0%})")
            print(f"   Promoting v{next_version} to production.")
            promote_to_production(PROJECT_ROOT, next_version, new_score)
            restart_server()

        elif new_score < old_score:
            print(f"\n❌ Score dropped ({old_score:.0%} → {new_score:.0%})")
            print(f"   Keeping v{old_version} as production.")
            print(f"   v{next_version} saved as staging — you can inspect it.")

        else:
            # Equal scores — ask the user
            print(f"\n⚠️  Score unchanged ({old_score:.0%} → {new_score:.0%})")
            answer = input(f"   Promote v{next_version} to production anyway? (y/n): ").strip().lower()
            if answer == "y":
                promote_to_production(PROJECT_ROOT, next_version, new_score)
                print(f"   ✅ v{next_version} promoted to production.")
                restart_server()
            else:
                print(f"   Keeping v{old_version} as production.")

    # ── Final registry summary ─────────────────────────────────────────────────
    print_manifest_summary(PROJECT_ROOT)


if __name__ == "__main__":
    main()
