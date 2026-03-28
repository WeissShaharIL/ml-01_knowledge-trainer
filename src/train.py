import glob
import os
import sys
import json
import re
import mlflow
import argparse
import torch
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
from rank_bm25 import BM25Okapi

# ── train.py ───────────────────────────────────────────────────────────────────
# RAG mode — base model is frozen, no weights are updated.
# "Training" benchmarks BM25 retrieval quality on the current quiz bank
# and logs metrics to MLflow.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

MLFLOW_DIR = os.path.join(PROJECT_ROOT, "mlruns")
mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")
mlflow.set_experiment("knowledge-trainer")


def load_pages(pages_dir: str) -> list[dict]:
    pattern = os.path.join(pages_dir, "*.txt")
    pages   = []
    for filepath in sorted(glob.glob(pattern)):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        title = os.path.basename(filepath).replace(".txt", "").replace("_", " ")
        pages.append({"title": title, "text": text})
    return pages


def find_best_context(question: str, pages: list[dict], chunk_words: int = 300) -> str:
    """BM25 retrieval over all chunks from all pages."""
    if not pages:
        return ""

    stop_words = {"what", "is", "the", "a", "an", "of", "in", "was", "were",
                  "how", "when", "where", "who", "why", "did", "do", "does",
                  "that", "this", "which", "by", "at", "from", "with", "and",
                  "or", "to", "for", "on", "are", "has", "had", "have", "be"}

    chunks = []
    for page in pages:
        words = page["text"].split()
        step  = chunk_words // 2
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i : i + chunk_words]))

    if not chunks:
        return ""

    tokenized    = [[w for w in c.lower().split() if w not in stop_words] for c in chunks]
    query_tokens = [w for w in question.lower().split() if w not in stop_words]

    if not query_tokens:
        return chunks[0]

    bm25     = BM25Okapi(tokenized)
    scores   = bm25.get_scores(query_tokens)
    best_idx = int(scores.argmax())
    return chunks[best_idx]


def benchmark_retrieval(pages: list[dict], quiz_bank: list[dict]) -> dict:
    """Check if correct answer appears in BM25-retrieved context."""
    if not quiz_bank:
        return {"retrieval_hit_rate": 0.0, "questions": 0}

    hits = 0
    for item in quiz_bank:
        question       = item["question"]
        correct_answer = item["answer"].lower()
        context        = find_best_context(question, pages)

        if not context:
            continue

        context = re.sub(r'TITLE:.*?(?=\w{4})', '', context, flags=re.DOTALL)
        context = re.sub(r'=+\s*', ' ', context).strip()

        if correct_answer in context.lower():
            hits += 1

    hit_rate = hits / len(quiz_bank)
    return {"retrieval_hit_rate": hit_rate, "questions": len(quiz_bank)}


def main(round_num: int, epochs: int = None, batch_size: int = None):
    pages_dir = os.path.join(PROJECT_ROOT, "data", "pages")
    quiz_path = os.path.join(PROJECT_ROOT, "data", "quiz_bank.json")
    pages     = load_pages(pages_dir)

    base_dir = os.path.join(PROJECT_ROOT, "models", "base")
    if not os.path.exists(base_dir):
        raise FileNotFoundError("models/base not found. Run: python src/setup.py")

    print(f"\nRound {round_num} — RAG mode (frozen base model, BM25 retrieval)")
    print(f"Pages loaded: {len(pages)}")
    for p in pages:
        print(f"  {p['title']}")

    quiz_bank = []
    if os.path.exists(quiz_path):
        with open(quiz_path, "r", encoding="utf-8") as f:
            quiz_bank = json.load(f)

    print(f"\nBenchmarking BM25 retrieval on {len(quiz_bank)} questions...")
    metrics  = benchmark_retrieval(pages, quiz_bank)
    hit_rate = metrics["retrieval_hit_rate"]
    hits     = int(hit_rate * len(quiz_bank))
    print(f"Retrieval hit rate: {hit_rate:.0%} ({hits}/{len(quiz_bank)} answers found in context)")

    run_name = f"round-{round_num}-{len(pages)}pages-bm25"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("round",      round_num)
        mlflow.log_param("pages",      len(pages))
        mlflow.log_param("retrieval",  "bm25")
        mlflow.log_param("questions",  len(quiz_bank))
        mlflow.log_metric("train_loss",         0.0)
        mlflow.log_metric("retrieval_hit_rate", hit_rate)

    print(f"MLflow run logged: {run_name}")
    print(f"Model: models/base (frozen)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round",      type=int, default=0)
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    main(args.round, args.epochs, args.batch_size)