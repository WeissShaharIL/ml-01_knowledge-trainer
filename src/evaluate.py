import argparse
import glob
import json
import os
import re
import requests
from rank_bm25 import BM25Okapi
import mlflow

# ── evaluate.py ────────────────────────────────────────────────────────────────
# Step 4 of the knowledge trainer pipeline.
# Uses BM25 to retrieve context, Ollama to generate answers,
# and Ollama again as a lenient judge.
# No DistilBERT — fully generative RAG pipeline.

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DIR   = os.path.join(PROJECT_ROOT, "mlruns")
mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")
mlflow.set_experiment("knowledge-trainer")


def call_ollama(prompt: str, timeout: int = 120) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["response"].strip()


def load_all_pages(pages_dir: str) -> list[dict]:
    pages   = []
    pattern = os.path.join(pages_dir, "*.txt")
    for filepath in sorted(glob.glob(pattern)):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        title = os.path.basename(filepath).replace(".txt", "").replace("_", " ")
        pages.append({"title": title, "text": text})
    return pages


def find_best_context(question: str, pages: list[dict], chunk_words: int = 300) -> str:
    """BM25 retrieval — returns top 3 chunks concatenated."""
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

    bm25        = BM25Okapi(tokenized)
    scores      = bm25.get_scores(query_tokens)
    top_indices = scores.argsort()[::-1][:3]
    return " ".join(chunks[i] for i in top_indices)


def clean_context(context: str) -> str:
    context = re.sub(r'TITLE:.*?(?=\w{4})', '', context, flags=re.DOTALL)
    context = re.sub(r'=+\s*', ' ', context).strip()
    return context


def get_student_answer(question: str, context: str) -> str:
    """Ask Ollama to generate an answer from the retrieved context."""
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."
Keep your answer concise — 1 to 2 sentences maximum.

Context:
{context}

Question: {question}

Answer:"""
    return call_ollama(prompt)


def judge_answer(question: str, correct_answer: str, student_answer: str) -> bool:
    """Ask Ollama to judge if the student answer is semantically correct."""
    prompt = f"""You are a lenient quiz grader. Judge if the student answer conveys the same meaning as the correct answer.

Question: {question}
Correct answer: {correct_answer}
Student answer: {student_answer}

Rules:
- Dates in different formats are correct (e.g. "26 March 1979" = "1979")
- Approximate equivalents are correct (e.g. "10th millennium BCE" = "c. 10,000 BCE")
- Partial answers containing the key fact are correct
- Extra words are fine as long as the core fact is there

Reply with only YES or NO. One word only."""
    return call_ollama(prompt).strip().upper().startswith("YES")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiz-bank",  type=str, default="data/quiz_bank.json")
    parser.add_argument("--pages-dir",  type=str, default="data/pages")
    parser.add_argument("--round",      type=int, default=1)
    parser.add_argument("--limit",      type=int, default=None)
    args = parser.parse_args()

    quiz_path = os.path.join(PROJECT_ROOT, args.quiz_bank)
    if not os.path.exists(quiz_path):
        print("quiz_bank.json not found. Run generate_quiz.py first.")
        return

    with open(quiz_path, "r", encoding="utf-8") as f:
        quiz_bank = json.load(f)

    if args.limit:
        quiz_bank = quiz_bank[:args.limit]

    print(f"Loaded {len(quiz_bank)} questions from quiz bank")

    pages_dir = os.path.join(PROJECT_ROOT, args.pages_dir)
    pages     = load_all_pages(pages_dir)
    print(f"Loaded {len(pages)} page(s) for retrieval")
    print(f"Reader: {OLLAMA_MODEL} (generative)")

    print(f"\nRunning evaluation...\n")
    print("=" * 60)

    correct = 0
    results = []

    for i, item in enumerate(quiz_bank):
        question       = item["question"]
        correct_answer = item["answer"]
        source         = item.get("source", "unknown")

        context = find_best_context(question, pages)
        if not context:
            print(f"[{i+1}/{len(quiz_bank)}] SKIP — no pages loaded")
            continue

        context_clean  = clean_context(context)
        student_answer = get_student_answer(question, context_clean)
        is_correct     = judge_answer(question, correct_answer, student_answer)

        if is_correct:
            correct += 1
            verdict = "CORRECT ✓"
        else:
            verdict = "WRONG ✗"

        print(f"[{i+1}/{len(quiz_bank)}] {verdict}")
        print(f"  Q: {question}")
        print(f"  A (correct): {correct_answer}")
        print(f"  A (student): {student_answer}")
        print(f"  Source: {source}")
        print()

        results.append({
            "question":       question,
            "correct_answer": correct_answer,
            "student_answer": student_answer,
            "is_correct":     is_correct,
            "source":         source,
        })

    total = len(quiz_bank)
    score = correct / total if total else 0

    print("=" * 60)
    print(f"\nFinal score: {correct}/{total}  ({score:.0%})")
    print()

    with mlflow.start_run(run_name=f"eval-round-{args.round}"):
        mlflow.log_param("round",     args.round)
        mlflow.log_param("retrieval", "bm25")
        mlflow.log_param("reader",    OLLAMA_MODEL)
        mlflow.log_param("questions", total)
        mlflow.log_metric("score",    score)
        mlflow.log_metric("correct",  correct)
        mlflow.log_metric("total",    total)

        results_path = os.path.join(PROJECT_ROOT, "data", f"eval_round_{args.round}.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(results_path)

    print(f"Results logged to MLflow")
    print(f"Detailed results saved to: data/eval_round_{args.round}.json")

    if score < 0.5:
        print("\nTip: score below 50% — try adding more pages")
    elif score < 0.8:
        print("\nTip: good progress! Add more pages or improve retrieval")
    else:
        print("\nExcellent! Strong coverage across the knowledge base")


if __name__ == "__main__":
    main()