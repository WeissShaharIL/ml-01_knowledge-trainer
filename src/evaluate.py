import argparse
import glob
import json
import os
import requests
import torch
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
import mlflow

# ── evaluate.py ────────────────────────────────────────────────────────────────
# Step 4 of the knowledge trainer pipeline.
# Loads the trained student model and quizzes it using questions from
# quiz_bank.json.
#
# How evaluation works (extractive QA):
# For each question we find the relevant Wikipedia page (by source tag),
# load it as context, and ask the QA model to extract the answer span.
# This is exactly how models like BERT are used in production QA systems.
#
# We then use Ollama as a semantic judge — exact string matching fails for
# natural language ("Jacob" vs "the patriarch Jacob" are the same answer).
# Ollama judges equivalence the way a human grader would.

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DIR   = os.path.join(PROJECT_ROOT, "mlruns")
mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")
mlflow.set_experiment("knowledge-trainer")


def call_ollama(prompt: str, timeout: int = 60) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["response"].strip()


def load_page_text(source: str, pages_dir: str) -> str:
    """Load the Wikipedia page text for a given source title."""
    # source is like "Israel" — match to Israel.txt
    pattern = os.path.join(pages_dir, "*.txt")
    for filepath in glob.glob(pattern):
        basename = os.path.basename(filepath).replace(".txt", "").replace("_", " ")
        if basename.lower() == source.lower():
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
    return ""


def get_student_answer(question: str, context: str, model, tokenizer) -> str:
    """
    Ask the QA model to extract an answer from the context passage.
    This is extractive QA — the model finds the answer span in the text.
    """
    # Truncate context to fit within model limits
    context_words = context.split()[:800]
    context = " ".join(context_words)

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the most likely start and end positions of the answer span
    start_idx = outputs.start_logits.argmax()
    end_idx   = outputs.end_logits.argmax()

    if end_idx < start_idx:
        end_idx = start_idx

    # Cap answer length to 20 tokens max
    if end_idx - start_idx > 20:
        end_idx = start_idx + 20

    answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    return answer if answer else "[no answer found]"


def judge_answer(question: str, correct_answer: str, student_answer: str) -> bool:
    """Ask Ollama to judge if the student answer is semantically correct."""
    prompt = f"""You are a quiz grader. Judge if the student answer is correct.

Question: {question}
Correct answer: {correct_answer}
Student answer: {student_answer}

Reply with only YES if the student answer is correct or contains the key information,
or NO if it is wrong or irrelevant. One word only."""

    verdict = call_ollama(prompt).strip().upper()
    return verdict.startswith("YES")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",  type=str, required=True)
    parser.add_argument("--quiz-bank",  type=str, default="data/quiz_bank.json")
    parser.add_argument("--pages-dir",  type=str, default="data/pages")
    parser.add_argument("--round",      type=int, default=1)
    parser.add_argument("--limit",      type=int, default=None)
    args = parser.parse_args()

    # Load quiz bank
    quiz_path = os.path.join(PROJECT_ROOT, args.quiz_bank)
    if not os.path.exists(quiz_path):
        print("quiz_bank.json not found. Run generate_quiz.py first.")
        return

    with open(quiz_path, "r", encoding="utf-8") as f:
        quiz_bank = json.load(f)

    if args.limit:
        quiz_bank = quiz_bank[:args.limit]

    print(f"Loaded {len(quiz_bank)} questions from quiz bank")

    # Load student model
    model_dir = os.path.join(PROJECT_ROOT, args.model_dir)
    print(f"Loading student model from: {model_dir}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model     = DistilBertForQuestionAnswering.from_pretrained(model_dir)
    model.eval()

    pages_dir = os.path.join(PROJECT_ROOT, args.pages_dir)

    print(f"\nRunning evaluation...\n")
    print("=" * 60)

    correct = 0
    results = []

    for i, item in enumerate(quiz_bank):
        question       = item["question"]
        correct_answer = item["answer"]
        source         = item.get("source", "unknown")

        # Load the relevant Wikipedia page as context
        context = load_page_text(source, pages_dir)
        if not context:
            print(f"[{i+1}/{len(quiz_bank)}] SKIP — no page found for source: {source}")
            continue

        student_answer = get_student_answer(question, context, model, tokenizer)
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

    score     = correct / len(quiz_bank) if quiz_bank else 0
    score_str = f"{correct}/{len(quiz_bank)}"

    print("=" * 60)
    print(f"\nFinal score: {score_str}  ({score:.0%})")
    print()

    with mlflow.start_run(run_name=f"eval-round-{args.round}"):
        mlflow.log_param("round",     args.round)
        mlflow.log_param("model_dir", args.model_dir)
        mlflow.log_param("questions", len(quiz_bank))
        mlflow.log_metric("score",    score)
        mlflow.log_metric("correct",  correct)
        mlflow.log_metric("total",    len(quiz_bank))

        results_path = os.path.join(PROJECT_ROOT, "data", f"eval_round_{args.round}.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(results_path)

    print(f"Results logged to MLflow")
    print(f"Detailed results saved to: data/eval_round_{args.round}.json")

    if score < 0.5:
        print("\nTip: score below 50% — try adding more pages and retraining")
    elif score < 0.8:
        print("\nTip: good progress! Add more pages or increase training epochs")
    else:
        print("\nExcellent! The model has strong knowledge of this topic")


if __name__ == "__main__":
    main()
