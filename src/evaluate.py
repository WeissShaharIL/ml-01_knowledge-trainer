import argparse
import json
import os
import requests
import torch
from transformers import DistilBertTokenizerFast, DistilBertForMaskedLM
import mlflow

# ── evaluate.py ────────────────────────────────────────────────────────────────
# Step 4 of the knowledge trainer pipeline.
# Loads the trained student model and quizzes it using questions from
# quiz_bank.json. Prints a score and logs it to MLflow.
#
# How evaluation works:
# For each question in the quiz bank we ask BOTH the student model AND Ollama.
# - Ollama gives us the "gold standard" answer (it can read the question and reason)
# - The student model gives us a fill-in-the-blank style answer via MLM
# - We then ask Ollama to judge whether the student's answer is correct
#
# Why use Ollama as the judge?
# Exact string matching fails for natural language — "Alan Turing" and
# "Turing" are the same answer but wouldn't match. Ollama can judge semantic
# equivalence the way a human grader would. This is called LLM-as-a-judge
# and is a standard evaluation pattern in modern MLOps.

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DIR   = os.path.join(PROJECT_ROOT, "mlruns")
mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")
mlflow.set_experiment("knowledge-trainer")


def call_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the response."""
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["response"].strip()


def get_student_answer(question: str, model, tokenizer, top_k: int = 5) -> str:
    """
    Ask the student model a question using masked language modeling.
    We append [MASK] to the question and ask the model to fill it in.
    The top predicted token becomes the student's answer.
    """
    # Format: "Question: Who invented the telephone? Answer: [MASK]"
    masked_input = f"Question: {question} Answer: [MASK]"

    inputs = tokenizer(
        masked_input,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        outputs = model(**inputs)

    logits       = outputs.logits[0, mask_idx, :]
    top_token_ids = logits.topk(top_k).indices[0].tolist()
    predictions   = [tokenizer.decode([tid]).strip() for tid in top_token_ids]

    return ", ".join(predictions)


def judge_answer(question: str, correct_answer: str, student_answer: str) -> bool:
    """
    Ask Ollama to judge whether the student's answer is semantically correct.
    Returns True if correct, False if not.
    """
    prompt = f"""You are a quiz grader. Judge if the student answer is correct.

Question: {question}
Correct answer: {correct_answer}
Student answer: {student_answer}

Reply with only YES if the student answer is correct or contains the key information,
or NO if it is wrong or irrelevant. One word only."""

    verdict = call_ollama(prompt).strip().upper()
    return verdict.startswith("YES")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the student model against the quiz bank"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the trained model directory e.g. models/round_1"
    )
    parser.add_argument(
        "--quiz-bank",
        type=str,
        default="data/quiz_bank.json",
        help="Path to quiz_bank.json"
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Round number for MLflow logging"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to N questions (useful for quick checks)"
    )
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
    model     = DistilBertForMaskedLM.from_pretrained(model_dir)
    model.eval()

    # Run evaluation
    print(f"\nRunning evaluation...\n")
    print("=" * 60)

    correct = 0
    results = []

    for i, item in enumerate(quiz_bank):
        question       = item["question"]
        correct_answer = item["answer"]
        source         = item.get("source", "unknown")

        student_answer = get_student_answer(question, model, tokenizer)
        is_correct     = judge_answer(question, correct_answer, student_answer)

        if is_correct:
            correct += 1
            verdict = "CORRECT"
        else:
            verdict = "WRONG"

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

    score     = correct / len(quiz_bank)
    score_str = f"{correct}/{len(quiz_bank)}"

    print("=" * 60)
    print(f"\nFinal score: {score_str}  ({score:.0%})")
    print()

    # Log to MLflow
    with mlflow.start_run(run_name=f"eval-round-{args.round}"):
        mlflow.log_param("round",          args.round)
        mlflow.log_param("model_dir",      args.model_dir)
        mlflow.log_param("questions",      len(quiz_bank))
        mlflow.log_metric("score",         score)
        mlflow.log_metric("correct",       correct)
        mlflow.log_metric("total",         len(quiz_bank))

        # Save detailed results as an artifact
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
