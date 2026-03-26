import argparse
import glob
import json
import os
import re
import requests

# ── generate_quiz.py ───────────────────────────────────────────────────────────
# Step 2 of the knowledge trainer pipeline.
# Sends each page in data/pages/ to Ollama (llama3.2:3b) and asks it to
# generate question/answer pairs. Results are saved to data/quiz_bank.json.
#
# The quiz bank grows with every new page — so by Round 3 you're being
# tested on everything the model has ever been taught, not just the latest page.
# This makes evaluation increasingly rigorous over time, which is exactly
# what you want: a model that remembers everything, not just the last thing.
#
# Why Ollama instead of a Python QA library?
# A general LLM produces much more natural, varied questions than rule-based
# extractors. It can ask about causes, implications, and relationships —
# not just "who/what/when" facts. Better questions = better evaluation.

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

PROMPT_TEMPLATE = """You are a quiz generator. Read the following text carefully and generate {n} question and answer pairs that test factual knowledge from the text.

Rules:
- Questions must be answerable ONLY from the text below
- Answers must be short (1 sentence maximum)
- Cover different parts of the text, not just the beginning
- Do not generate opinion or interpretation questions — facts only

Return ONLY a JSON array in this exact format, no other text:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]

Text:
{text}
"""

def call_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,  # wait for full response before returning
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"]


def extract_json(text: str) -> list:
    """
    Extract a JSON array from Ollama's response.
    LLMs sometimes wrap JSON in markdown code blocks — this handles that.
    """
    # Try to find a JSON array in the response
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # If that fails, try parsing the whole response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []


def generate_questions_for_page(filepath: str, n_questions: int) -> list:
    """Read a page file and generate n_questions Q&A pairs from it."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Use only the first 3000 words — Ollama has a context limit and
    # the opening of a Wikipedia article is the most information-dense part
    words = content.split()[:3000]
    text  = " ".join(words)

    # Extract title from the first line (we write "TITLE: ..." in ingest.py)
    title = filepath.replace("\\", "/").split("/")[-1].replace(".txt", "").replace("_", " ")

    print(f"  Generating {n_questions} questions for: {title} ...")
    prompt   = PROMPT_TEMPLATE.format(text=text, n=n_questions)
    response = call_ollama(prompt)
    pairs    = extract_json(response)

    if not pairs:
        print(f"  Warning: could not parse questions for {title}. Skipping.")
        return []

    # Tag each pair with the source page so we know where it came from
    for pair in pairs:
        pair["source"] = title

    print(f"  Generated {len(pairs)} questions")
    return pairs


def load_quiz_bank(path: str) -> list:
    """Load existing quiz bank or return empty list if it doesn't exist yet."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_quiz_bank(path: str, bank: list):
    """Save the quiz bank to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate quiz questions from pages in data/pages/"
    )
    parser.add_argument(
        "--pages-dir",
        type=str,
        default="data/pages",
        help="Directory containing page .txt files"
    )
    parser.add_argument(
        "--quiz-bank",
        type=str,
        default="data/quiz_bank.json",
        help="Path to the quiz bank JSON file"
    )
    parser.add_argument(
        "--questions-per-page",
        type=int,
        default=5,
        help="Number of questions to generate per page (default: 5)"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate questions for all pages, not just new ones"
    )
    args = parser.parse_args()

    # Find all page files
    pattern = os.path.join(args.pages_dir, "*.txt")
    pages   = sorted(glob.glob(pattern))

    if not pages:
        print(f"No .txt files found in {args.pages_dir}/")
        print("Run ingest.py first.")
        return

    # Load existing quiz bank
    existing_bank   = load_quiz_bank(args.quiz_bank)
    existing_sources = {q["source"] for q in existing_bank}

    print(f"Found {len(pages)} page(s) in {args.pages_dir}/")
    print(f"Existing quiz bank: {len(existing_bank)} questions from {len(existing_sources)} page(s)")
    print()

    new_pairs = []
    for page_path in pages:
        title = page_path.replace("\\", "/").split("/")[-1].replace(".txt", "").replace("_", " ")

        # Skip pages we already have questions for (unless --regenerate)
        if title in existing_sources and not args.regenerate:
            print(f"  Skipping {title} — already in quiz bank (use --regenerate to redo)")
            continue

        pairs = generate_questions_for_page(page_path, args.questions_per_page)
        new_pairs.extend(pairs)

    if not new_pairs:
        print("\nNo new questions generated.")
        return

    # Merge new questions into existing bank
    if args.regenerate:
        updated_bank = new_pairs
    else:
        updated_bank = existing_bank + new_pairs

    save_quiz_bank(args.quiz_bank, updated_bank)

    print(f"\nQuiz bank updated: {args.quiz_bank}")
    print(f"Total questions: {len(updated_bank)}")
    print(f"New questions added: {len(new_pairs)}")
    print("\nSample questions:")
    for q in new_pairs[:3]:
        print(f"  Q: {q['question']}")
        print(f"  A: {q['answer']}")
        print()
    print("Next step: run train.py")


if __name__ == "__main__":
    main()
