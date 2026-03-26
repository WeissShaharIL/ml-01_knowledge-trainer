import argparse
import os
import re
import requests

# ── ingest.py ──────────────────────────────────────────────────────────────────
# Step 1 of the knowledge trainer pipeline.
# Fetches a Wikipedia page, strips all the HTML and wiki markup noise,
# and saves clean plain text to data/pages/<title>.txt
#
# Why save raw .txt files instead of keeping it in memory?
# Same reason you don't regenerate your Docker image on every deploy —
# the raw page is the source of truth. If Ollama generates bad questions
# or the fine-tuning produces a bad model, you can always reprocess the
# same page without re-fetching it. Also means you can inspect exactly
# what text the model was trained on, which is critical for debugging.

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

def fetch_wikipedia_page(url: str) -> tuple[str, str]:
    """
    Given a Wikipedia URL, fetch the page title and clean plain text content.
    Returns (title, text).
    """
    # Extract the page title from the URL
    # e.g. https://en.wikipedia.org/wiki/Alan_Turing → Alan_Turing
    title = url.rstrip("/").split("/wiki/")[-1]
    title = requests.utils.unquote(title)  # decode %20 etc.

    print(f"Fetching: {title} ...")

    # Use the Wikipedia API to get clean plain text — much better than
    # scraping HTML which is full of nav menus, references, and markup
    params = {
        "action":      "query",
        "titles":      title,
        "prop":        "extracts",
        "explaintext": True,      # plain text, no HTML
        "exsectionformat": "plain",
        "format":      "json",
        "redirects":   True,      # follow redirects automatically
    }

    response = requests.get(WIKIPEDIA_API, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    pages = data["query"]["pages"]
    page  = next(iter(pages.values()))

    if "missing" in page:
        raise ValueError(f"Wikipedia page not found: {title}")

    raw_text = page.get("extract", "")
    clean_title = page.get("title", title)

    # Clean up the text — remove excessive blank lines and whitespace
    # We keep section structure (headers like == History ==) because
    # they help the question generator understand context boundaries
    text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

    return clean_title, text


def save_page(title: str, text: str, output_dir: str) -> str:
    """Save the page text to output_dir/<safe_title>.txt"""
    os.makedirs(output_dir, exist_ok=True)

    # Make the title safe for use as a filename
    safe_title = re.sub(r'[\\/*?:"<>|]', "_", title).replace(" ", "_")
    filepath = os.path.join(output_dir, f"{safe_title}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {title}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a Wikipedia page and save it to data/pages/"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help='Wikipedia URL e.g. "https://en.wikipedia.org/wiki/Alan_Turing"'
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pages",
        help="Directory to save page text (default: data/pages)"
    )
    args = parser.parse_args()

    title, text = fetch_wikipedia_page(args.url)
    filepath = save_page(title, text, args.output_dir)

    word_count = len(text.split())
    print(f"Saved: {filepath}")
    print(f"Title: {title}")
    print(f"Words: {word_count:,}")
    print(f"\nFirst 300 characters:")
    print("-" * 40)
    print(text[:300])
    print("-" * 40)
    print("\nNext step: run generate_quiz.py")


if __name__ == "__main__":
    main()
