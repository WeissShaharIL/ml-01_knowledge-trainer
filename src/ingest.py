import argparse
import os
import re
import requests
from datetime import datetime

# ── ingest.py ──────────────────────────────────────────────────────────────────
# Step 1 of the knowledge trainer pipeline.
# Fetches a Wikipedia page, strips all the HTML and wiki markup noise,
# and saves clean plain text to data/pages/<title>.txt
#
# Also updates data/manifest.json with metadata about what was fetched,
# so we have a complete audit trail of what data each model version saw.

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fetch_wikipedia_page(url: str) -> tuple[str, str]:
    title = url.rstrip("/").split("/wiki/")[-1]
    title = requests.utils.unquote(title)

    print(f"Fetching: {title} ...")

    params = {
        "action":          "query",
        "titles":          title,
        "prop":            "extracts",
        "explaintext":     True,
        "exsectionformat": "plain",
        "format":          "json",
        "redirects":       True,
    }

    response = requests.get(WIKIPEDIA_API, params=params, headers={
        "User-Agent": "Mozilla/5.0 (compatible; KnowledgeTrainer/1.0; shahar@example.com)"
    })
    response.raise_for_status()
    data = response.json()

    pages = data["query"]["pages"]
    page  = next(iter(pages.values()))

    if "missing" in page:
        raise ValueError(f"Wikipedia page not found: {title}")

    raw_text    = page.get("extract", "")
    clean_title = page.get("title", title)
    text        = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

    return clean_title, text


def save_page(title: str, text: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    safe_title = re.sub(r'[\\/*?:"<>|]', "_", title).replace(" ", "_")
    filepath   = os.path.join(output_dir, f"{safe_title}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {title}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)

    return filepath


def update_ingest_log(title: str, url: str, word_count: int, filepath: str):
    """
    Update data/ingest_log.json with metadata about this fetch.
    This log is used by pipeline.py to pass page metadata to versioning.
    """
    log_path = os.path.join(PROJECT_ROOT, "data", "ingest_log.json")

    # Load existing log
    if os.path.exists(log_path):
        import json
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = {}

    log[title] = {
        "title":      title,
        "url":        url,
        "word_count": word_count,
        "filepath":   filepath,
        "fetched_at": datetime.now().isoformat(),
    }

    import json
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"Ingest log updated: data/ingest_log.json")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a Wikipedia page and save it to data/pages/"
    )
    parser.add_argument("--url",        type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/pages")
    args = parser.parse_args()

    title, text = fetch_wikipedia_page(args.url)
    output_dir  = os.path.join(PROJECT_ROOT, args.output_dir)
    filepath    = save_page(title, text, output_dir)
    word_count  = len(text.split())

    print(f"Saved: {filepath}")
    print(f"Title: {title}")
    print(f"Words: {word_count:,}")
    print(f"\nFirst 300 characters:")
    print("-" * 40)
    print(text[:300])
    print("-" * 40)

    update_ingest_log(title, args.url, word_count, filepath)

    print("\nNext step: run generate_quiz.py")


if __name__ == "__main__":
    main()
