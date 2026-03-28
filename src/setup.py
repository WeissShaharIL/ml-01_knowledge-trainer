import os
import sys

# ── setup.py ───────────────────────────────────────────────────────────────────
# Run this once after cloning the repo.
# Downloads the base model and creates required directories.
# Safe to rerun — skips anything already done.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_dirs():
    dirs = [
        os.path.join(PROJECT_ROOT, "data", "pages"),
        os.path.join(PROJECT_ROOT, "models"),
        os.path.join(PROJECT_ROOT, "logs"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  ✓ {d}")


def download_base_model():
    from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering

    base_dir = os.path.join(PROJECT_ROOT, "models", "base")
    if os.path.exists(os.path.join(base_dir, "config.json")):
        print(f"  ✓ Base model already exists at models/base — skipping")
        return

    print("  Downloading distilbert-base-uncased-distilled-squad (~250MB)...")
    m = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
    t = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased-distilled-squad")
    m.save_pretrained(base_dir)
    t.save_pretrained(base_dir)
    print(f"  ✓ Base model saved to models/base")


def main():
    print("\n── Knowledge Trainer Setup ───────────────────────────")
    print("\nCreating directories...")
    setup_dirs()
    print("\nDownloading base model...")
    download_base_model()
    print("\n✓ Setup complete. You can now run:")
    print("  python src\\pipeline.py --url <wikipedia_url> --round 1")
    print("──────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()