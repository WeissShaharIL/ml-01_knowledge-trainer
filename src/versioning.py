import json
import os
from datetime import datetime

# ── versioning.py ──────────────────────────────────────────────────────────────
# Manages the model version registry for Knowledge Trainer.
#
# Versions follow semantic versioning: MAJOR.MINOR.PATCH
#   1.0.0 = baseline (round 0, no pages)
#   1.1.0 = first Wikipedia page added
#   1.2.0 = second Wikipedia page added
#   etc.
#
# The manifest.json is the human-readable source of truth.
# It is committed to git so you always know what version is in production and why.

MANIFEST_FILENAME = "manifest.json"
INITIAL_VERSION   = "1.0.0"


def get_manifest_path(project_root: str) -> str:
    return os.path.join(project_root, "data", MANIFEST_FILENAME)


def load_manifest(project_root: str) -> dict:
    """Load the manifest or create a fresh one if it doesn't exist."""
    path = get_manifest_path(project_root)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # First time — create initial manifest
    return {
        "current_version": None,
        "production_version": None,
        "versions": {}
    }


def save_manifest(project_root: str, manifest: dict):
    path = get_manifest_path(project_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def bump_minor_version(version: str) -> str:
    """Bump the minor version: 1.0.0 → 1.1.0 → 1.2.0"""
    if version is None:
        return INITIAL_VERSION
    major, minor, patch = version.split(".")
    return f"{major}.{int(minor) + 1}.{patch}"


def get_next_version(manifest: dict) -> str:
    current = manifest.get("current_version")
    if current is None:
        return INITIAL_VERSION
    return bump_minor_version(current)


def register_version(
    project_root: str,
    version: str,
    round_num: int,
    pages: list[str],
    urls: list[str],
    word_counts: dict[str, int],
    model_dir: str,
    score: float = None,
    status: str = "staging",
    mlflow_run_id: str = None,
) -> dict:
    """
    Register a new model version in the manifest.
    Status is 'staging' until evaluation passes, then 'production'.
    """
    manifest = load_manifest(project_root)

    manifest["versions"][version] = {
        "version":       version,
        "round":         round_num,
        "pages":         pages,
        "urls":          urls,
        "word_counts":   word_counts,
        "model_dir":     model_dir,
        "score":         score,
        "status":        status,
        "mlflow_run_id": mlflow_run_id,
        "created_at":    datetime.now().isoformat(),
    }

    manifest["current_version"] = version

    save_manifest(project_root, manifest)
    return manifest


def promote_to_production(project_root: str, version: str, score: float):
    """Mark a version as production and retire the old one."""
    manifest = load_manifest(project_root)

    # Retire previous production version
    old_production = manifest.get("production_version")
    if old_production and old_production in manifest["versions"]:
        manifest["versions"][old_production]["status"] = "retired"

    # Promote new version
    if version in manifest["versions"]:
        manifest["versions"][version]["status"] = "production"
        manifest["versions"][version]["score"]  = score

    manifest["production_version"] = version
    save_manifest(project_root, manifest)


def get_production_version(project_root: str) -> dict | None:
    """Get the current production version entry, or None."""
    manifest = load_manifest(project_root)
    prod_ver = manifest.get("production_version")
    if prod_ver:
        return manifest["versions"].get(prod_ver)
    return None


def print_manifest_summary(project_root: str):
    """Print a clean summary of all versions."""
    manifest = load_manifest(project_root)
    versions = manifest.get("versions", {})

    if not versions:
        print("No versions registered yet.")
        return

    print("\n── Version Registry ──────────────────────────────────────")
    for ver, info in sorted(versions.items()):
        status  = info.get("status", "unknown")
        score   = info.get("score")
        pages   = info.get("pages", [])
        created = info.get("created_at", "")[:10]  # just the date

        score_str = f"{score:.0%}" if score is not None else "not evaluated"
        pages_str = ", ".join(pages) if pages else "none (baseline)"

        icon = {"production": "🟢", "staging": "🟡", "retired": "⚫"}.get(status, "⚪")
        print(f"  {icon} v{ver}  [{status}]  score={score_str}  pages={pages_str}  ({created})")

    print("──────────────────────────────────────────────────────────\n")
