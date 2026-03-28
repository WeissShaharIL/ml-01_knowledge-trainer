import os
import yaml
import torch

# ── config.py ──────────────────────────────────────────────────────────────────
# Loads config.yaml and resolves the device setting.
# Import this in any script that needs configuration.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH  = os.path.join(PROJECT_ROOT, "config.yaml")


def load_config() -> dict:
    """Load config.yaml from the project root."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_setting: str) -> str:
    """
    Resolve the device setting to an actual device string.
    - "cpu"  → always use CPU
    - "cuda" → use GPU (will error if no GPU available)
    - "auto" → use GPU if available, otherwise CPU
    """
    if device_setting == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: auto → using {device.upper()}")
        return device

    if device_setting == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: device=cuda but no CUDA GPU found! Falling back to CPU.")
            return "cpu"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device: GPU ({gpu_name})")
        return "cuda"

    print("Device: CPU")
    return "cpu"


def get_device() -> str:
    """Convenience function — load config and return resolved device."""
    cfg    = load_config()
    setting = cfg.get("hardware", {}).get("device", "cpu")
    return resolve_device(setting)
