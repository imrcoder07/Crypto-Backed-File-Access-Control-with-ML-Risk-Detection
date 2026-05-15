#!/usr/bin/env python3
"""
generate_model_hashes.py
------------------------
Run this script ONCE after training (or whenever models are retrained) to
record the SHA-256 fingerprint of each model file.

Usage:
    python Crypto-models/generate_model_hashes.py

Output:
    Crypto-models/model_hashes.json  (safe to commit to version control)

The Flask app reads model_hashes.json at startup to verify that no model
file has been tampered with since it was last approved.
"""

import hashlib
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "model_hashes.json")

MODEL_FILES = [
    "random_forest_pipeline.pkl",
    "svm_pipeline.pkl",
    "isolation_forest_pipeline.pkl",
]


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 60)
    print("  ML Model Hash Generator")
    print("=" * 60)

    if not os.path.isdir(MODELS_DIR):
        print(f"\n❌ Models directory not found: {MODELS_DIR}")
        print("   Train your models first, then re-run this script.")
        sys.exit(1)

    hashes = {}
    missing = []

    for filename in MODEL_FILES:
        filepath = os.path.join(MODELS_DIR, filename)
        if os.path.exists(filepath):
            digest = sha256_of_file(filepath)
            hashes[filename] = digest
            print(f"  ✅ {filename}")
            print(f"     SHA-256: {digest}")
        else:
            missing.append(filename)
            print(f"  ⚠️  {filename} — NOT FOUND, skipped")

    if missing:
        print(
            f"\n⚠️  Warning: {len(missing)} model file(s) were missing and will not be "
            "integrity-checked at runtime."
        )

    if not hashes:
        print("\n❌ No model files found. Nothing written.")
        sys.exit(1)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(hashes, f, indent=2)

    print(f"\n✅ Hashes written to: {OUTPUT_PATH}")
    print("   Commit this file to lock down model integrity.")
    print("   Re-run this script whenever models are retrained.")
    print("=" * 60)


if __name__ == "__main__":
    main()
