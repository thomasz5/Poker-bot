"""
Dataset Scaffold Script

Creates a minimal training dataset from available sample hand histories
and writes a manifest under training_data/ for reproducibility.

Usage:
  python scripts/scaffold_dataset.py
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any

import numpy as np

from training.data_processor import DataProcessor


def find_sample_files() -> List[str]:
    candidates = [
        "data/sample_hands/ggpoker_sample.txt",
        "data/sample_hands/pokerstars_sample.txt",
    ]
    return [p for p in candidates if os.path.exists(p)]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def action_distribution(action_targets: np.ndarray, id_to_action: Dict[int, str]) -> Dict[str, int]:
    counts = {}
    if action_targets is None or len(action_targets) == 0:
        return counts
    binc = np.bincount(action_targets)
    for i, c in enumerate(binc):
        name = id_to_action.get(i, f"unknown_{i}")
        counts[name] = int(c)
    return counts


def main() -> None:
    print("Scaffolding dataset...")
    out_dir = "training_data"
    ensure_dir(out_dir)

    files = find_sample_files()
    if not files:
        print("No sample hand histories found in data/sample_hands. Add files and rerun.")
        return

    processor = DataProcessor()
    dataset = processor.create_training_dataset(files)
    if not dataset:
        print("Failed to create dataset.")
        return

    out_npz = os.path.join(out_dir, "dataset.npz")
    processor.save_dataset(dataset, out_npz)

    meta = dataset.get("metadata", {})
    manifest: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_files": files,
        "num_examples": int(len(dataset["features"])) if "features" in dataset else 0,
        "feature_dim": int(dataset.get("feature_dim", 0)),
        "num_actions": int(dataset.get("num_actions", 0)),
        "action_mapping": dataset.get("action_mapping", {}),
        "action_distribution": action_distribution(dataset.get("action_targets", np.array([])), processor.id_to_action),
        "positions": list(set(meta.get("positions", []))) if meta else [],
        "streets": list(set(meta.get("streets", []))) if meta else [],
        "artifact": os.path.relpath(out_npz),
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDataset scaffold complete:")
    print(f"  -> {out_npz}")
    print(f"  -> {manifest_path}")
    print(f"  Examples: {manifest['num_examples']}")
    print(f"  Feature dim: {manifest['feature_dim']}")
    print(f"  Actions: {manifest['action_distribution']}")


if __name__ == "__main__":
    main()


