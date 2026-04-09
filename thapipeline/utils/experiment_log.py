"""Experiment logging helpers for reproducible dissertation runs."""

from __future__ import annotations

import csv
import json
import platform
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_snapshot(config: Any) -> Dict[str, Any]:
    """Convert config dataclasses into a JSON-serialisable snapshot."""
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    raise TypeError(f"Unsupported config type for snapshot: {type(config)!r}")


def collect_dataset_summary(pairing_table: Path) -> Dict[str, Any]:
    """Summarise split counts and post-op reuse from the pairing table."""
    if not pairing_table.exists():
        return {"available": False}

    split_pairs: Dict[str, int] = {}
    split_posts: Dict[str, set] = {}
    max_reuse = 0
    total_pairs = 0

    with open(pairing_table, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = row.get("split", "unknown")
            post_id = row.get("post_id", "")
            total_pairs += 1
            split_pairs[split] = split_pairs.get(split, 0) + 1
            split_posts.setdefault(split, set()).add(post_id)
            try:
                max_reuse = max(max_reuse, int(float(row.get("post_reuse_count", 0) or 0)))
            except ValueError:
                pass

    return {
        "available": True,
        "total_pairs": total_pairs,
        "pairs_by_split": split_pairs,
        "unique_posts_by_split": {split: len(ids) for split, ids in split_posts.items()},
        "max_post_reuse": max_reuse,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def write_history_csv(path: Path, history: Dict[str, Iterable[Any]]) -> None:
    """Persist training history into a simple CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = {key: list(values) for key, values in history.items()}
    keys = list(prepared.keys())
    max_len = max((len(values) for values in prepared.values()), default=0)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for index in range(max_len):
            writer.writerow(
                [index + 1] + [prepared[key][index] if index < len(prepared[key]) else "" for key in keys]
            )


def environment_snapshot(device: str) -> Dict[str, Any]:
    """Collect lightweight runtime environment metadata."""
    return {
        "device": device,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }
