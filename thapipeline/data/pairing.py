"""Leakage-safe domain matching and pairing for the THA pipeline."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from thapipeline.config import PipelineConfig
from thapipeline.data.transforms import PreprocessPipeline
from thapipeline.utils.io import load_image, save_json, sanitize_id


def compute_features(image: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Extract a pairing feature vector from a preprocessed image."""
    h, w = image.shape[:2]
    img_float = image.astype(np.float32) / 255.0

    threshold = np.percentile(image, 75)
    bright_mask = image > threshold
    col_sums = bright_mask.sum(axis=0)
    bright_cols = np.where(col_sums > h * 0.1)[0]
    width_ratio = (bright_cols[-1] - bright_cols[0]) / w if len(bright_cols) > 0 else 0.5

    row_sums = img_float.sum(axis=1)
    total = float(row_sums.sum())
    y_com = float(np.sum(np.arange(h) * row_sums) / (total * h)) if total > 0 else 0.5
    mean_intensity = float(np.mean(img_float))
    std_intensity = float(np.std(img_float))

    hist, _ = np.histogram(img_float.ravel(), bins=n_bins, range=(0, 1))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()

    return np.concatenate([[width_ratio, y_com, mean_intensity, std_intensity], hist]).astype(np.float32)


def _processed_path(
    config: PipelineConfig,
    split: str,
    role: str,
    dataset: str,
    source_id: str,
) -> str:
    filename = f"{sanitize_id(dataset)}__{sanitize_id(source_id)}.png"
    return str(config.paths.data_processed / split / role / filename)


def _prepare_feature_frame(catalogue: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    preprocess = PreprocessPipeline(
        target_size=config.image.target_size,
        crop_ratio=config.image.center_crop_ratio,
        clahe_clip=config.image.clahe_clip_limit,
        clahe_grid=config.image.clahe_tile_grid,
    )

    rows: List[Dict[str, object]] = []
    for _, row in catalogue.iterrows():
        try:
            image = load_image(Path(row["filepath"]))
            enhanced = preprocess(image)["enhanced"]
            features = compute_features(enhanced)
        except Exception:
            continue

        record = row.to_dict()
        for idx, value in enumerate(features):
            record[f"feat_{idx:02d}"] = float(value)
        rows.append(record)

    return pd.DataFrame(rows)


def _split_ids(ids: Iterable[str], config: PipelineConfig, seed: int) -> Dict[str, List[str]]:
    ids = sorted(set(ids))
    rng = np.random.RandomState(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * config.split.train_ratio))
    n_val = int(round(n * config.split.val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, max(n - n_train, 0))
    n_test = max(n - n_train - n_val, 0)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }


def _build_folds(ids: Iterable[str], n_folds: int, seed: int) -> List[List[str]]:
    ids = sorted(set(ids))
    rng = np.random.RandomState(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)

    fold_sizes = np.full(n_folds, len(shuffled) // n_folds, dtype=int)
    fold_sizes[: len(shuffled) % n_folds] += 1

    folds: List[List[str]] = []
    start = 0
    for size in fold_sizes:
        folds.append(shuffled[start : start + int(size)])
        start += int(size)
    return folds


def _build_split_membership(features_df: pd.DataFrame, config: PipelineConfig) -> Dict[str, Dict[str, List[str]]]:
    pre_df = features_df[features_df["postop_flag"] == 0]
    post_df = features_df[features_df["postop_flag"] == 1]

    pre_splits = _split_ids(pre_df["canonical_source_id"], config, config.seed)
    post_splits = _split_ids(post_df["canonical_source_id"], config, config.seed + 1)

    return {
        split: {
            "pre_ids": pre_splits[split],
            "post_ids": post_splits[split],
            "pair_ids": [],
        }
        for split in ("train", "val", "test")
    }


def _match_split(split_name: str, split_df: pd.DataFrame, config: PipelineConfig) -> List[Dict[str, object]]:
    pre_df = split_df[split_df["postop_flag"] == 0].reset_index(drop=True)
    post_df = split_df[split_df["postop_flag"] == 1].reset_index(drop=True)
    if pre_df.empty or post_df.empty:
        return []

    feat_cols = [col for col in split_df.columns if col.startswith("feat_")]
    pre_features = pre_df[feat_cols].to_numpy(dtype=np.float32)
    post_features = post_df[feat_cols].to_numpy(dtype=np.float32)

    all_features = np.vstack([pre_features, post_features])
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0) + 1e-8
    pre_norm = (pre_features - mean) / std
    post_norm = (post_features - mean) / std

    distances = cdist(pre_norm, post_norm, metric="euclidean")
    reuse_counter = Counter()
    matched_rows: List[Dict[str, object]] = []

    for pre_idx in np.argsort(distances.min(axis=1)):
        sorted_posts = np.argsort(distances[pre_idx])
        pre_row = pre_df.iloc[pre_idx]
        for post_idx in sorted_posts:
            post_row = post_df.iloc[post_idx]
            post_id = str(post_row["canonical_source_id"])
            if reuse_counter[post_id] >= config.split.max_postop_reuse:
                continue

            reuse_counter[post_id] += 1
            matched_rows.append(
                {
                    "pair_id": f"{split_name}_{len(matched_rows):05d}",
                    "split": split_name,
                    "pre_path": str(pre_row["filepath"]),
                    "post_path": str(post_row["filepath"]),
                    "pre_processed_path": _processed_path(
                        config,
                        split_name,
                        "pre",
                        str(pre_row["source_dataset"]),
                        str(pre_row["canonical_source_id"]),
                    ),
                    "post_processed_path": _processed_path(
                        config,
                        split_name,
                        "post",
                        str(post_row["source_dataset"]),
                        str(post_row["canonical_source_id"]),
                    ),
                    "distance": float(distances[pre_idx, post_idx]),
                    "pre_source": str(pre_row["source_dataset"]),
                    "pre_id": str(pre_row["canonical_source_id"]),
                    "post_id": post_id,
                    "post_reuse_count": int(reuse_counter[post_id]),
                }
            )
            break

    return matched_rows


def create_pairs(catalogue: pd.DataFrame, config: PipelineConfig) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[str]]]]:
    features_df = _prepare_feature_frame(catalogue, config)
    split_indices = _build_split_membership(features_df, config)

    split_lookup: Dict[str, str] = {}
    for split_name, groups in split_indices.items():
        for pre_id in groups["pre_ids"]:
            split_lookup[f"pre::{pre_id}"] = split_name
        for post_id in groups["post_ids"]:
            split_lookup[f"post::{post_id}"] = split_name

    def _assign_row_split(row: pd.Series) -> str:
        key_prefix = "post" if int(row["postop_flag"]) == 1 else "pre"
        return split_lookup[f"{key_prefix}::{row['canonical_source_id']}"]

    features_df = features_df.copy()
    features_df["split"] = features_df.apply(_assign_row_split, axis=1)

    pairs: List[Dict[str, object]] = []
    for split_name in ("train", "val", "test"):
        pairs.extend(_match_split(split_name, features_df[features_df["split"] == split_name], config))

    pairs_df = pd.DataFrame(
        pairs,
        columns=[
            "pair_id",
            "split",
            "pre_path",
            "post_path",
            "pre_processed_path",
            "post_processed_path",
            "distance",
            "pre_source",
            "pre_id",
            "post_id",
            "post_reuse_count",
        ],
    )

    for split_name, groups in split_indices.items():
        groups["pair_ids"] = pairs_df[pairs_df["split"] == split_name]["pair_id"].tolist()

    return pairs_df, split_indices


def save_pairing_results(
    pairs_df: pd.DataFrame,
    split_indices: Dict[str, Dict[str, List[str]]],
    config: PipelineConfig,
    pairing_table_path: Path | None = None,
    split_indices_path: Path | None = None,
    qa_path: Path | None = None,
) -> None:
    pairing_table_path = pairing_table_path or config.paths.pairing_table
    split_indices_path = split_indices_path or config.paths.split_indices
    qa_path = qa_path or config.paths.pairing_qa

    pairing_table_path.parent.mkdir(parents=True, exist_ok=True)
    split_indices_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.parent.mkdir(parents=True, exist_ok=True)

    pairs_df.to_csv(pairing_table_path, index=False)
    save_json(split_indices, split_indices_path)

    reuse_hist = (
        pairs_df.groupby(["split", "post_id"]).size().reset_index(name="reuse").groupby("split")["reuse"].apply(list)
    )
    qa_report = {
        "n_pairs": int(len(pairs_df)),
        "split_counts": {split: int((pairs_df["split"] == split).sum()) for split in ("train", "val", "test")},
        "source_counts": pairs_df["pre_source"].value_counts().to_dict(),
        "post_reuse_histogram": {split: reuse_hist.get(split, []) for split in ("train", "val", "test")},
        "representative_pairs": pairs_df.sort_values("distance").head(10).to_dict(orient="records"),
    }
    save_json(qa_report, qa_path)


def run_pairing_pipeline(config: PipelineConfig) -> pd.DataFrame:
    catalogue = pd.read_csv(config.paths.catalogue_csv)
    pairs_df, split_indices = create_pairs(catalogue, config)
    save_pairing_results(pairs_df, split_indices, config)
    return pairs_df


def create_kfold_pairs(
    catalogue: pd.DataFrame,
    config: PipelineConfig,
    n_folds: int = 5,
) -> List[Tuple[pd.DataFrame, Dict[str, Dict[str, List[str]]]]]:
    features_df = _prepare_feature_frame(catalogue, config)
    pre_df = features_df[features_df["postop_flag"] == 0]
    post_df = features_df[features_df["postop_flag"] == 1]

    pre_folds = _build_folds(pre_df["canonical_source_id"], n_folds, config.seed)
    post_folds = _build_folds(post_df["canonical_source_id"], n_folds, config.seed + 1)

    fold_results: List[Tuple[pd.DataFrame, Dict[str, Dict[str, List[str]]]]] = []
    for fold_idx in range(n_folds):
        val_idx = (fold_idx + 1) % n_folds
        train_indices = [idx for idx in range(n_folds) if idx not in {fold_idx, val_idx}]

        split_indices = {
            "train": {
                "pre_ids": [item for idx in train_indices for item in pre_folds[idx]],
                "post_ids": [item for idx in train_indices for item in post_folds[idx]],
                "pair_ids": [],
            },
            "val": {
                "pre_ids": list(pre_folds[val_idx]),
                "post_ids": list(post_folds[val_idx]),
                "pair_ids": [],
            },
            "test": {
                "pre_ids": list(pre_folds[fold_idx]),
                "post_ids": list(post_folds[fold_idx]),
                "pair_ids": [],
            },
        }

        split_lookup: Dict[str, str] = {}
        for split_name, groups in split_indices.items():
            for pre_id in groups["pre_ids"]:
                split_lookup[f"pre::{pre_id}"] = split_name
            for post_id in groups["post_ids"]:
                split_lookup[f"post::{post_id}"] = split_name

        def _assign_row_split(row: pd.Series) -> str:
            key_prefix = "post" if int(row["postop_flag"]) == 1 else "pre"
            return split_lookup[f"{key_prefix}::{row['canonical_source_id']}"]

        fold_df = features_df.copy()
        fold_df["split"] = fold_df.apply(_assign_row_split, axis=1)

        pairs: List[Dict[str, object]] = []
        for split_name in ("train", "val", "test"):
            pairs.extend(_match_split(split_name, fold_df[fold_df["split"] == split_name], config))

        pairs_df = pd.DataFrame(
            pairs,
            columns=[
                "pair_id",
                "split",
                "pre_path",
                "post_path",
                "pre_processed_path",
                "post_processed_path",
                "distance",
                "pre_source",
                "pre_id",
                "post_id",
                "post_reuse_count",
            ],
        )

        for split_name, groups in split_indices.items():
            groups["pair_ids"] = pairs_df[pairs_df["split"] == split_name]["pair_id"].tolist()

        fold_results.append((pairs_df, split_indices))

    return fold_results


def run_kfold_pairing_pipeline(
    config: PipelineConfig,
    n_folds: int = 5,
    output_root: Path | None = None,
) -> List[Path]:
    catalogue = pd.read_csv(config.paths.catalogue_csv)
    output_root = output_root or (config.paths.metadata_dir / "cross_validation")
    output_root.mkdir(parents=True, exist_ok=True)

    fold_paths: List[Path] = []
    for fold_number, (pairs_df, split_indices) in enumerate(create_kfold_pairs(catalogue, config, n_folds=n_folds), start=1):
        fold_dir = output_root / f"fold_{fold_number:02d}"
        pairing_path = fold_dir / "pairing_table.csv"
        save_pairing_results(
            pairs_df,
            split_indices,
            config,
            pairing_table_path=pairing_path,
            split_indices_path=fold_dir / "split_indices.json",
            qa_path=fold_dir / "pairing_qa.json",
        )
        fold_paths.append(pairing_path)

    summary = {
        "n_folds": n_folds,
        "folds": [str(path) for path in fold_paths],
    }
    save_json(summary, output_root / "cross_validation_manifest.json")
    return fold_paths
