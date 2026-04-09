"""Segmentation model training for post-operative implant masks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from thapipeline.config import PipelineConfig, get_device
from thapipeline.data.transforms import PreprocessPipeline
from thapipeline.eval.metrics import (
    compute_dice,
    compute_iou,
    compute_pixel_accuracy,
    compute_precision_recall_f1,
)
from thapipeline.eval.statistics import summary_with_ci
from thapipeline.models.segmenter import (
    ImplantSegmenter,
    PixelDataset,
    PixelSegmentationModel,
    SegmentationUNet,
    build_feature_tensor,
    gradient_map,
)
from thapipeline.utils.io import load_image, save_checkpoint, save_json


def _lookup_postop_image(case_id: str, catalogue: pd.DataFrame) -> Optional[Path]:
    matches = catalogue[
        (catalogue["source_dataset"] == "hipxnet")
        & (catalogue["canonical_source_id"].astype(str) == str(case_id))
    ]
    if matches.empty:
        return None
    return Path(matches.iloc[0]["filepath"])


def prepare_training_records(
    config: PipelineConfig,
    gan_outputs: Optional[List[Dict]] = None,
) -> List[Dict[str, np.ndarray]]:
    """Prepare segmentation records from annotated HipXNet implant masks."""
    mask_dir = config.paths.implant_masks_dir
    if not mask_dir.exists():
        return []

    catalogue = pd.read_csv(config.paths.catalogue_csv)
    preprocess = PreprocessPipeline(
        target_size=config.image.target_size,
        crop_ratio=config.image.center_crop_ratio,
        clahe_clip=config.image.clahe_clip_limit,
        clahe_grid=config.image.clahe_tile_grid,
    )

    records: List[Dict[str, np.ndarray]] = []
    for mask_path in sorted(mask_dir.glob("*.png")):
        case_id = mask_path.stem
        image_path = _lookup_postop_image(case_id, catalogue)
        if image_path is None or not image_path.exists():
            continue

        enhanced = preprocess(load_image(image_path))["enhanced"]
        mask = load_image(mask_path)
        mask = (mask > 0).astype(np.float32)

        if mask.shape != enhanced.shape:
            import cv2

            mask = cv2.resize(mask, config.image.target_size, interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)

        records.append(
            {
                "case_id": case_id,
                "enhanced": enhanced,
                "gan": enhanced.copy(),
                "grad": gradient_map(enhanced),
                "label": mask,
            }
        )
    return records


def split_segmentation_records(
    records: List[Dict[str, np.ndarray]],
    config: PipelineConfig,
) -> Tuple[Dict[str, List[Dict[str, np.ndarray]]], Dict[str, Dict[str, object]]]:
    """Split annotated segmentation records into train/val/test by case ID."""
    seg_config = config.segmentation
    case_ids = sorted({record["case_id"] for record in records})
    rng = np.random.RandomState(config.seed)
    rng.shuffle(case_ids)

    n = len(case_ids)
    n_train = int(round(n * seg_config.train_ratio))
    n_val = int(round(n * seg_config.val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, max(n - n_train, 0))
    n_test = max(n - n_train - n_val, 0)

    split_case_ids = {
        "train": case_ids[:n_train],
        "val": case_ids[n_train : n_train + n_val],
        "test": case_ids[n_train + n_val : n_train + n_val + n_test],
    }
    case_to_split = {
        case_id: split_name
        for split_name, ids in split_case_ids.items()
        for case_id in ids
    }
    split_records = {
        "train": [],
        "val": [],
        "test": [],
    }
    for record in records:
        split_records[case_to_split[record["case_id"]]].append(record)

    manifest = {
        split_name: {
            "n_records": len(split_records[split_name]),
            "case_ids": split_case_ids[split_name],
        }
        for split_name in ("train", "val", "test")
    }
    save_json(manifest, config.paths.segmentation_split_json)
    return split_records, manifest


def train_pixel_mlp(
    train_records: List[Dict[str, np.ndarray]],
    val_records: List[Dict[str, np.ndarray]],
    config: PipelineConfig,
    device: Optional[str] = None,
) -> Tuple[PixelSegmentationModel, float]:
    dev = device or get_device()
    seg_config = config.segmentation
    model = PixelSegmentationModel(
        in_features=seg_config.mlp_features,
        hidden=seg_config.mlp_hidden,
    ).to(dev)

    dataset = PixelDataset(train_records, seg_config.mlp_samples_per_record)
    loader = DataLoader(dataset, batch_size=seg_config.mlp_batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=seg_config.mlp_lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(seg_config.mlp_epochs):
        for features, targets in loader:
            features = features.to(dev)
            targets = targets.to(dev)
            logits = model(features)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    threshold = calibrate_threshold(model, val_records or train_records, dev)
    save_checkpoint(
        {"model_state": model.state_dict(), "threshold": threshold},
        config.paths.segmenter_dir / "pixel_mlp.pt",
    )
    return model, threshold


def calibrate_threshold(
    model: PixelSegmentationModel,
    records: List[Dict[str, np.ndarray]],
    device: str,
) -> float:
    model.eval()
    best_threshold = 0.5
    best_dice = -1.0
    if not records:
        return best_threshold

    for threshold in np.arange(0.1, 0.9, 0.05):
        scores = []
        for record in records:
            features = build_feature_tensor(record["enhanced"], record["gan"])
            with torch.no_grad():
                probs = torch.sigmoid(model(features.to(device))).cpu().numpy().reshape(record["label"].shape)
            pred = probs >= threshold
            label = record["label"] > 0
            denom = pred.sum() + label.sum()
            if denom > 0:
                scores.append(float(2 * (pred & label).sum() / denom))
        if scores:
            mean_score = float(np.mean(scores))
            if mean_score > best_dice:
                best_dice = mean_score
                best_threshold = float(threshold)
    return best_threshold


def _dice_bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    bce = nn.functional.binary_cross_entropy(pred, target)
    intersection = (pred * target).sum()
    dice = 1 - (2 * intersection + 1) / (pred.sum() + target.sum() + 1)
    return bce + dice


def train_unet_fallback(
    train_records: List[Dict[str, np.ndarray]],
    config: PipelineConfig,
    device: Optional[str] = None,
) -> SegmentationUNet:
    dev = device or get_device()
    seg_config = config.segmentation
    model = SegmentationUNet(in_channels=2, base=seg_config.unet_base_filters).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=seg_config.unet_lr)

    model.train()
    for _ in range(seg_config.unet_epochs):
        for record in train_records:
            inp = np.stack(
                [
                    record["enhanced"].astype(np.float32) / 255.0,
                    record["gan"].astype(np.float32) / 255.0,
                ],
                axis=0,
            )
            target = record["label"].astype(np.float32)

            inp_tensor = torch.from_numpy(inp).unsqueeze(0).float().to(dev)
            tgt_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float().to(dev)
            pred = model(inp_tensor)
            loss = _dice_bce_loss(pred, tgt_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    save_checkpoint({"model_state": model.state_dict()}, config.paths.segmenter_dir / "unet_fallback.pt")
    return model


def _evaluate_mask(pred_mask: np.ndarray, label_mask: np.ndarray) -> Dict[str, Optional[float]]:
    metrics = {
        "dice": compute_dice(pred_mask, label_mask),
        "iou": compute_iou(pred_mask, label_mask),
        "pixel_accuracy": compute_pixel_accuracy(pred_mask, label_mask),
    }
    metrics.update(compute_precision_recall_f1(pred_mask, label_mask))
    return metrics


def evaluate_segmenter(
    records: List[Dict[str, np.ndarray]],
    segmenter: ImplantSegmenter,
    modes: Tuple[str, ...] = ("classical", "mlp", "unet", "combined"),
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """Evaluate segmentation modes on a held-out test set."""
    case_rows: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, object]] = {}

    for mode in modes:
        metric_store: Dict[str, List[float]] = {
            "dice": [],
            "iou": [],
            "pixel_accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }
        for record in records:
            pred_mask, used_method = segmenter.segment(
                record["enhanced"],
                record["gan"],
                force_mode=mode,
            )
            metrics = _evaluate_mask(pred_mask, record["label"])
            case_rows.append(
                {
                    "case_id": record["case_id"],
                    "requested_mode": mode,
                    "used_method": used_method,
                    **metrics,
                }
            )
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    metric_store[metric_name].append(float(metric_value))

        summary[mode] = {}
        for metric_name, values in metric_store.items():
            if values:
                summary[mode][metric_name] = summary_with_ci(values)

    return pd.DataFrame(case_rows), summary


def train_and_evaluate_segmenter(
    records: List[Dict[str, np.ndarray]],
    config: PipelineConfig,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Train the segmentation models and write a split-aware report."""
    split_records, split_manifest = split_segmentation_records(records, config)
    mlp_model, threshold = train_pixel_mlp(
        split_records["train"],
        split_records["val"],
        config,
        device,
    )
    _ = mlp_model
    _ = train_unet_fallback(split_records["train"], config, device)

    dev = device or get_device()
    segmenter = ImplantSegmenter(
        device=dev,
        mlp_checkpoint=config.paths.segmenter_dir / "pixel_mlp.pt",
        unet_checkpoint=config.paths.segmenter_dir / "unet_fallback.pt",
        threshold=threshold,
    )
    case_metrics_df, summary = evaluate_segmenter(split_records["test"], segmenter)

    report = {
        "split_manifest": split_manifest,
        "mlp_threshold": threshold,
        "test_summary": summary,
    }
    case_metrics_df.to_csv(config.paths.segmentation_case_metrics_csv, index=False)
    save_json(report, config.paths.segmentation_report_json)
    return report
