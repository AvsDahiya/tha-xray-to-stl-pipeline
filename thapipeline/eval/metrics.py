"""Evaluation metrics for the THA pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial import cKDTree

from thapipeline.utils.mesh_utils import project_mesh_to_mask


def compute_ssim(generated: np.ndarray, target: np.ndarray, data_range: float = 255.0) -> float:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(generated, target, data_range=data_range))
    except ImportError:
        generated = generated.astype(np.float64)
        target = target.astype(np.float64)
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        mu_x = cv2.GaussianBlur(generated, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(target, (11, 11), 1.5)
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = cv2.GaussianBlur(generated * generated, (11, 11), 1.5) - mu_x_sq
        sigma_y_sq = cv2.GaussianBlur(target * target, (11, 11), 1.5) - mu_y_sq
        sigma_xy = cv2.GaussianBlur(generated * target, (11, 11), 1.5) - mu_xy
        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        return float((numerator / np.maximum(denominator, 1e-12)).mean())


def compute_psnr(generated: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((generated.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))


def compute_dice(pred: np.ndarray, target: np.ndarray) -> Optional[float]:
    pred_bin = pred > 0
    target_bin = target > 0
    denom = pred_bin.sum() + target_bin.sum()
    if denom == 0:
        return None
    return float(2 * (pred_bin & target_bin).sum() / denom)


def compute_iou(pred: np.ndarray, target: np.ndarray) -> Optional[float]:
    pred_bin = pred > 0
    target_bin = target > 0
    union = (pred_bin | target_bin).sum()
    if union == 0:
        return None
    return float((pred_bin & target_bin).sum() / union)


def compute_pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.shape != target.shape:
        raise ValueError("Prediction and target must have identical shapes.")
    return float(np.mean((pred > 0) == (target > 0)))


def compute_precision_recall_f1(pred: np.ndarray, target: np.ndarray) -> Dict[str, Optional[float]]:
    pred_bin = pred > 0
    target_bin = target > 0
    tp = int((pred_bin & target_bin).sum())
    fp = int((pred_bin & ~target_bin).sum())
    fn = int((~pred_bin & target_bin).sum())

    precision = float(tp / (tp + fp)) if tp + fp > 0 else None
    recall = float(tp / (tp + fn)) if tp + fn > 0 else None
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_hausdorff_distance(contour_a: np.ndarray, contour_b: np.ndarray) -> Optional[float]:
    if len(contour_a) == 0 or len(contour_b) == 0:
        return None
    tree_a = cKDTree(contour_a)
    tree_b = cKDTree(contour_b)
    dists_a, _ = tree_b.query(contour_a, k=1)
    dists_b, _ = tree_a.query(contour_b, k=1)
    return float(max(dists_a.max(), dists_b.max()))


def compute_chamfer_distance(contour_a: np.ndarray, contour_b: np.ndarray) -> Optional[float]:
    if len(contour_a) == 0 or len(contour_b) == 0:
        return None
    tree_a = cKDTree(contour_a)
    tree_b = cKDTree(contour_b)
    dists_a, _ = tree_b.query(contour_a, k=1)
    dists_b, _ = tree_a.query(contour_b, k=1)
    return float((dists_a.mean() + dists_b.mean()) / 2.0)


def compute_reprojection_dice(seg_mask: np.ndarray, proj_mask: np.ndarray) -> Optional[float]:
    return compute_dice(proj_mask, seg_mask)


def compute_reprojection_error(seg_mask: np.ndarray, proj_mask: np.ndarray) -> Optional[float]:
    seg_edges = np.column_stack(np.where(cv2.Canny(seg_mask, 60, 120) > 0))
    proj_edges = np.column_stack(np.where(cv2.Canny(proj_mask, 60, 120) > 0))
    if seg_edges.size == 0 or proj_edges.size == 0:
        return None

    tree_seg = cKDTree(seg_edges)
    tree_proj = cKDTree(proj_edges)
    fwd, _ = tree_proj.query(seg_edges, k=1)
    bwd, _ = tree_seg.query(proj_edges, k=1)
    return float((fwd.mean() + bwd.mean()) / 2.0)


def compute_dimensional_error(measured_mm: float, reference_mm: float) -> Dict[str, float]:
    abs_error = abs(measured_mm - reference_mm)
    rel_error = abs_error / reference_mm if reference_mm > 0 else float("inf")
    return {
        "measured_mm": measured_mm,
        "reference_mm": reference_mm,
        "absolute_error_mm": abs_error,
        "relative_error_pct": rel_error * 100,
        "within_2mm": abs_error <= 2.0,
    }


def _extract_inception_features(
    image_paths: Sequence[Path],
    device: str = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    try:
        from PIL import Image
        import torch
        import torchvision.transforms as T
        from pytorch_fid.inception import InceptionV3
        from torch.nn.functional import adaptive_avg_pool2d
    except ImportError as exc:
        raise RuntimeError("FID/KID dependencies are not installed.") from exc

    if not image_paths:
        return np.empty((0, 2048), dtype=np.float32)

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    transform = T.Compose(
        [
            T.Resize((299, 299)),
            T.ToTensor(),
        ]
    )

    features = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch = [transform(Image.open(path).convert("RGB")) for path in batch_paths]
            batch_tensor = torch.stack(batch).to(device)
            pred = model(batch_tensor)[0]
            if pred.shape[-1] != 1 or pred.shape[-2] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            features.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fid_from_paths(
    real_paths: Sequence[Path],
    fake_paths: Sequence[Path],
    device: str = "cpu",
    batch_size: int = 16,
) -> Optional[float]:
    if len(real_paths) < 2 or len(fake_paths) < 2:
        return None
    real_features = _extract_inception_features(real_paths, device=device, batch_size=batch_size)
    fake_features = _extract_inception_features(fake_paths, device=device, batch_size=batch_size)
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_kid_from_paths(
    real_paths: Sequence[Path],
    fake_paths: Sequence[Path],
    device: str = "cpu",
    batch_size: int = 16,
    subset_size: int = 50,
    n_subsets: int = 10,
) -> Optional[float]:
    if len(real_paths) < 2 or len(fake_paths) < 2:
        return None

    real = _extract_inception_features(real_paths, device=device, batch_size=batch_size)
    fake = _extract_inception_features(fake_paths, device=device, batch_size=batch_size)
    if len(real) < 2 or len(fake) < 2:
        return None

    rng = np.random.RandomState(42)
    subset_size = min(subset_size, len(real), len(fake))
    scores = []
    for _ in range(n_subsets):
        real_subset = real[rng.choice(len(real), subset_size, replace=False)]
        fake_subset = fake[rng.choice(len(fake), subset_size, replace=False)]
        xx = ((real_subset @ real_subset.T) / real_subset.shape[1] + 1.0) ** 3
        yy = ((fake_subset @ fake_subset.T) / fake_subset.shape[1] + 1.0) ** 3
        xy = ((real_subset @ fake_subset.T) / real_subset.shape[1] + 1.0) ** 3
        m = subset_size
        score = (xx.sum() - np.trace(xx)) / (m * (m - 1))
        score += (yy.sum() - np.trace(yy)) / (m * (m - 1))
        score -= 2 * xy.mean()
        scores.append(score)
    return float(np.mean(scores))


def compute_all_metrics(
    generated: np.ndarray,
    target: np.ndarray,
    seg_pred: Optional[np.ndarray] = None,
    seg_target: Optional[np.ndarray] = None,
    mesh_proj: Optional[np.ndarray] = None,
    seg_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "gan": {
            "ssim": compute_ssim(generated, target),
            "psnr": compute_psnr(generated, target),
        }
    }

    if seg_pred is not None and seg_target is not None:
        metrics["segmentation"] = {
            "dice": compute_dice(seg_pred, seg_target),
            "iou": compute_iou(seg_pred, seg_target),
            "pixel_accuracy": compute_pixel_accuracy(seg_pred, seg_target),
            **compute_precision_recall_f1(seg_pred, seg_target),
        }

    if mesh_proj is not None and seg_mask is not None:
        seg_edges = np.column_stack(np.where(cv2.Canny(seg_mask, 60, 120) > 0))
        proj_edges = np.column_stack(np.where(cv2.Canny(mesh_proj, 60, 120) > 0))
        metrics["reconstruction"] = {
            "reprojection_dice": compute_reprojection_dice(seg_mask, mesh_proj),
            "reprojection_error_px": compute_reprojection_error(seg_mask, mesh_proj),
            "hausdorff_px": compute_hausdorff_distance(seg_edges, proj_edges),
            "chamfer_px": compute_chamfer_distance(seg_edges, proj_edges),
        }

    return metrics
