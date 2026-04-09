"""Segmentation models for implant detection (D1 §3.8).

Three tiers:
  1. Classical pipeline: Otsu + morphology + Canny + geometric fitting
  2. Pixel-level MLP classifier (5-feature input)
  3. U-Net segmentation fallback
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ── Tier 1: Classical Pipeline ──────────────────────────────────────────────

def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Apply Otsu thresholding (D1 §3.8)."""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def morphological_cleanup(mask: np.ndarray, kernel_size: int = 5, iterations: int = 2) -> np.ndarray:
    """Apply morphological closing and median blur."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    cleaned = cv2.medianBlur(cleaned, 5)
    return cleaned


def canny_edges(image: np.ndarray, low: int = 25, high: int = 100) -> np.ndarray:
    """Canny edge detection."""
    return cv2.Canny(image, low, high)


def analyze_components(mask: np.ndarray, min_area: int = 800) -> List[Dict[str, Any]]:
    """Connected component analysis with geometric features."""
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    components = []
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x, y, w, h, _ = stats[idx]
        component_mask = np.where(labels == idx, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        aspect = float(w / h) if h > 0 else float("inf")
        extent = float(area) / float(w * h) if w * h > 0 else 0.0
        circularity = 4.0 * math.pi * area / (perimeter ** 2 + 1e-6) if perimeter > 0 else 0.0

        components.append({
            "id": idx,
            "area": float(area),
            "bbox": (int(x), int(y), int(w), int(h)),
            "centroid": (float(centroids[idx][0]), float(centroids[idx][1])),
            "aspect_ratio": aspect,
            "extent": extent,
            "circularity": circularity,
            "contour": contour,
            "perimeter": float(perimeter),
        })

    components.sort(key=lambda c: c["area"], reverse=True)
    return components


def select_cup_component(components: Sequence[Dict]) -> Optional[Dict]:
    """Select the acetabular cup component (roughly circular, upper region)."""
    for comp in components:
        if (0.6 <= comp["aspect_ratio"] <= 1.6 and
                comp["circularity"] >= 0.35 and
                comp["extent"] >= 0.3):
            return comp
    return components[0] if components else None


def _reference_y_from_cup(cup: Optional[Dict]) -> Optional[float]:
    """Extract a vertical reference coordinate from either a component or geometry dict."""
    if not cup:
        return None
    if "centroid" in cup and isinstance(cup["centroid"], (tuple, list)) and len(cup["centroid"]) >= 2:
        return float(cup["centroid"][1])
    if "center" in cup and isinstance(cup["center"], (tuple, list)) and len(cup["center"]) >= 2:
        return float(cup["center"][1])
    return None


def select_stem_component(components: Sequence[Dict], cup: Optional[Dict]) -> Optional[Dict]:
    """Select the femoral stem component (elongated, below cup).

    `cup` may be either:
      - a connected-component record from `analyze_components()` with `centroid`
      - a reconstruction geometry dict with `center`
    """
    candidates = [c for c in components if c is not cup]
    elongated = [c for c in candidates if c["aspect_ratio"] >= 1.8 or c["aspect_ratio"] <= 0.55]
    elongated.sort(key=lambda c: c["area"], reverse=True)
    if elongated:
        return elongated[0]
    cup_y = _reference_y_from_cup(cup)
    if cup_y is not None and candidates:
        lower = [c for c in candidates if c["centroid"][1] > cup_y + 20]
        lower.sort(key=lambda c: c["area"], reverse=True)
        if lower:
            return lower[0]
    return candidates[0] if candidates else None


def classical_segmentation(image: np.ndarray) -> Tuple[np.ndarray, Optional[Dict], Optional[Dict]]:
    """Full classical segmentation pipeline.

    Returns:
        Tuple of (binary mask, cup geometry, stem geometry).
    """
    # Otsu
    binary = otsu_threshold(image)
    # Cleanup
    cleaned = morphological_cleanup(binary)
    # Analyze components
    components = analyze_components(cleaned)
    cup = select_cup_component(components)
    stem = select_stem_component(components, cup)

    return cleaned, cup, stem


# ── Tier 2: Pixel-Level MLP ─────────────────────────────────────────────────

class PixelSegmentationModel(nn.Module):
    """5-feature pixel-level MLP classifier (D1 §3.8, Appendix E).

    Features:
      1. CLAHE intensity (normalised)
      2. GAN output intensity (normalised)
      3. Gradient magnitude (normalised)
      4. Normalised x coordinate
      5. Normalised y coordinate
    """

    def __init__(self, in_features: int = 5, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PixelDataset(Dataset):
    """Dataset that samples random pixels from image records for MLP training."""

    def __init__(self, records: Sequence[Dict[str, np.ndarray]], samples_per_record: int):
        self.records = list(records)
        self.samples_per_record = samples_per_record
        self.length = len(self.records) * max(1, samples_per_record)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        record_idx = idx // max(1, self.samples_per_record)
        record = self.records[record_idx % len(self.records)]
        h, w = record["label"].shape
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)

        feat = np.array([
            record["enhanced"][y, x] / 255.0,
            record["gan"][y, x] / 255.0,
            record["grad"][y, x] / 255.0,
            x / max(w - 1, 1),
            y / max(h - 1, 1),
        ], dtype=np.float32)
        target = np.array([record["label"][y, x]], dtype=np.float32)
        return feat, target


def gradient_map(image: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude map using Sobel filters."""
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)


def build_feature_tensor(enhanced: np.ndarray, gan_image: np.ndarray) -> torch.Tensor:
    """Build feature tensor for full-image MLP prediction."""
    grad = gradient_map(enhanced)
    h, w = enhanced.shape
    x_grid = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
    y_grid = np.tile(np.linspace(0, 1, h, dtype=np.float32)[:, None], (1, w))

    features = np.stack([
        enhanced.astype(np.float32) / 255.0,
        gan_image.astype(np.float32) / 255.0,
        grad.astype(np.float32) / 255.0,
        x_grid,
        y_grid,
    ], axis=-1)
    return torch.from_numpy(features.reshape(-1, 5))


# ── Tier 3: U-Net Segmentation Fallback ─────────────────────────────────────

class SegmentationUNet(nn.Module):
    """Lightweight U-Net for binary implant segmentation (fallback).

    4-level encoder-decoder with base 32 filters.
    Input: 2-channel (enhanced + GAN output), Output: 1-channel binary mask.
    """

    def __init__(self, in_channels: int = 2, base: int = 32):
        super().__init__()

        def _enc_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        def _dec_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.enc1 = _enc_block(in_channels, base)
        self.enc2 = _enc_block(base, base * 2)
        self.enc3 = _enc_block(base * 2, base * 4)
        self.enc4 = _enc_block(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _dec_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _dec_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _dec_block(base * 2, base)

        self.final = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))


# ── Combined Segmenter ──────────────────────────────────────────────────────

class ImplantSegmenter:
    """Combined segmentation pipeline: classical → MLP → U-Net fallback.

    Applies classical segmentation first. If confidence is low,
    uses MLP pixel classifier. If still low, falls back to U-Net.
    """

    def __init__(
        self,
        device: str = "cpu",
        mlp_checkpoint: Optional[Path] = None,
        unet_checkpoint: Optional[Path] = None,
        threshold: float = 0.5,
        fallback_confidence: float = 0.3,
    ):
        self.device = device
        self.threshold = threshold
        self.fallback_confidence = fallback_confidence

        # Load MLP if available
        self.mlp = None
        if mlp_checkpoint and mlp_checkpoint.exists():
            self.mlp = PixelSegmentationModel().to(device)
            state = torch.load(mlp_checkpoint, map_location=device)
            self.mlp.load_state_dict(state["model_state"])
            self.mlp.eval()
            self.threshold = state.get("threshold", threshold)

        # Load U-Net if available
        self.unet = None
        if unet_checkpoint and unet_checkpoint.exists():
            self.unet = SegmentationUNet().to(device)
            state = torch.load(unet_checkpoint, map_location=device)
            self.unet.load_state_dict(state["model_state"])
            self.unet.eval()

    def segment(
        self,
        enhanced: np.ndarray,
        gan_image: np.ndarray,
        force_mode: str = "combined",
    ) -> Tuple[np.ndarray, str]:
        """Run segmentation pipeline.

        Args:
            enhanced: CLAHE-enhanced image (uint8).
            gan_image: GAN output image (uint8).
            force_mode: One of 'combined', 'classical', 'mlp', or 'unet'.

        Returns:
            Tuple of (binary mask, method used).
        """
        mode = force_mode.lower().strip()
        classical_mask, _, _ = classical_segmentation(enhanced)
        classical_valid = self._is_plausible_mask(classical_mask)

        if mode == "classical":
            return classical_mask, "classical"

        if classical_valid and self.mlp is None and self.unet is None:
            return classical_mask, "classical"

        if self.mlp is not None and mode in {"combined", "mlp"}:
            mask, confidence = self._mlp_segment(enhanced, gan_image)
            if mode == "mlp":
                if self._is_plausible_mask(mask):
                    return mask, "mlp"
                return classical_mask, "mlp_unavailable_or_implausible"
            if confidence >= self.fallback_confidence and self._is_plausible_mask(mask):
                return mask, "mlp"

        if classical_valid:
            return classical_mask, "classical_fallback"

        if self.unet is not None and mode in {"combined", "unet"}:
            mask = self._unet_segment(enhanced, gan_image)
            if mode == "unet":
                if self._is_plausible_mask(mask):
                    return mask, "unet"
                return classical_mask, "unet_unavailable_or_implausible"
            if self._is_plausible_mask(mask):
                return mask, "unet"

        if self.mlp is not None:
            mask, _ = self._mlp_segment(enhanced, gan_image)
            if np.count_nonzero(mask) > 0:
                return mask, "mlp_low_confidence"

        return classical_mask, "classical_fallback"

    def _mlp_segment(self, enhanced: np.ndarray, gan_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """MLP pixel-level segmentation."""
        features = build_feature_tensor(enhanced, gan_image)
        with torch.no_grad():
            logits = self.mlp(features.to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy().reshape(enhanced.shape)

        mask = (probs >= self.threshold).astype(np.uint8) * 255
        mask = morphological_cleanup(mask)
        confidence = float(np.mean(np.abs(probs - 0.5)))  # Mean certainty
        return mask, confidence

    def _unet_segment(self, enhanced: np.ndarray, gan_image: np.ndarray) -> np.ndarray:
        """U-Net segmentation fallback."""
        # Stack as 2-channel input
        enh_norm = enhanced.astype(np.float32) / 255.0
        gan_norm = gan_image.astype(np.float32) / 255.0
        inp = np.stack([enh_norm, gan_norm], axis=0)
        inp_tensor = torch.from_numpy(inp).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            pred = self.unet(inp_tensor)
            mask = (pred.squeeze().cpu().numpy() >= 0.5).astype(np.uint8) * 255

        return morphological_cleanup(mask)

    @staticmethod
    def _is_plausible_mask(mask: np.ndarray) -> bool:
        foreground = float(np.count_nonzero(mask))
        total = float(mask.size)
        if foreground <= 0:
            return False

        ratio = foreground / total
        if ratio < 0.001 or ratio > 0.40:
            return False

        components = analyze_components(mask, min_area=200)
        return len(components) > 0
