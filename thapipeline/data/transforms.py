"""Image preprocessing and augmentation transforms (D1 §3.6).

Four-stage pipeline:
  1. Intensity normalisation → [-1, 1]
  2. Spatial standardisation → 512×512
  3. CLAHE contrast enhancement
  4. Paired data augmentation (training only)
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch


class IntensityNormalize:
    """Min-max normalise image to [-1, 1] range (D1 §3.6).

    I_norm = 2 * (I - I_min) / (I_max - I_min) - 1
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = 2.0 * (img - vmin) / (vmax - vmin) - 1.0
        else:
            img = np.zeros_like(img)
        return img


class CenterCropAndResize:
    """Central square crop and resize to target size (D1 §3.6).

    Crops the central `crop_ratio` portion of the image to a square,
    then resizes to `target_size` using bilinear interpolation.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        crop_ratio: float = 0.80,
    ):
        self.target_size = target_size
        self.crop_ratio = crop_ratio

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        # Compute central crop region, then pad to a square field of view.
        crop_h = int(h * self.crop_ratio)
        crop_w = int(w * self.crop_ratio)
        y_start = max((h - crop_h) // 2, 0)
        x_start = max((w - crop_w) // 2, 0)
        cropped = image[y_start : y_start + crop_h, x_start : x_start + crop_w]

        pad_side = max(cropped.shape[:2])
        pad_y = pad_side - cropped.shape[0]
        pad_x = pad_side - cropped.shape[1]
        top = pad_y // 2
        bottom = pad_y - top
        left = pad_x // 2
        right = pad_x - left
        squared = cv2.copyMakeBorder(
            cropped,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        resized = cv2.resize(squared, self.target_size, interpolation=cv2.INTER_LINEAR)
        return resized


class ApplyCLAHE:
    """Contrast Limited Adaptive Histogram Equalization (D1 §3.6).

    Applied in [0, 255] domain with tile grid 8×8 and clip limit 2.0.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Ensure uint8 for CLAHE
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
            else:
                img_uint8 = image.clip(0, 255).astype(np.uint8)
        else:
            img_uint8 = image

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid,
        )
        return clahe.apply(img_uint8)


class PairedRandomAugment:
    """Random augmentations applied identically to paired (pre, post) images (D1 §3.6).

    Augmentations:
      - Random horizontal flip (p=0.5) for bilateral symmetry
      - Random rotation within ±5°
      - Brightness and contrast jitter ±15%
      - Additive Gaussian noise (σ=0.02)

    The same random state is applied to both images in a pair.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotation_range: float = 5.0,
        brightness_range: float = 0.15,
        contrast_range: float = 0.15,
        noise_std: float = 0.02,
        augment_prob: float = 0.5,
    ):
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.augment_prob = augment_prob

    def __call__(
        self,
        pre: np.ndarray,
        post: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply identical random augmentations to pre/post pair.

        Args:
            pre: Pre-operative image (normalised float32, [-1, 1]).
            post: Post-operative image (normalised float32, [-1, 1]).

        Returns:
            Tuple of augmented (pre, post) images.
        """
        if random.random() > self.augment_prob:
            return pre, post

        # Determine random params (shared for both images)
        do_flip = random.random() < self.flip_prob
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        brightness_delta = random.uniform(-self.brightness_range, self.brightness_range)
        contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)

        pre = self._apply(pre, do_flip, angle, brightness_delta, contrast_factor)
        post = self._apply(post, do_flip, angle, brightness_delta, contrast_factor)

        return pre, post

    def _apply(
        self,
        image: np.ndarray,
        do_flip: bool,
        angle: float,
        brightness_delta: float,
        contrast_factor: float,
    ) -> np.ndarray:
        img = image.copy()

        # Horizontal flip
        if do_flip:
            img = np.fliplr(img).copy()

        # Rotation
        if abs(angle) > 0.01:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=0)

        # Brightness and contrast
        img = img * contrast_factor + brightness_delta

        # Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, img.shape).astype(np.float32)
            img = img + noise

        # Clamp to [-1, 1]
        img = np.clip(img, -1.0, 1.0)
        return img


class PreprocessPipeline:
    """Full preprocessing pipeline combining all stages.

    Usage:
        pipeline = PreprocessPipeline()
        result = pipeline(raw_image)
        # result contains 'resized', 'enhanced' (CLAHE), 'normalized' keys
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        crop_ratio: float = 0.80,
        clahe_clip: float = 2.0,
        clahe_grid: Tuple[int, int] = (8, 8),
    ):
        self.crop_resize = CenterCropAndResize(target_size, crop_ratio)
        self.clahe = ApplyCLAHE(clahe_clip, clahe_grid)
        self.normalize = IntensityNormalize()

    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Process a raw radiograph through all stages.

        Args:
            image: Raw grayscale radiograph (uint8).

        Returns:
            Dictionary with keys:
              - 'resized': Cropped and resized to 512×512 (uint8)
              - 'enhanced': After CLAHE enhancement (uint8)
              - 'normalized': Normalised to [-1, 1] (float32)
        """
        resized = self.crop_resize(image)
        enhanced = self.clahe(resized)
        normalized = self.normalize(enhanced)
        return {
            "resized": resized,
            "enhanced": enhanced,
            "normalized": normalized,
        }


def to_tensor(normalized: np.ndarray, device: Optional[str] = None) -> torch.Tensor:
    """Convert normalised image to PyTorch tensor (1, 1, H, W).

    Args:
        normalized: Image in [-1, 1] range, shape (H, W).
        device: Target device string.

    Returns:
        Tensor of shape (1, 1, H, W).
    """
    t = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor back to uint8 image.

    Args:
        tensor: Tensor in [-1, 1] range.

    Returns:
        numpy array of dtype uint8 in [0, 255].
    """
    arr = tensor.squeeze().detach().cpu().numpy()
    return ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def normalize_preprocessed(image: np.ndarray) -> np.ndarray:
    """Normalize a preprocessed CLAHE image from [0, 255] to [-1, 1]."""
    return IntensityNormalize()(image)
