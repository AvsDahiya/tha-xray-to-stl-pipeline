"""I/O utilities: image loading, NIfTI handling, checkpoint management."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


def load_image(path: Path, grayscale: bool = True) -> np.ndarray:
    """Load an image from disk.

    Args:
        path: Path to the image file (PNG, JPG, etc.).
        grayscale: If True, load as single-channel grayscale.

    Returns:
        numpy array of shape (H, W) for grayscale or (H, W, 3) for colour.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return img


def _extract_nifti_slice_array(path: Path) -> np.ndarray:
    import nibabel as nib

    volume = nib.load(str(path))
    data = volume.get_fdata()

    if data.ndim == 3:
        if data.shape[-1] == 1:
            slice_ = data[:, :, 0]
        else:
            idx = data.shape[2] // 2
            slice_ = data[:, :, idx]
    elif data.ndim == 2:
        slice_ = data
    else:
        slice_ = data.reshape(-1, data.shape[-2], data.shape[-1])[data.shape[0] // 2]

    return slice_.astype(np.float32)


def load_nifti_midslice(path: Path) -> np.ndarray:
    """Extract the central axial slice from a NIfTI volume.

    The volume is loaded, the middle slice along the third axis is selected,
    and the result is normalised to uint8 [0, 255].

    Args:
        path: Path to .nii or .nii.gz file.

    Returns:
        2D numpy array (H, W) of dtype uint8.
    """
    slice_ = _extract_nifti_slice_array(path)
    vmin, vmax = slice_.min(), slice_.max()
    if vmax > vmin:
        slice_ = (slice_ - vmin) / (vmax - vmin)
    else:
        slice_ = np.zeros_like(slice_)

    return (slice_ * 255.0).clip(0, 255).astype(np.uint8)


def load_nifti_label_slice(path: Path) -> np.ndarray:
    """Extract a central NIfTI label slice while preserving integer labels."""
    slice_ = _extract_nifti_slice_array(path)
    return np.rint(slice_).astype(np.uint8)


def save_image(image: np.ndarray, path: Path) -> None:
    """Save an image to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def save_checkpoint(
    state: Dict[str, Any],
    path: Path,
    is_best: bool = False,
) -> None:
    """Save a PyTorch checkpoint.

    Args:
        state: Dictionary containing model state, optimiser state, epoch, etc.
        path: Where to save the checkpoint.
        is_best: If True, also save a copy as 'best_model.pt' in the same dir.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(state, path)
    if is_best:
        best_path = path.parent / "best_model.pt"
        _atomic_torch_save(state, best_path)


def _atomic_torch_save(state: Dict[str, Any], path: Path) -> None:
    """Write a checkpoint atomically to avoid partial files on interruption/failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def load_checkpoint(
    path: Path,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a PyTorch checkpoint.

    Args:
        path: Path to the checkpoint file.
        device: Device to map tensors to.

    Returns:
        Dictionary containing the checkpoint contents.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_json(data: Any, path: Path) -> None:
    """Save data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Load data from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def natural_sort_key(path: Path) -> List:
    """Generate a sort key for natural ordering of filenames."""
    parts: List = []
    for token in re.findall(r"\d+|\D+", path.name.lower()):
        if token.isdigit():
            parts.append((0, int(token)))
        else:
            parts.append((1, token))
    return parts


def sanitize_id(value: str) -> str:
    """Sanitize identifiers for filenames and JSON keys."""
    value = value.strip()
    value = re.sub(r"[^\w.-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._") or "unknown"


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_paths(
    directory: Path,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> List[Path]:
    """Recursively find all image files in a directory.

    Args:
        directory: Root directory to search.
        extensions: Tuple of valid file extensions.

    Returns:
        Sorted list of image file paths.
    """
    paths = []
    for ext in extensions:
        paths.extend(directory.rglob(f"*{ext}"))
        paths.extend(directory.rglob(f"*{ext.upper()}"))
    return sorted(set(paths), key=natural_sort_key)


def get_nifti_paths(directory: Path) -> List[Path]:
    """Find all NIfTI files in a directory."""
    paths = list(directory.glob("*.nii*"))
    return sorted(paths, key=natural_sort_key)


def latest_epoch_checkpoint(run_dir: Path) -> Optional[Path]:
    """Return the latest epoch checkpoint in a run directory, if any."""
    checkpoints = sorted(run_dir.glob("epoch_*.pt"), key=natural_sort_key)
    return checkpoints[-1] if checkpoints else None


def checkpoint_is_valid(
    path: Path,
    device: str = "cpu",
) -> bool:
    """Return True when a checkpoint file can be loaded successfully."""
    if not path.exists() or not path.is_file():
        return False
    try:
        load_checkpoint(path, device=device)
        return True
    except Exception:
        return False


def latest_valid_epoch_checkpoint(
    run_dir: Path,
    device: str = "cpu",
) -> Optional[Path]:
    """Return the newest loadable epoch checkpoint, skipping corrupted files."""
    checkpoints = sorted(run_dir.glob("epoch_*.pt"), key=natural_sort_key, reverse=True)
    for checkpoint in checkpoints:
        if checkpoint_is_valid(checkpoint, device=device):
            return checkpoint
    return None


def best_resume_checkpoint(
    run_dir: Path,
    device: str = "cpu",
) -> Optional[Path]:
    """Return the best checkpoint to resume from.

    Preference order:
    1. Newest valid epoch checkpoint
    2. Valid best-model checkpoint
    """
    epoch_checkpoint = latest_valid_epoch_checkpoint(run_dir, device=device)
    if epoch_checkpoint is not None:
        return epoch_checkpoint

    best_path = run_dir / "best_model.pt"
    if checkpoint_is_valid(best_path, device=device):
        return best_path
    return None


def prune_epoch_checkpoints(
    run_dir: Path,
    keep_last: int = 3,
    keep_paths: Optional[List[Path]] = None,
) -> List[Path]:
    """Delete old epoch checkpoints while preserving the newest few and protected paths."""
    checkpoints = sorted(run_dir.glob("epoch_*.pt"), key=natural_sort_key)
    if keep_last < 0:
        keep_last = 0
    protected = {path.resolve() for path in (keep_paths or []) if path and path.exists()}
    to_keep = set(checkpoints[-keep_last:] if keep_last > 0 else [])
    kept_resolved = {path.resolve() for path in to_keep if path.exists()}
    protected |= kept_resolved

    removed: List[Path] = []
    for checkpoint in checkpoints:
        try:
            resolved = checkpoint.resolve()
        except OSError:
            resolved = checkpoint
        if resolved in protected:
            continue
        try:
            checkpoint.unlink()
            removed.append(checkpoint)
        except OSError:
            continue
    return removed
