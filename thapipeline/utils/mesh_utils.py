"""Mesh utility functions: validation, repair, projection, and STL export."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import trimesh


def check_euler(mesh: trimesh.Trimesh) -> Dict[str, object]:
    """Check Euler characteristic (χ = V - E + F = 2 for closed manifold)."""
    euler = mesh.euler_number
    return {
        "euler": int(euler),
        "target": 2,
        "valid": euler == 2,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "is_watertight": mesh.is_watertight,
    }


def export_stl(mesh: trimesh.Trimesh, path: Path) -> bool:
    """Export mesh as binary STL file.

    Returns True if export succeeded and mesh is watertight.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type="stl")
    return mesh.is_watertight


def measure_mesh_dimensions(mesh: trimesh.Trimesh) -> Dict[str, float]:
    """Measure bounding box dimensions of a mesh in mm."""
    bounds = mesh.bounds
    dims = bounds[1] - bounds[0]
    return {
        "width_mm": float(dims[0]),
        "height_mm": float(dims[1]),
        "depth_mm": float(dims[2]),
        "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
    }


def project_mesh_to_mask(
    mesh: trimesh.Trimesh,
    shape: Tuple[int, int],
    dpi: float = 150.0,
) -> np.ndarray:
    """Project a 3D mesh to a 2D binary silhouette mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    if mesh is None:
        return mask

    px_per_mm = dpi / 25.4
    verts_xy = np.asarray(mesh.vertices[:, :2], dtype=np.float64)
    finite = np.isfinite(verts_xy).all(axis=1)
    verts_xy = verts_xy[finite]
    if verts_xy.size == 0:
        return mask

    coords = np.round(verts_xy * px_per_mm).astype(int)
    h, w = shape

    valid = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < w)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < h)
    )
    coords = coords[valid]
    if coords.size == 0:
        return mask

    mask[coords[:, 1], coords[:, 0]] = 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return mask
