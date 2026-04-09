"""3D reconstruction pipeline (D1 §3.9–3.11).

Reconstructs 3D implant meshes from 2D segmentation masks:
  - Acetabular cup: hemispherical primitive from min-enclosing circle
  - Femoral stem: surface of revolution from medial axis + taper
  - Reprojection optimisation via Levenberg-Marquardt
  - Mesh validation: Euler characteristic, manifold repair, smoothing
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


# ── Geometry Extraction ─────────────────────────────────────────────────────

def extract_cup_geometry(
    mask: np.ndarray,
    components: Optional[list] = None,
) -> Optional[Dict[str, float]]:
    """Extract acetabular cup geometry from segmentation mask.

    Strategy (D1 §3.8):
      1. Component analysis → find most circular component in upper region
      2. Fallback: Hough circle detection
      3. Fallback: Upper-region contour + min-enclosing circle

    Returns:
        Dictionary with 'center' and 'radius', or None.
    """
    from thapipeline.models.segmenter import analyze_components, select_cup_component

    if components is None:
        components = analyze_components(mask)

    cup_comp = select_cup_component(components)
    if cup_comp is not None:
        (cx, cy), radius = cv2.minEnclosingCircle(cup_comp["contour"])
        return {"center": (float(cx), float(cy)), "radius": float(max(radius, 8.0))}

    # Fallback: Hough circles
    circles = cv2.HoughCircles(
        mask, cv2.HOUGH_GRADIENT, dp=1.1, minDist=45,
        param1=110, param2=23, minRadius=25, maxRadius=170,
    )
    if circles is not None:
        cx, cy, r = circles[0][0]
        return {"center": (float(cx), float(cy)), "radius": float(r)}

    # Fallback: upper region contour
    cutoff = int(mask.shape[0] * 0.65)
    upper = np.zeros_like(mask)
    upper[:cutoff] = mask[:cutoff]
    contours, _ = cv2.findContours(upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        return {"center": (float(cx), float(cy)), "radius": float(max(radius, 8.0))}

    return None


def extract_stem_geometry(
    mask: np.ndarray,
    components: Optional[list] = None,
    cup: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """Extract femoral stem geometry from segmentation mask.

    Strategy:
      1. Component analysis → find elongated component below cup
      2. Ellipse fitting for orientation and dimensions
      3. Medial axis extraction for taper profile
      4. Fallback: lower region contour analysis

    Returns:
        Dictionary with 'center', 'axes', 'angle', and optionally 'taper_profile'.
    """
    from thapipeline.models.segmenter import (
        analyze_components,
        select_cup_component,
        select_stem_component,
    )

    if components is None:
        components = analyze_components(mask)

    stem_comp = select_stem_component(
        components,
        select_cup_component(components) if cup is None else cup,
    )

    if stem_comp is not None and len(stem_comp["contour"]) >= 5:
        ellipse = cv2.fitEllipse(stem_comp["contour"])
        center, axes, angle = ellipse
        major = max(axes[0], axes[1], 10.0)
        minor = max(min(axes[0], axes[1]), 6.0)

        # Extract taper profile via medial axis
        taper = _extract_taper_profile(mask, stem_comp)

        return {
            "center": (float(center[0]), float(center[1])),
            "axes": (float(major), float(minor)),
            "angle": float(angle),
            "taper_profile": taper,
        }

    # Fallback: lower region
    start = int(mask.shape[0] * 0.35)
    lower = np.zeros_like(mask)
    lower[start:] = mask[start:]
    contours, _ = cv2.findContours(lower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            return {
                "center": (float(center[0]), float(center[1])),
                "axes": (float(max(axes[0], axes[1], 10.0)), float(max(min(axes[0], axes[1]), 6.0))),
                "angle": float(angle),
                "taper_profile": None,
            }

    return None


def _extract_taper_profile(mask: np.ndarray, component: Dict) -> Optional[List[Tuple[float, float]]]:
    """Extract width profile along stem medial axis for taper modelling."""
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return None

    comp_mask = np.zeros_like(mask)
    cv2.drawContours(comp_mask, [component["contour"]], -1, 255, -1)
    skeleton = skeletonize(comp_mask > 0)

    # Get skeleton points sorted by y
    pts = np.column_stack(np.where(skeleton))
    if len(pts) < 5:
        return None

    pts = pts[pts[:, 0].argsort()]  # Sort by y (row)

    # Measure width at each skeleton point
    profile = []
    for y, x in pts[::max(1, len(pts) // 20)]:  # Sample ~20 points
        row = comp_mask[y, :]
        nonzero = np.where(row > 0)[0]
        if len(nonzero) >= 2:
            width = float(nonzero[-1] - nonzero[0])
            profile.append((float(y) / mask.shape[0], width))

    return profile if len(profile) >= 3 else None


# ── 3D Mesh Construction ────────────────────────────────────────────────────

def reconstruct_acetabular_cup(
    cup: Dict,
    dpi: float = 150.0,
    magnification: float = 1.15,
    subdivisions: int = 4,
) -> trimesh.Trimesh:
    """Build 3D hemispherical cup mesh (D1 §3.9).

    Args:
        cup: Dict with 'center' and 'radius' (in pixels).
        dpi: Image DPI for pixel-to-mm conversion.
        magnification: Radiographic magnification factor (1.15).
        subdivisions: Icosphere subdivision level.

    Returns:
        trimesh.Trimesh of the acetabular cup.
    """
    px_per_mm = dpi / 25.4
    radius_mm = (cup["radius"] / px_per_mm) / magnification

    # Clamp to realistic range
    radius_mm = max(12.5, min(radius_mm, 35.0))

    # Create hemisphere
    sphere = trimesh.creation.icosphere(radius=radius_mm, subdivisions=subdivisions)
    # Cut to hemisphere (keep z >= 0)
    vertices = sphere.vertices
    faces = sphere.faces

    # Keep faces where all vertices have z >= -0.5mm (tolerance)
    keep_mask = np.all(vertices[faces, 2] >= -0.5, axis=1)
    hemisphere = sphere.submesh([np.where(keep_mask)[0]], append=True)

    # Position
    cx_mm = cup["center"][0] / px_per_mm
    cy_mm = cup["center"][1] / px_per_mm
    hemisphere.apply_translation([cx_mm, cy_mm, 0])

    return hemisphere


def reconstruct_femoral_stem(
    stem: Dict,
    dpi: float = 150.0,
    magnification: float = 1.15,
    sections: int = 32,
) -> trimesh.Trimesh:
    """Build 3D femoral stem mesh via surface of revolution (D1 §3.9).

    Uses medial axis + exponential taper if available,
    otherwise falls back to cylinder.

    Args:
        stem: Dict with 'center', 'axes', 'angle', optionally 'taper_profile'.
        dpi: Image DPI for conversion.
        magnification: Radiographic magnification factor.
        sections: Number of angular sections for revolution surface.

    Returns:
        trimesh.Trimesh of the femoral stem.
    """
    px_per_mm = dpi / 25.4

    major_mm = (max(stem["axes"]) / px_per_mm) / magnification
    minor_mm = (min(stem["axes"]) / px_per_mm) / magnification

    # Clamp to realistic range
    major_mm = max(45, min(major_mm, 130))
    minor_mm = max(5, min(minor_mm, 25))

    taper = stem.get("taper_profile")

    if taper and len(taper) >= 3:
        # Build surface of revolution from taper profile
        mesh = _build_revolution_surface(taper, major_mm, minor_mm, sections, px_per_mm)
    else:
        # Fallback: tapered cylinder
        mesh = _build_tapered_cylinder(major_mm, minor_mm, sections)

    # Rotate to match stem angle
    angle_rad = math.radians(stem["angle"])
    rotation = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
    mesh.apply_transform(rotation)

    # Position
    cx_mm = stem["center"][0] / px_per_mm
    cy_mm = stem["center"][1] / px_per_mm
    mesh.apply_translation([cx_mm, cy_mm, 0])

    return mesh


def _build_tapered_cylinder(
    length: float,
    max_radius: float,
    sections: int,
) -> trimesh.Trimesh:
    """Build a tapered cylinder (wider at top, narrower at bottom)."""
    n_rings = 20
    vertices = []
    faces = []

    for i in range(n_rings):
        t = i / (n_rings - 1)
        y = -length / 2 + t * length
        # Exponential taper: wider at top (proximal), narrower at bottom (distal)
        radius = max_radius * math.exp(-1.5 * t)
        radius = max(radius, max_radius * 0.15)  # Min radius

        for j in range(sections):
            theta = 2 * math.pi * j / sections
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Create faces between rings
    for i in range(n_rings - 1):
        for j in range(sections):
            v0 = i * sections + j
            v1 = i * sections + (j + 1) % sections
            v2 = (i + 1) * sections + j
            v3 = (i + 1) * sections + (j + 1) % sections
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    faces = np.array(faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def _build_revolution_surface(
    taper: List[Tuple[float, float]],
    length: float,
    max_radius: float,
    sections: int,
    px_per_mm: float,
) -> trimesh.Trimesh:
    """Build surface of revolution from actual taper profile."""
    vertices = []
    faces = []

    for i, (t_norm, width_px) in enumerate(taper):
        y = -length / 2 + t_norm * length
        radius = min((width_px / 2) / px_per_mm, max_radius)
        radius = max(radius, 1.0)

        for j in range(sections):
            theta = 2 * math.pi * j / sections
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            vertices.append([x, y, z])

    vertices = np.array(vertices)
    n_rings = len(taper)

    for i in range(n_rings - 1):
        for j in range(sections):
            v0 = i * sections + j
            v1 = i * sections + (j + 1) % sections
            v2 = (i + 1) * sections + j
            v3 = (i + 1) * sections + (j + 1) % sections
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    faces = np.array(faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


# ── Reprojection Optimisation ───────────────────────────────────────────────

def reprojection_optimise(
    mesh: trimesh.Trimesh,
    contour_2d: np.ndarray,
    max_iterations: int = 50,
) -> trimesh.Trimesh:
    """Optimise mesh parameters to match 2D silhouette (D1 §3.10).

    Uses Levenberg-Marquardt to minimise distance between projected
    mesh silhouette and 2D segmentation contour.

    Args:
        mesh: Initial 3D mesh.
        contour_2d: 2D contour points from segmentation.
        max_iterations: Max LM iterations.

    Returns:
        Optimised mesh.
    """
    if len(contour_2d) < 5:
        return mesh

    # Initial parameters: [tx, ty, tz, sx, sy, sz, rx, ry, rz]
    x0 = np.zeros(9)
    x0[3:6] = 1.0  # Initial scale = 1

    original_vertices = mesh.vertices.copy()

    def residuals(params):
        tx, ty, tz, sx, sy, sz, rx, ry, rz = params

        # Apply transform
        verts = original_vertices.copy()
        verts[:, 0] *= max(sx, 0.1)
        verts[:, 1] *= max(sy, 0.1)
        verts[:, 2] *= max(sz, 0.1)

        # Simple rotation (small angles)
        cos_r, sin_r = math.cos(rz), math.sin(rz)
        x_rot = verts[:, 0] * cos_r - verts[:, 1] * sin_r
        y_rot = verts[:, 0] * sin_r + verts[:, 1] * cos_r
        verts[:, 0] = x_rot
        verts[:, 1] = y_rot

        verts[:, 0] += tx
        verts[:, 1] += ty

        # Project to 2D (orthographic)
        projected = verts[:, :2]

        # Compute distance to 2D contour
        tree = cKDTree(contour_2d)
        dists, _ = tree.query(projected, k=1)
        return dists

    result = least_squares(
        residuals, x0,
        method="lm",
        max_nfev=max_iterations,
    )

    # Apply optimised transform
    params = result.x
    tx, ty, tz, sx, sy, sz, rx, ry, rz = params
    verts = original_vertices.copy()
    verts[:, 0] *= max(sx, 0.1)
    verts[:, 1] *= max(sy, 0.1)
    verts[:, 2] *= max(sz, 0.1)

    cos_r, sin_r = math.cos(rz), math.sin(rz)
    x_rot = verts[:, 0] * cos_r - verts[:, 1] * sin_r
    y_rot = verts[:, 0] * sin_r + verts[:, 1] * cos_r
    verts[:, 0] = x_rot + tx
    verts[:, 1] = y_rot + ty
    verts[:, 2] += tz

    optimised = mesh.copy()
    optimised.vertices = verts
    return optimised


# ── Mesh Validation & Repair ────────────────────────────────────────────────

def validate_and_fix_mesh(
    mesh: trimesh.Trimesh,
    smooth_iterations: int = 5,
    smooth_lambda: float = 0.5,
) -> Tuple[trimesh.Trimesh, bool]:
    """Validate mesh watertightness and repair if needed (D1 §3.11).

    Checks:
      1. Euler characteristic χ = V - E + F = 2
      2. Manifold status
      3. Consistent face winding

    Repairs:
      1. Fill holes
      2. Fix normals
      3. Laplacian smoothing
      4. Remove degenerate faces

    Returns:
        Tuple of (repaired mesh, is_watertight).
    """
    # Remove degenerate and duplicate faces using the trimesh API available
    # in the current environment. Newer versions prefer update_faces(...).
    try:
        if hasattr(mesh, "nondegenerate_faces") and hasattr(mesh, "update_faces"):
            mesh.update_faces(mesh.nondegenerate_faces())
        elif hasattr(mesh, "remove_degenerate_faces"):
            mesh.remove_degenerate_faces()
    except Exception:
        pass

    try:
        if hasattr(mesh, "unique_faces") and hasattr(mesh, "update_faces"):
            mesh.update_faces(mesh.unique_faces())
        elif hasattr(mesh, "remove_duplicate_faces"):
            mesh.remove_duplicate_faces()
    except Exception:
        pass

    # Fix face winding
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)

    # Fill holes
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)

    # Laplacian smoothing
    if smooth_iterations > 0:
        mesh = _laplacian_smooth(mesh, smooth_iterations, smooth_lambda)

    # Check Euler characteristic
    euler = mesh.euler_number
    is_valid = mesh.is_watertight and euler == 2

    return mesh, is_valid


def _laplacian_smooth(
    mesh: trimesh.Trimesh,
    iterations: int,
    lam: float,
) -> trimesh.Trimesh:
    """Apply Laplacian smoothing to mesh vertices."""
    original_vertices = mesh.vertices.copy()
    try:
        with np.errstate(all="ignore"):
            trimesh.smoothing.filter_laplacian(mesh, iterations=iterations, lamb=lam)
        if not np.isfinite(mesh.vertices).all():
            mesh.vertices = original_vertices
    except Exception:
        mesh.vertices = original_vertices
    return mesh


# ── Full Reconstruction Pipeline ────────────────────────────────────────────

def reconstruct_from_mask(
    mask: np.ndarray,
    dpi: float = 150.0,
    magnification: float = 1.15,
    optimize: bool = True,
    smooth_iterations: int = 5,
) -> Tuple[Optional[trimesh.Trimesh], Dict[str, Any]]:
    """Full 3D reconstruction from segmentation mask.

    Args:
        mask: Binary segmentation mask (uint8, 512×512).
        dpi: Image DPI.
        magnification: Radiographic magnification factor.
        optimize: Whether to run reprojection optimisation.
        smooth_iterations: Laplacian smoothing iterations.

    Returns:
        Tuple of (combined mesh or None, metadata dict).
    """
    from thapipeline.models.segmenter import analyze_components

    components = analyze_components(mask)
    cup_geom = extract_cup_geometry(mask, components)
    stem_geom = extract_stem_geometry(mask, components, cup_geom)

    meshes = []
    metadata = {"cup": None, "stem": None, "watertight": False, "euler": None}

    if cup_geom is not None:
        cup_mesh = reconstruct_acetabular_cup(cup_geom, dpi, magnification)
        meshes.append(cup_mesh)
        metadata["cup"] = {
            "radius_mm": cup_geom["radius"] / (dpi / 25.4) / magnification,
            "center_px": cup_geom["center"],
        }

    if stem_geom is not None:
        stem_mesh = reconstruct_femoral_stem(stem_geom, dpi, magnification)
        meshes.append(stem_mesh)
        metadata["stem"] = {
            "length_mm": max(stem_geom["axes"]) / (dpi / 25.4) / magnification,
            "width_mm": min(stem_geom["axes"]) / (dpi / 25.4) / magnification,
            "angle_deg": stem_geom["angle"],
        }

    if not meshes:
        return None, metadata

    # Combine meshes
    combined = trimesh.util.concatenate(meshes)

    # Reprojection optimisation
    if optimize:
        contour_pts = np.column_stack(np.where(cv2.Canny(mask, 25, 100) > 0)).astype(np.float32)
        if len(contour_pts) > 5:
            # Convert to mm space for optimisation
            px_per_mm = dpi / 25.4
            contour_mm = contour_pts / px_per_mm
            combined = reprojection_optimise(combined, contour_mm)

    # Validate and repair
    combined, is_valid = validate_and_fix_mesh(combined, smooth_iterations)
    metadata["watertight"] = combined.is_watertight
    metadata["euler"] = combined.euler_number

    return combined, metadata
