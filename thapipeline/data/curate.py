"""Dataset curation and acquisition metadata generation."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from thapipeline.config import HipRegionFilterConfig, PipelineConfig
from thapipeline.utils.io import (
    get_image_paths,
    get_nifti_paths,
    load_image,
    load_nifti_label_slice,
    load_nifti_midslice,
    save_image,
    save_json,
    sanitize_id,
)


ACQUISITION_SOURCES = {
    "fracatlas": {
        "label": "FracAtlas – Musculoskeletal Fracture Dataset (hip subset)",
        "host": "Figshare / DOI",
        "url": "https://doi.org/10.6084/m9.figshare.22363012",
        "expected_count": 4083,
    },
    "hbfmid": {
        "label": "Human Bone Fractures Multi-modal Image Dataset (HBFMID)",
        "host": "Mendeley Data / Kaggle mirror",
        "url": "https://data.mendeley.com/datasets/xwfs6xbk47/1",
        "expected_count": 1539,
    },
    "mendeley_hip": {
        "label": "X-ray images of the hip joints",
        "host": "Mendeley Data",
        "url": "https://data.mendeley.com/datasets/zm6bxzhmfz/1",
        "expected_count": 140,
    },
    "hipxnet": {
        "label": "Aseptic Loose Hip Implant X-Ray Database (HipXNet)",
        "host": "Kaggle",
        "url": "https://www.kaggle.com/datasets/tawsifurrahman/aseptic-loose-hip-implant-xray-database",
        "expected_count": 206,
    },
}


CATALOGUE_COLUMNS = [
    "source_id",
    "canonical_source_id",
    "source_dataset",
    "filepath",
    "view",
    "region",
    "fracture_label",
    "postop_flag",
    "raw_split",
    "has_label",
    "label_path",
    "notes",
]


def _empty_catalogue() -> pd.DataFrame:
    return pd.DataFrame(columns=CATALOGUE_COLUMNS)


def is_mri_image(filename: str) -> bool:
    return "mri" in filename.lower()


def canonicalize_hbfmid_name(path: Path) -> str:
    stem = re.sub(r"\.rf\.[^.]+$", "", path.stem)
    stem = re.sub(r"_jpeg$", "", stem)
    return sanitize_id(stem)


def is_hip_xray(image: np.ndarray, config: HipRegionFilterConfig) -> bool:
    h, w = image.shape[:2]
    if min(h, w) < config.min_dimension:
        return False

    aspect = w / h if h else 0.0
    if aspect < config.min_aspect_ratio or aspect > config.max_aspect_ratio:
        return False

    mean_intensity = float(np.mean(image))
    if mean_intensity < config.min_mean_intensity or mean_intensity > config.max_mean_intensity:
        return False

    center_region = image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    if center_region.size == 0:
        return False

    if float(np.mean(center_region)) < mean_intensity * 0.7:
        return False

    left_half = image[:, : w // 2]
    right_half = np.fliplr(image[:, w // 2 :])
    overlap = min(left_half.shape[1], right_half.shape[1])
    if overlap == 0:
        return False

    left_flat = left_half[:, :overlap].flatten().astype(np.float32)
    right_flat = right_half[:, :overlap].flatten().astype(np.float32)
    if np.std(left_flat) <= 1 or np.std(right_flat) <= 1:
        return False

    correlation = float(np.corrcoef(left_flat, right_flat)[0, 1])
    return correlation >= 0.2


def build_data_sources_metadata(config: PipelineConfig) -> Dict[str, Dict[str, object]]:
    metadata: Dict[str, Dict[str, object]] = {}
    local_paths = {
        "fracatlas": config.paths.fracatlas_dir,
        "hbfmid": config.paths.hbfmid_dir,
        "mendeley_hip": config.paths.mendeley_dir,
        "hipxnet": config.paths.hipxnet_dir,
    }

    for key, source in ACQUISITION_SOURCES.items():
        resolved = local_paths[key]
        metadata[key] = {
            **source,
            "local_path": str(resolved),
            "resolved": resolved.exists(),
        }
    return metadata


def curate_fracatlas(config: PipelineConfig) -> pd.DataFrame:
    csv_path = config.paths.fracatlas_csv
    images_dir = config.paths.fracatlas_images
    if not csv_path.exists():
        return _empty_catalogue()

    df = pd.read_csv(csv_path)
    filtered = df[(df["hip"] == 1) & (df["frontal"] == 1)].copy()

    records: List[Dict[str, object]] = []
    for _, row in filtered.iterrows():
        image_id = str(row["image_id"])
        filepath = None
        for subdir in ("Fractured", "Non_fractured"):
            candidate = images_dir / subdir / image_id
            if candidate.exists():
                filepath = candidate
                break
        if filepath is None:
            candidate = images_dir / image_id
            if candidate.exists():
                filepath = candidate
        if filepath is None:
            continue

        records.append(
            {
                "source_id": sanitize_id(Path(image_id).stem),
                "canonical_source_id": sanitize_id(Path(image_id).stem),
                "source_dataset": "fracatlas",
                "filepath": str(filepath),
                "view": "AP",
                "region": "hip",
                "fracture_label": int(row.get("fractured", 0)),
                "postop_flag": 0,
                "raw_split": "",
                "has_label": False,
                "label_path": "",
                "notes": "",
            }
        )

    return pd.DataFrame(records, columns=CATALOGUE_COLUMNS)


def curate_hbfmid(config: PipelineConfig) -> pd.DataFrame:
    roots = config.paths.hbfmid_image_roots
    if not any(root.exists() for root in roots):
        return _empty_catalogue()

    candidates: List[Dict[str, object]] = []
    for root in roots:
        image_paths = get_image_paths(root)
        for img_path in image_paths:
            if is_mri_image(img_path.name):
                continue

            try:
                image = load_image(img_path)
            except Exception:
                continue

            if not is_hip_xray(image, config.hip_filter):
                continue

            raw_split = ""
            for token in ("train", "valid", "val", "test"):
                if token in img_path.parts:
                    raw_split = token
                    break

            canonical_id = canonicalize_hbfmid_name(img_path)
            candidates.append(
                {
                    "source_id": sanitize_id(img_path.stem),
                    "canonical_source_id": canonical_id,
                    "source_dataset": "hbfmid",
                    "filepath": str(img_path),
                    "view": "AP",
                    "region": "hip",
                    "fracture_label": 1,
                    "postop_flag": 0,
                    "raw_split": raw_split,
                    "has_label": False,
                    "label_path": "",
                    "notes": "",
                }
            )

    if not candidates:
        return _empty_catalogue()

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in candidates:
        grouped[str(record["canonical_source_id"])].append(record)

    selected: List[Dict[str, object]] = []
    split_priority = {"train": 0, "valid": 1, "val": 1, "test": 2, "": 3}
    for canonical_id, records in grouped.items():
        records = sorted(
            records,
            key=lambda item: (
                split_priority.get(str(item["raw_split"]), 9),
                str(item["filepath"]),
            ),
        )
        chosen = records[0].copy()
        if len(records) > 1:
            chosen["notes"] = f"deduped_hbfmid_variants={len(records)}"
        selected.append(chosen)

    return pd.DataFrame(selected, columns=CATALOGUE_COLUMNS)


def curate_mendeley_hip(config: PipelineConfig) -> pd.DataFrame:
    if not config.paths.mendeley_images.exists():
        return _empty_catalogue()

    records: List[Dict[str, object]] = []
    for nii_path in get_nifti_paths(config.paths.mendeley_images):
        case_id = sanitize_id(nii_path.stem.split(".")[0])
        image_slice = load_nifti_midslice(nii_path)
        image_out = config.paths.mendeley_slice_dir / f"{case_id}.png"
        save_image(image_slice, image_out)

        label_out = ""
        has_label = False
        label_name = f"label_{case_id.split('_')[-1]}.nii.gz"
        label_path = config.paths.mendeley_labels / label_name
        if not label_path.exists():
            alt = config.paths.mendeley_labels / f"{case_id.replace('image', 'label')}.nii.gz"
            label_path = alt if alt.exists() else label_path
        if label_path.exists():
            label_slice = load_nifti_label_slice(label_path)
            label_png = config.paths.mendeley_label_slice_dir / f"{case_id}.png"
            save_image(label_slice, label_png)
            label_out = str(label_png)
            has_label = True

        records.append(
            {
                "source_id": case_id,
                "canonical_source_id": case_id,
                "source_dataset": "mendeley_hip",
                "filepath": str(image_out),
                "view": "AP",
                "region": "hip",
                "fracture_label": 0,
                "postop_flag": 0,
                "raw_split": "",
                "has_label": has_label,
                "label_path": label_out,
                "notes": "anatomy_labels_only" if has_label else "",
            }
        )

    return pd.DataFrame(records, columns=CATALOGUE_COLUMNS)


def curate_hipxnet(config: PipelineConfig) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for implant_status, directory in (
        ("control", config.paths.hipxnet_control),
        ("loose", config.paths.hipxnet_loose),
    ):
        if not directory.exists():
            continue
        for img_path in get_image_paths(directory):
            case_id = sanitize_id(img_path.stem)
            records.append(
                {
                    "source_id": case_id,
                    "canonical_source_id": case_id,
                    "source_dataset": "hipxnet",
                    "filepath": str(img_path),
                    "view": "AP",
                    "region": "hip_implant",
                    "fracture_label": 0,
                    "postop_flag": 1,
                    "raw_split": "",
                    "has_label": False,
                    "label_path": "",
                    "notes": implant_status,
                }
            )
    return pd.DataFrame(records, columns=CATALOGUE_COLUMNS)


def curate_all_datasets(config: PipelineConfig) -> pd.DataFrame:
    data_sources = build_data_sources_metadata(config)
    save_json(data_sources, config.paths.data_sources_json)

    datasets = [
        curate_fracatlas(config),
        curate_hbfmid(config),
        curate_mendeley_hip(config),
        curate_hipxnet(config),
    ]
    datasets = [df for df in datasets if not df.empty]
    if not datasets:
        raise RuntimeError("No datasets were found during curation.")

    catalogue = pd.concat(datasets, ignore_index=True)
    catalogue = catalogue[CATALOGUE_COLUMNS].copy()
    catalogue.to_csv(config.paths.catalogue_csv, index=False)
    return catalogue
