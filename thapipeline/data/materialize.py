"""Materialize processed radiographs declared by pairing metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from thapipeline.config import PipelineConfig
from thapipeline.data.transforms import PreprocessPipeline
from thapipeline.utils.io import load_image, save_image


def preprocess_all(config: PipelineConfig) -> dict:
    """Materialize processed PNGs declared in pairing_table.csv."""
    pairs = pd.read_csv(config.paths.pairing_table)
    pipeline = PreprocessPipeline(
        target_size=config.image.target_size,
        crop_ratio=config.image.center_crop_ratio,
        clahe_clip=config.image.clahe_clip_limit,
        clahe_grid=config.image.clahe_tile_grid,
    )

    unique_mappings = {}
    for _, row in pairs.iterrows():
        unique_mappings[str(row["pre_processed_path"])] = str(row["pre_path"])
        unique_mappings[str(row["post_processed_path"])] = str(row["post_path"])

    print(f"Materializing {len(unique_mappings)} processed images...")
    processed = 0
    skipped = 0
    errors = 0

    for dst_str, src_str in unique_mappings.items():
        src = Path(src_str)
        dst = Path(dst_str)
        if dst.exists():
            skipped += 1
            continue
        try:
            img = load_image(src)
            result = pipeline(img)
            save_image(result["enhanced"], dst)
            processed += 1
        except Exception as exc:
            print(f"  Error processing {src.name}: {exc}")
            errors += 1

    summary = {
        "requested": len(unique_mappings),
        "processed": processed,
        "skipped_existing": skipped,
        "errors": errors,
    }
    print(
        f"\nDone! Materialized {processed} images, "
        f"skipped {skipped} existing, {errors} errors"
    )
    return summary
