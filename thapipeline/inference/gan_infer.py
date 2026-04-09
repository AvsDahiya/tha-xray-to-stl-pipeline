"""GAN inference: generate synthetic post-operative radiographs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from thapipeline.config import PipelineConfig, get_device
from thapipeline.data.transforms import PreprocessPipeline, normalize_preprocessed, tensor_to_image
from thapipeline.models.pix2pix_unet import UNetGenerator
from thapipeline.utils.io import load_image, save_image, load_checkpoint


def load_generator(
    checkpoint_path: Path,
    config: PipelineConfig,
    device: Optional[str] = None,
) -> UNetGenerator:
    """Load trained generator from checkpoint."""
    dev = device or get_device()
    generator = UNetGenerator(
        in_channels=config.generator.in_channels,
        out_channels=config.generator.out_channels,
        base=config.generator.base_filters,
    ).to(dev)

    ckpt = load_checkpoint(checkpoint_path, device=dev)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    print(f"Loaded generator from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return generator


def infer_single(
    generator: UNetGenerator,
    image_path: Path,
    config: PipelineConfig,
    device: str,
    preprocessed: bool = False,
) -> dict:
    """Run GAN inference on a single pre-operative image.

    Returns:
        Dict with 'generated' (uint8), 'enhanced' (uint8), 'normalized' arrays.
    """
    if preprocessed:
        enhanced = load_image(image_path)
        normalized = normalize_preprocessed(enhanced)
        original = enhanced.copy()
    else:
        preprocess = PreprocessPipeline(
            target_size=config.image.target_size,
            crop_ratio=config.image.center_crop_ratio,
            clahe_clip=config.image.clahe_clip_limit,
            clahe_grid=config.image.clahe_tile_grid,
        )
        original = load_image(image_path)
        processed = preprocess(original)
        enhanced = processed["enhanced"]
        normalized = processed["normalized"]

    input_tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = generator(input_tensor)

    generated = tensor_to_image(output_tensor)

    return {
        "generated": generated,
        "enhanced": enhanced,
        "normalized": normalized,
        "original": original,
    }


def batch_inference(
    generator: UNetGenerator,
    image_paths: List[Path],
    output_dir: Path,
    config: PipelineConfig,
    device: str,
    preprocessed: bool = False,
) -> List[dict]:
    """Run GAN inference on a batch of images.

    Saves generated images to output_dir and returns results list.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for path in tqdm(image_paths, desc="GAN Inference"):
        try:
            result = infer_single(generator, path, config, device, preprocessed=preprocessed)
            out_path = output_dir / f"{path.stem}_generated.png"
            save_image(result["generated"], out_path)
            result["output_path"] = str(out_path)
            result["input_path"] = str(path)
            results.append(result)
        except Exception as e:
            print(f"Failed on {path.name}: {e}")
            results.append({"input_path": str(path), "error": str(e)})

    print(f"Generated {len([r for r in results if 'error' not in r])}/{len(image_paths)} images")
    return results
