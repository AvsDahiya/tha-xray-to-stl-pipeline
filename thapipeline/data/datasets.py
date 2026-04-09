"""PyTorch dataset classes for processed THA pipeline inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from thapipeline.config import PipelineConfig
from thapipeline.data.transforms import PairedRandomAugment, PreprocessPipeline, normalize_preprocessed
from thapipeline.utils.io import load_image, save_image


class RadiographPairDataset(Dataset):
    """Dataset of processed pre/post radiograph pairs for Pix2Pix training."""

    def __init__(
        self,
        pairs_csv: Path,
        split: str = "train",
        config: Optional[PipelineConfig] = None,
        augment: bool = True,
    ):
        self.config = config or PipelineConfig()
        self.split = split
        self.do_augment = augment and split == "train"
        df = pd.read_csv(pairs_csv)
        self.pairs = df[df["split"] == split].reset_index(drop=True)
        self.preprocess = PreprocessPipeline(
            target_size=self.config.image.target_size,
            crop_ratio=self.config.image.center_crop_ratio,
            clahe_clip=self.config.image.clahe_clip_limit,
            clahe_grid=self.config.image.clahe_tile_grid,
        )

        if self.do_augment:
            aug_cfg = self.config.augment
            self.augmenter = PairedRandomAugment(
                flip_prob=aug_cfg.flip_prob,
                rotation_range=aug_cfg.rotation_range,
                brightness_range=aug_cfg.brightness_range,
                contrast_range=aug_cfg.contrast_range,
                noise_std=aug_cfg.noise_std,
                augment_prob=aug_cfg.augment_prob,
            )
        else:
            self.augmenter = None

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_or_materialize_processed(self, processed_path: Path, raw_path: Path) -> np.ndarray:
        if processed_path.exists():
            return load_image(processed_path)

        raw_image = load_image(raw_path)
        enhanced = self.preprocess(raw_image)["enhanced"]
        try:
            save_image(enhanced, processed_path)
        except Exception:
            pass
        return enhanced

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.pairs.iloc[idx]
        pre_path = Path(row["pre_processed_path"])
        post_path = Path(row["post_processed_path"])
        raw_pre_path = Path(row["pre_path"])
        raw_post_path = Path(row["post_path"])

        pre_img = self._load_or_materialize_processed(pre_path, raw_pre_path)
        post_img = self._load_or_materialize_processed(post_path, raw_post_path)
        pre_norm = normalize_preprocessed(pre_img)
        post_norm = normalize_preprocessed(post_img)

        if self.augmenter is not None:
            pre_norm, post_norm = self.augmenter(pre_norm, post_norm)

        return {
            "pair_id": str(row["pair_id"]),
            "split": str(row["split"]),
            "pre": torch.from_numpy(pre_norm).float().unsqueeze(0),
            "post": torch.from_numpy(post_norm).float().unsqueeze(0),
            "pre_path": str(row["pre_path"]),
            "post_path": str(row["post_path"]),
            "pre_processed_path": str(pre_path),
            "post_processed_path": str(post_path),
            "pre_id": str(row["pre_id"]),
            "post_id": str(row["post_id"]),
        }


class SingleImageDataset(Dataset):
    """Dataset of single processed radiographs for inference."""

    def __init__(self, image_paths: list[str], preprocessed: bool = True):
        self.paths = [str(path) for path in image_paths]
        self.preprocessed = preprocessed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = Path(self.paths[idx])
        image = load_image(path)
        normalized = normalize_preprocessed(image)
        return {
            "path": str(path),
            "image": torch.from_numpy(normalized).float().unsqueeze(0),
            "enhanced": torch.from_numpy(image.astype("float32") / 255.0).float().unsqueeze(0),
        }
