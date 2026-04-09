"""Central configuration for the THA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Tuple


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _first_existing_or_default(*candidates: Path) -> Path:
    existing = _first_existing(candidates)
    return existing if existing is not None else candidates[0]


@dataclass
class PathConfig:
    """Dataset and output directory paths."""

    project_root: Path = field(default_factory=_project_root)

    @property
    def data_raw(self) -> Path:
        return self.project_root / "data_raw"

    @property
    def data_intermediate(self) -> Path:
        return self.project_root / "data_intermediate"

    @property
    def data_processed(self) -> Path:
        return self.project_root / "data_processed"

    @property
    def metadata_dir(self) -> Path:
        return self.project_root / "metadata"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def annotations_dir(self) -> Path:
        return self.project_root / "annotations"

    @property
    def fracatlas_dir(self) -> Path:
        canonical = self.data_raw / "fracatlas"
        fallback = (
            self.project_root
            / "Pre-operative domain (source)"
            / "FracAtlas – Musculoskeletal Fracture Dataset (hip subset)"
            / "FracAtlas"
        )
        return _first_existing_or_default(canonical, fallback)

    @property
    def fracatlas_csv(self) -> Path:
        return self.fracatlas_dir / "dataset.csv"

    @property
    def fracatlas_images(self) -> Path:
        return self.fracatlas_dir / "images"

    @property
    def hbfmid_dir(self) -> Path:
        canonical = self.data_raw / "hbfmid"
        canonical_alt = canonical / "Bone Fractures Detection"
        fallback = (
            self.project_root
            / "Pre-operative domain (source)"
            / "Human Bone Fractures Multi-modal Image Dataset (HBFMID)"
            / "Bone Fractures Detection"
        )
        return _first_existing_or_default(canonical_alt, canonical, fallback)

    @property
    def hbfmid_image_roots(self) -> Tuple[Path, ...]:
        root = self.hbfmid_dir
        candidates = [
            root / "train" / "images",
            root / "valid" / "images",
            root / "val" / "images",
            root / "test" / "images",
            root / "xray",
            root / "images",
        ]
        existing = tuple(path for path in candidates if path.exists())
        return existing or (root,)

    @property
    def mendeley_dir(self) -> Path:
        canonical = self.data_raw / "hip_mendeley"
        fallback = (
            self.project_root
            / "Pre-operative domain (source)"
            / "X-ray images of the hip joints (Mendeley Data)"
        )
        return _first_existing_or_default(canonical, fallback)

    @property
    def mendeley_images(self) -> Path:
        return self.mendeley_dir / "images"

    @property
    def mendeley_labels(self) -> Path:
        return self.mendeley_dir / "labels"

    @property
    def hipxnet_dir(self) -> Path:
        canonical = self.data_raw / "hipxnet_kaggle"
        fallback = (
            self.project_root
            / "Post-operative domain (target)"
            / "Aseptic Loose Hip Implant X-Ray Database (HipXNet)"
        )
        return _first_existing_or_default(canonical, fallback)

    @property
    def hipxnet_control(self) -> Path:
        return self.hipxnet_dir / "Control"

    @property
    def hipxnet_loose(self) -> Path:
        return self.hipxnet_dir / "Loose"

    @property
    def mendeley_slice_dir(self) -> Path:
        return self.data_intermediate / "mendeley_slices" / "images"

    @property
    def mendeley_label_slice_dir(self) -> Path:
        return self.data_intermediate / "mendeley_slices" / "labels"

    @property
    def pix2pix_dir(self) -> Path:
        return self.models_dir / "pix2pix"

    @property
    def segmenter_dir(self) -> Path:
        return self.models_dir / "segmenter"

    @property
    def recon_dir(self) -> Path:
        return self.models_dir / "recon"

    @property
    def samples_dir(self) -> Path:
        return self.outputs_dir / "samples"

    @property
    def meshes_dir(self) -> Path:
        return self.outputs_dir / "meshes_stl"

    @property
    def metrics_dir(self) -> Path:
        return self.outputs_dir / "metrics"

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    @property
    def generated_dir(self) -> Path:
        return self.outputs_dir / "generated"

    @property
    def experiments_dir(self) -> Path:
        return self.outputs_dir / "experiments"

    @property
    def catalogue_csv(self) -> Path:
        return self.metadata_dir / "catalogue.csv"

    @property
    def pairing_table(self) -> Path:
        return self.metadata_dir / "pairing_table.csv"

    @property
    def split_indices(self) -> Path:
        return self.metadata_dir / "split_indices.json"

    @property
    def pairing_qa(self) -> Path:
        return self.metadata_dir / "pairing_qa.json"

    @property
    def data_sources_json(self) -> Path:
        return self.metadata_dir / "data_sources.json"

    @property
    def print_validation_template(self) -> Path:
        return self.metrics_dir / "print_validation_template.csv"

    @property
    def experiment_registry(self) -> Path:
        return self.metadata_dir / "experiment_registry.jsonl"

    @property
    def segmentation_split_json(self) -> Path:
        return self.metadata_dir / "segmentation_split.json"

    @property
    def segmentation_report_json(self) -> Path:
        return self.metrics_dir / "segmentation_training_report.json"

    @property
    def segmentation_case_metrics_csv(self) -> Path:
        return self.metrics_dir / "segmentation_case_metrics.csv"

    @property
    def statistics_summary_json(self) -> Path:
        return self.metrics_dir / "statistical_summary.json"

    @property
    def statistics_summary_csv(self) -> Path:
        return self.metrics_dir / "statistical_summary.csv"

    @property
    def implant_masks_dir(self) -> Path:
        candidates = (
            self.annotations_dir / "hipxnet_masks",
            self.data_raw / "hipxnet_masks",
            self.project_root / "labels" / "hipxnet_masks",
        )
        return _first_existing_or_default(*candidates)

    def ensure_dirs(self) -> None:
        for path in (
            self.data_raw,
            self.data_intermediate,
            self.data_processed,
            self.metadata_dir,
            self.models_dir,
            self.logs_dir,
            self.outputs_dir,
            self.annotations_dir,
            self.mendeley_slice_dir,
            self.mendeley_label_slice_dir,
            self.pix2pix_dir,
            self.segmenter_dir,
            self.recon_dir,
            self.samples_dir,
            self.meshes_dir,
            self.metrics_dir,
            self.figures_dir,
            self.generated_dir,
            self.experiments_dir,
            self.implant_masks_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ImageConfig:
    target_size: Tuple[int, int] = (512, 512)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    center_crop_ratio: float = 0.80
    norm_range: Tuple[float, float] = (-1.0, 1.0)


@dataclass
class AugmentConfig:
    flip_prob: float = 0.5
    rotation_range: float = 5.0
    brightness_range: float = 0.15
    contrast_range: float = 0.15
    noise_std: float = 0.02
    augment_prob: float = 0.5


@dataclass
class GeneratorConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_filters: int = 64
    encoder_channels: Tuple[int, ...] = (64, 128, 256, 512, 512, 512, 512, 512)
    dropout_layers: Tuple[int, ...] = (0, 1, 2)
    dropout_rate: float = 0.5


@dataclass
class DiscriminatorConfig:
    in_channels: int = 2
    base_filters: int = 64
    use_spectral_norm: bool = True
    output_map_size: Tuple[int, int] = (30, 30)


@dataclass
class TrainingConfig:
    lr_G: float = 2e-4
    lr_D: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_L1: float = 100.0
    lambda_SSIM: float = 10.0
    epochs: int = 300
    batch_size: int = 4
    num_workers: int = 2
    grad_accum_steps: int = 1
    grad_clip_norm: float = 0.0
    warmup_epochs: int = 100
    decay_step: int = 50
    decay_factor: float = 0.5
    patience: int = 10
    label_smoothing: float = 0.9
    checkpoint_every: int = 10
    keep_last_checkpoints: int = 3
    sample_every: int = 5
    use_amp: bool = True


@dataclass
class SegmentationConfig:
    mlp_features: int = 5
    mlp_hidden: int = 64
    mlp_lr: float = 1e-3
    mlp_epochs: int = 20
    mlp_batch_size: int = 4096
    mlp_samples_per_record: int = 2048
    unet_base_filters: int = 32
    unet_depth: int = 4
    unet_lr: float = 1e-4
    unet_epochs: int = 50
    unet_batch_size: int = 4
    default_threshold: float = 0.5
    fallback_confidence: float = 0.3
    min_foreground_ratio: float = 0.001
    max_foreground_ratio: float = 0.40
    train_ratio: float = 2.0 / 3.0
    val_ratio: float = 1.0 / 6.0
    test_ratio: float = 1.0 / 6.0


@dataclass
class ReconstructionConfig:
    cup_radius_min_mm: float = 25.0
    cup_radius_max_mm: float = 35.0
    cup_subdivisions: int = 4
    stem_length_min_mm: float = 90.0
    stem_length_max_mm: float = 130.0
    stem_sections: int = 32
    magnification_factor: float = 1.15
    laplacian_iterations: int = 5
    laplacian_lambda: float = 0.5
    default_dpi: float = 150.0


@dataclass
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    max_postop_reuse: int = 5


@dataclass
class HipRegionFilterConfig:
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    min_dimension: int = 100
    min_mean_intensity: float = 30.0
    max_mean_intensity: float = 200.0


@dataclass
class PipelineConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    hip_filter: HipRegionFilterConfig = field(default_factory=HipRegionFilterConfig)
    seed: int = 42

    def __post_init__(self) -> None:
        self.paths.ensure_dirs()


def get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
