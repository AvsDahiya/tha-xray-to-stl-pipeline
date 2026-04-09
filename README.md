# THA X-Ray to STL Pipeline

Research codebase for a dissertation pipeline that translates a single pre-operative hip X-ray into a predicted post-operative implant X-ray, then runs implant segmentation, coarse 3D reconstruction, STL export, ablation studies, and 5-fold cross-validation.

## Overview

This repository contains the cleaned public snapshot of the project code, tests, and a curated set of final result tables and figures.

The repository does **not** include:
- raw datasets
- processed datasets
- manual mask annotations
- trained model checkpoints
- large local outputs and caches

Those artefacts were excluded for licensing, storage, and reproducibility reasons.

## Research Question

The project investigates whether a single pre-operative hip-region X-ray can be used to:

1. predict a plausible post-operative implant X-ray
2. segment the implant from that generated image
3. reconstruct a coarse 3D implant mesh
4. export the result as an STL artefact

The strongest completed result is the image-translation stage. The downstream segmentation and reconstruction stages remain proof-of-concept rather than clinically robust.

## Pipeline

The implemented workflow is:

1. curate source and target X-ray datasets
2. build synthetic pre-op to post-op paired training data
3. preprocess all images to `512x512` grayscale
4. train Pix2Pix baseline and SSIM-augmented models
5. select the final GAN by validation SSIM
6. evaluate on a held-out test split
7. train implant segmentation models on HipXNet masks
8. run downstream segmentation and 3D reconstruction
9. export STL meshes
10. run ablations and 5-fold cross-validation

## Repository Structure

| Path | Purpose |
| --- | --- |
| `thapipeline/` | Core Python package for data processing, models, training, inference, evaluation, and orchestration |
| `scripts/` | Command-line entrypoints for each pipeline stage |
| `tests/` | Regression tests for the public code snapshot |
| `docs/results/` | Curated dissertation-ready tables and figures |
| `requirements.txt` | Python dependencies |

## Key Files

### Configuration and utilities

| File | Purpose |
| --- | --- |
| `thapipeline/config.py` | Central configuration for paths, image settings, model settings, and training hyperparameters |
| `thapipeline/utils/io.py` | Filesystem helpers, image loading, checkpoint saving, and resume support |
| `thapipeline/utils/experiment_log.py` | Per-run manifest logging |

### Data pipeline

| File | Purpose |
| --- | --- |
| `scripts/01_curate_datasets.py` | Builds the source catalogue from local raw datasets |
| `scripts/02_create_pairs.py` | Creates the synthetic pairing table and split files |
| `scripts/03_preprocess.py` | Materialises processed `512x512` grayscale images |
| `thapipeline/data/curate.py` | Raw dataset discovery and catalogue creation |
| `thapipeline/data/pairing.py` | Pairing logic and split generation |
| `thapipeline/data/datasets.py` | PyTorch dataset definitions |

### GAN training and inference

| File | Purpose |
| --- | --- |
| `scripts/04_train_pix2pix.py` | Trains Pix2Pix baseline or SSIM models |
| `scripts/06_run_inference.py` | Runs trained generators on a chosen split |
| `thapipeline/models/pix2pix_unet.py` | U-Net generator |
| `thapipeline/models/patchgan.py` | PatchGAN discriminator |
| `thapipeline/training/train_pix2pix.py` | Training loop, early stopping, checkpointing, and resume logic |
| `thapipeline/training/losses.py` | Adversarial, L1, and SSIM loss composition |
| `thapipeline/inference/gan_infer.py` | Generator loading and forward inference |

### Segmentation and reconstruction

| File | Purpose |
| --- | --- |
| `scripts/05_train_segmenter.py` | Trains segmentation models |
| `scripts/07_segment_and_reconstruct.py` | Runs segmentation plus STL reconstruction |
| `thapipeline/models/segmenter.py` | Classical, MLP, and U-Net segmentation logic |
| `thapipeline/models/recon_3d.py` | Cup/stem geometry extraction, mesh generation, smoothing, and validation |
| `thapipeline/inference/segment_and_recon.py` | End-to-end downstream case processing |
| `thapipeline/utils/mesh_utils.py` | Mesh reprojection and rasterisation helpers |

### Evaluation and orchestration

| File | Purpose |
| --- | --- |
| `scripts/08_evaluate.py` | Held-out evaluation for GAN-only or full downstream runs |
| `scripts/09_ablation_studies.py` | Loss, segmentation, and reconstruction ablations |
| `scripts/10_run_5fold_cv.py` | Leakage-safe 5-fold cross-validation |
| `scripts/12_run_full_experiment.py` | Restartable one-shot single-split dissertation run |
| `thapipeline/eval/metrics.py` | SSIM, PSNR, FID, KID, Dice, IoU, Hausdorff, and Chamfer metrics |
| `thapipeline/eval/statistics.py` | Paired t-tests and summary statistics |
| `thapipeline/eval/cross_validation.py` | CV orchestration and reporting |
| `thapipeline/orchestration/full_experiment.py` | Full-pipeline orchestration |

## Installation

Use Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected Local Dataset Layout

The original dissertation workspace expected local raw datasets in either `data_raw/` or the named source folders:

```text
Pre-operative domain (source)/
  FracAtlas – Musculoskeletal Fracture Dataset (hip subset)/
  Human Bone Fractures Multi-modal Image Dataset (HBFMID)/
  X-ray images of the hip joints (Mendeley Data)/

Post-operative domain (target)/
  Aseptic Loose Hip Implant X-Ray Database (HipXNet)/
```

For the segmentation stage, manually validated implant masks were stored in:

```text
annotations/hipxnet_masks/
```

## Running The Pipeline

### Stage-by-stage

```bash
python3 scripts/01_curate_datasets.py
python3 scripts/02_create_pairs.py
python3 scripts/03_preprocess.py
python3 scripts/04_train_pix2pix.py --mode baseline --device mps --tag d1
python3 scripts/04_train_pix2pix.py --mode ssim --device mps --lambda-ssim 20 --tag d1
python3 scripts/05_train_segmenter.py --device mps
python3 scripts/08_evaluate.py --checkpoint models/pix2pix/d1_ssim_l20/best_model.pt --device mps --tag d1_eval
python3 scripts/09_ablation_studies.py
python3 scripts/10_run_5fold_cv.py --mode baseline --device mps --tag cv5_baseline
python3 scripts/10_run_5fold_cv.py --mode ssim --device mps --lambda-ssim 20 --tag cv5_ssim_l20
```

### One-shot single-split run

```bash
python3 scripts/12_run_full_experiment.py \
  --device mps \
  --tag d1_full \
  --epochs 300 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --checkpoint-every 1
```

## Public Results Snapshot

The curated publication-safe result bundle is in [docs/results](docs/results).

Included there:
- selected training curves
- single-split evaluation tables
- downstream full-pipeline comparison tables
- ablation tables
- cross-validation fold summaries
- paired CV significance testing

## Headline Results

| Experiment | SSIM | PSNR | FID | Notes |
| --- | ---: | ---: | ---: | --- |
| Single-split baseline GAN | 0.5606 | 12.08 | 390.56 | Held-out image translation baseline |
| Single-split selected SSIM GAN (`lambda=20`) | 0.5983 | 12.12 | 418.03 | Best validation SSIM model |
| 5-fold CV baseline mean | 0.5690 | 12.3885 | 391.44 | Mean across 5 folds |
| 5-fold CV SSIM-20 mean | 0.6045 | 12.6728 | 418.12 | Mean across 5 folds |

Interpretation:
- the SSIM-augmented model improved structural fidelity measured by SSIM
- the strongest and most defensible contribution is the 2D image translation result
- segmentation and reconstruction remained weak and are best treated as proof-of-concept

## Limitations

The published code documents a completed research pipeline, but not a clinically validated production system.

Main limitations from the final experiments:
- synthetic pairing rather than true longitudinal patient-specific pre-op to post-op pairs
- weak downstream segmentation accuracy
- weak reconstruction Dice
- `watertight_rate = 0.0` in final downstream runs
- STL export is exploratory rather than manufacturing-ready implant geometry

## Tests

Run the regression suite with:

```bash
python3 -m unittest discover -s tests
```

## Citation

If you use this repository, please cite it using the metadata in [CITATION.cff](CITATION.cff).

## License

This repository is released under the [MIT License](LICENSE).
