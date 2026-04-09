# THA X-Ray to STL Pipeline

End-to-end dissertation codebase for conditional image translation from single pre-operative hip X-rays to predicted post-operative implant X-rays, followed by implant segmentation, coarse 3D reconstruction, STL export, ablation studies, and 5-fold cross-validation.

## What This Repository Contains

This public repository contains the reproducible pipeline code, tests, and a curated snapshot of the final dissertation figures and results.

It does not include:
- raw datasets
- processed datasets
- annotation masks
- trained model weights
- local virtual environments or caches

Those artefacts were excluded because of dataset licensing, storage size, and machine-specific paths.

## Research Goal

The project investigates whether a single pre-operative hip-region X-ray can be translated into a plausible post-operative implant X-ray and then used as the input to a downstream implant reconstruction pipeline that produces STL outputs.

The completed workflow in this codebase is:

1. Curate pre-operative and post-operative source datasets
2. Build synthetic paired training data between source-domain and target-domain hip X-rays
3. Preprocess all images to a standard `512x512` grayscale format
4. Train Pix2Pix models with baseline and SSIM-augmented objectives
5. Select the final GAN model by validation SSIM
6. Evaluate the selected model on a held-out test split
7. Train implant segmenters on HipXNet masks
8. Run segmentation and coarse 3D reconstruction
9. Export STL meshes
10. Run ablation studies and 5-fold cross-validation

## Repository Layout

| Path | Purpose |
| --- | --- |
| `thapipeline/` | Core package containing data, models, training, inference, evaluation, and orchestration logic |
| `scripts/` | CLI entrypoints for each pipeline stage |
| `tests/` | Regression tests covering training, evaluation, reconstruction, CV, and script imports |
| `docs/results/` | Curated dissertation-ready result tables and figures for public sharing |
| `requirements.txt` | Python dependency list |

## Important Files

### Core configuration

| File | Role |
| --- | --- |
| `thapipeline/config.py` | Central path and hyperparameter configuration |
| `thapipeline/utils/io.py` | Image loading, checkpoint saving, and filesystem helpers |
| `thapipeline/utils/experiment_log.py` | Experiment manifest logging |

### Data pipeline

| File | Role |
| --- | --- |
| `scripts/01_curate_datasets.py` | Builds the dataset catalogue from the raw source folders |
| `scripts/02_create_pairs.py` | Creates the synthetic pre-op to post-op pairing table and split manifests |
| `scripts/03_preprocess.py` | Materialises `512x512` processed images for train, val, and test |
| `thapipeline/data/curate.py` | Raw dataset discovery and catalogue creation |
| `thapipeline/data/pairing.py` | Pairing logic, QA, and split handling |
| `thapipeline/data/datasets.py` | PyTorch dataset definitions and lazy materialisation fallback |

### GAN training and inference

| File | Role |
| --- | --- |
| `scripts/04_train_pix2pix.py` | Trains baseline or SSIM Pix2Pix models |
| `scripts/06_run_inference.py` | Runs inference on a chosen split using a trained generator |
| `thapipeline/models/pix2pix_unet.py` | 8-level U-Net generator used for Pix2Pix |
| `thapipeline/models/patchgan.py` | PatchGAN discriminator |
| `thapipeline/training/train_pix2pix.py` | Training loop, early stopping, checkpointing, resume logic |
| `thapipeline/training/losses.py` | Adversarial, L1, and SSIM loss composition |
| `thapipeline/inference/gan_infer.py` | Generator loading and image synthesis |

### Segmentation and reconstruction

| File | Role |
| --- | --- |
| `scripts/05_train_segmenter.py` | Trains the implant segmentation models |
| `scripts/07_segment_and_reconstruct.py` | Runs segmentation plus STL reconstruction |
| `thapipeline/models/segmenter.py` | Classical, MLP, and U-Net segmentation logic |
| `thapipeline/models/recon_3d.py` | Cup/stem geometry estimation, mesh generation, smoothing, validation |
| `thapipeline/inference/segment_and_recon.py` | End-to-end case processing for downstream reconstruction |
| `thapipeline/utils/mesh_utils.py` | Mesh rasterisation and reprojection helpers |

### Evaluation and orchestration

| File | Role |
| --- | --- |
| `scripts/08_evaluate.py` | Computes held-out metrics for GAN-only or downstream pipeline runs |
| `scripts/09_ablation_studies.py` | Runs the loss, segmentation, and reconstruction ablations |
| `scripts/10_run_5fold_cv.py` | Runs leakage-safe 5-fold cross-validation |
| `scripts/12_run_full_experiment.py` | One-shot dissertation runner for the full single-split workflow |
| `thapipeline/eval/metrics.py` | SSIM, PSNR, FID, KID, Dice, IoU, Hausdorff, Chamfer, CI summaries |
| `thapipeline/eval/statistics.py` | Paired t-tests and summary statistics |
| `thapipeline/eval/cross_validation.py` | Cross-validation orchestration and reporting |
| `thapipeline/orchestration/full_experiment.py` | Restartable full-pipeline orchestration |

## Installation

Create a Python 3.12 environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected Dataset Layout

The code expects local raw datasets in either `data_raw/` or the original dissertation folder names:

```text
Pre-operative domain (source)/
  FracAtlas – Musculoskeletal Fracture Dataset (hip subset)/
  Human Bone Fractures Multi-modal Image Dataset (HBFMID)/
  X-ray images of the hip joints (Mendeley Data)/

Post-operative domain (target)/
  Aseptic Loose Hip Implant X-Ray Database (HipXNet)/
```

The segmenter additionally expects manually validated implant masks in:

```text
annotations/hipxnet_masks/
```

## How To Run The Pipeline

### Stepwise execution

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

### One-shot single-split experiment

```bash
python3 scripts/12_run_full_experiment.py \
  --device mps \
  --tag d1_full \
  --epochs 300 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --checkpoint-every 1
```

This orchestrator resumes incomplete stages, skips completed stages, performs model selection automatically, and runs the downstream ablations after the main training/evaluation path is complete.

## Output Structure

When the full workflow is run locally, the main generated artefacts are:

| Path | Output |
| --- | --- |
| `metadata/` | Dataset catalogue, split manifests, pairing QA, CV manifests |
| `models/pix2pix/` | GAN checkpoints, training histories, run manifests |
| `models/segmenter/` | Segmenter checkpoints |
| `outputs/experiments/` | Training history CSVs and orchestration manifests |
| `outputs/figures/` | Training curves |
| `outputs/metrics/` | Held-out metrics, ablation tables, CV summaries, statistical summaries |
| `outputs/reconstruction/` | STL outputs, segmentation masks, reprojection masks, pipeline results |

## Curated Public Results

The public result snapshot is stored in [`docs/results/`](docs/results).

Key headline results from the completed dissertation runs:

| Experiment | SSIM | PSNR | FID | Notes |
| --- | ---: | ---: | ---: | --- |
| Single-split baseline GAN | 0.5606 | 12.08 | 390.56 | Held-out image translation baseline |
| Single-split selected SSIM GAN (`lambda=20`) | 0.5983 | 12.12 | 418.03 | Best validation SSIM model |
| 5-fold CV baseline mean | 0.5690 | 12.3885 | 391.44 | Mean across 5 folds |
| 5-fold CV SSIM-20 mean | 0.6045 | 12.6728 | 418.12 | Mean across 5 folds |

Interpretation:
- the SSIM-augmented model improved structural fidelity measured by SSIM
- the downstream segmentation and reconstruction stages remained weak
- the repository therefore best supports the claim of improved 2D post-operative image synthesis, rather than reliable patient-specific 3D implant recovery

## Public Results Bundle

The `docs/results/` folder includes:
- single-split GAN comparison tables
- downstream full-pipeline comparison tables
- ablation tables
- cross-validation summaries
- paired CV significance tests
- selected training curves
- selected ablation plots

## Limitations

This repository documents a completed research pipeline, but not a clinically validated deployment artefact.

Known limitations from the final experiments:
- synthetic pairing rather than true longitudinal pre-op to post-op patient pairs
- weak downstream segmentation accuracy
- weak reconstruction Dice
- `watertight_rate = 0.0` for the final reconstructed meshes
- STL export is proof-of-concept rather than production-ready implant geometry

## Test Suite

Run the regression tests with:

```bash
python3 -m unittest discover -s tests
```

## Reproducibility Notes

- The code was developed and executed on macOS with Apple Silicon using `mps`.
- The published repository intentionally excludes local dataset copies, model weights, and large generated artefacts.
- Some metadata and output files generated during dissertation execution contained machine-specific absolute paths; these were not included in the public snapshot.

