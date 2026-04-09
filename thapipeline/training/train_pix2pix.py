"""Pix2Pix cGAN training loop (D1 §3.7).

Alternating G/D training with:
  - Two-timescale update: lr_G=2e-4, lr_D=1e-4
  - Adam optimiser: β₁=0.5, β₂=0.999
  - Step LR decay ×0.5 every 50 epochs after epoch 100
  - Early stopping on validation SSIM (patience=10)
  - TensorBoard logging
  - Mixed precision (AMP) support
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from thapipeline.config import PipelineConfig, get_device
from thapipeline.data.datasets import RadiographPairDataset
from thapipeline.models.pix2pix_unet import UNetGenerator
from thapipeline.models.patchgan import PatchGANDiscriminator
from thapipeline.training.losses import CompositeLoss, BaselineLoss
from thapipeline.eval.metrics import compute_ssim
from thapipeline.utils.io import load_checkpoint, prune_epoch_checkpoints, save_checkpoint
from thapipeline.utils.experiment_log import (
    append_jsonl,
    collect_dataset_summary,
    config_snapshot,
    environment_snapshot,
    seed_everything,
    utc_timestamp,
    write_history_csv,
    write_json,
)


class Pix2PixTrainer:
    """Full Pix2Pix cGAN training manager.

    Handles:
      - Model creation and initialisation
      - Training/validation loop
      - Loss computation
      - LR scheduling
      - Early stopping
      - Checkpointing
      - TensorBoard logging
      - Mixed precision training
    """

    def __init__(
        self,
        config: PipelineConfig,
        use_ssim: bool = True,
        device: Optional[str] = None,
        experiment_name: Optional[str] = None,
        notes: str = "",
        logic_change_note: str = "",
        pairs_csv: Optional[Path] = None,
    ):
        self.config = config
        self.device = device or get_device()
        self.use_ssim = use_ssim
        self.notes = notes.strip()
        self.logic_change_note = logic_change_note.strip()
        self.pairs_csv = pairs_csv or config.paths.pairing_table
        default_name = "ssim_l10" if use_ssim else "baseline"
        self.model_name = experiment_name or default_name
        self.run_dir = self.config.paths.pix2pix_dir / self.model_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = self.config.paths.experiments_dir / self.model_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(config.seed)

        # ── Models ──────────────────────────────────────────────────────
        self.generator = UNetGenerator(
            in_channels=config.generator.in_channels,
            out_channels=config.generator.out_channels,
            base=config.generator.base_filters,
        ).to(self.device)

        self.discriminator = PatchGANDiscriminator(
            in_channels=config.discriminator.in_channels,
            base_filters=config.discriminator.base_filters,
            use_spectral_norm=config.discriminator.use_spectral_norm,
        ).to(self.device)

        # ── Loss ────────────────────────────────────────────────────────
        if use_ssim:
            self.criterion = CompositeLoss(
                lambda_L1=config.training.lambda_L1,
                lambda_SSIM=config.training.lambda_SSIM,
                label_smoothing=config.training.label_smoothing,
            ).to(self.device)
        else:
            self.criterion = BaselineLoss(
                lambda_L1=config.training.lambda_L1,
                label_smoothing=config.training.label_smoothing,
            ).to(self.device)

        # ── Optimisers ──────────────────────────────────────────────────
        self.opt_G = Adam(
            self.generator.parameters(),
            lr=config.training.lr_G,
            betas=(config.training.beta1, config.training.beta2),
        )
        self.opt_D = Adam(
            self.discriminator.parameters(),
            lr=config.training.lr_D,
            betas=(config.training.beta1, config.training.beta2),
        )

        # ── LR Schedulers ──────────────────────────────────────────────
        self.sched_G = StepLR(
            self.opt_G,
            step_size=config.training.decay_step,
            gamma=config.training.decay_factor,
        )
        self.sched_D = StepLR(
            self.opt_D,
            step_size=config.training.decay_step,
            gamma=config.training.decay_factor,
        )

        # ── Mixed precision ────────────────────────────────────────────
        self.use_amp = config.training.use_amp and self.device == "cuda"
        self.scaler_G = GradScaler("cuda", enabled=self.use_amp)
        self.scaler_D = GradScaler("cuda", enabled=self.use_amp)

        # ── Training state ──────────────────────────────────────────────
        self.epoch = 0
        self.best_ssim = 0.0
        self.patience_counter = 0
        self.history: Dict[str, list] = {
            "G_loss": [], "D_loss": [], "G_l1": [], "G_ssim": [],
            "G_adv": [], "D_real": [], "D_fake": [],
            "val_ssim": [], "val_psnr": [], "lr_G": [], "lr_D": [],
        }

        # ── TensorBoard ────────────────────────────────────────────────
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(config.paths.logs_dir / self.model_name))
        except ImportError:
            print("TensorBoard not available, logging to console only.")
        self.manifest = self._build_manifest(status="initialized")

    def _build_manifest(self, status: str) -> Dict[str, object]:
        return {
            "experiment_name": self.model_name,
            "status": status,
            "started_at": utc_timestamp(),
            "notes": self.notes,
            "logic_change_note": self.logic_change_note,
            "use_ssim": self.use_ssim,
            "environment": environment_snapshot(self.device),
            "dataset_summary": collect_dataset_summary(self.pairs_csv),
            "pairs_csv": str(self.pairs_csv),
            "config": config_snapshot(self.config),
        }

    def _write_manifest(self) -> None:
        write_json(self.run_dir / "run_manifest.json", self.manifest)
        write_json(self.experiment_dir / "run_manifest.json", self.manifest)

    def _write_history(self) -> None:
        payload = {
            "experiment_name": self.model_name,
            "best_val_ssim": self.best_ssim,
            "updated_at": utc_timestamp(),
            "history": self.history,
        }
        write_json(self.run_dir / "history.json", payload)
        write_json(self.experiment_dir / "history.json", payload)
        write_history_csv(self.experiment_dir / "history.csv", self.history)

    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        train_dataset = RadiographPairDataset(
            pairs_csv=self.pairs_csv,
            split="train",
            config=self.config,
            augment=True,
        )
        val_dataset = RadiographPairDataset(
            pairs_csv=self.pairs_csv,
            split="val",
            config=self.config,
            augment=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=(self.device == "cuda"),
            drop_last=True,
            persistent_workers=self.config.training.num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=(self.device == "cuda"),
            persistent_workers=self.config.training.num_workers > 0,
        )

        return train_loader, val_loader

    def _autocast_context(self):
        """Return the correct autocast context for the current device."""
        if self.use_amp:
            return autocast(device_type="cuda", enabled=True)
        return nullcontext()

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch."""
        self.generator.train()
        self.discriminator.train()

        epoch_metrics = {k: 0.0 for k in [
            "G_loss", "D_loss", "G_l1", "G_ssim", "G_adv", "D_real", "D_fake",
        ]}
        n_batches = 0
        grad_accum_steps = max(1, self.config.training.grad_accum_steps)
        self.opt_D.zero_grad(set_to_none=True)
        self.opt_G.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader, start=1):
            pre = batch["pre"].to(self.device)
            post = batch["post"].to(self.device)

            # ── Train Discriminator ─────────────────────────────────────
            with self._autocast_context():
                # Real pair
                real_pred = self.discriminator(pre, post)

                # Fake pair
                with torch.no_grad():
                    fake = self.generator(pre)
                fake_pred = self.discriminator(pre, fake.detach())

                d_losses = self.criterion.discriminator_loss(real_pred, fake_pred)

            self.scaler_D.scale(d_losses["total"] / grad_accum_steps).backward()
            if batch_idx % grad_accum_steps == 0 or batch_idx == len(train_loader):
                if self.config.training.grad_clip_norm > 0:
                    self.scaler_D.unscale_(self.opt_D)
                    nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(),
                        self.config.training.grad_clip_norm,
                    )
                self.scaler_D.step(self.opt_D)
                self.scaler_D.update()
                self.opt_D.zero_grad(set_to_none=True)

            # ── Train Generator ─────────────────────────────────────────
            with self._autocast_context():
                fake = self.generator(pre)
                fake_pred = self.discriminator(pre, fake)
                g_losses = self.criterion.generator_loss(fake_pred, fake, post)

            self.scaler_G.scale(g_losses["total"] / grad_accum_steps).backward()
            if batch_idx % grad_accum_steps == 0 or batch_idx == len(train_loader):
                if self.config.training.grad_clip_norm > 0:
                    self.scaler_G.unscale_(self.opt_G)
                    nn.utils.clip_grad_norm_(
                        self.generator.parameters(),
                        self.config.training.grad_clip_norm,
                    )
                self.scaler_G.step(self.opt_G)
                self.scaler_G.update()
                self.opt_G.zero_grad(set_to_none=True)

            # Accumulate metrics
            epoch_metrics["G_loss"] += g_losses["total"].item()
            epoch_metrics["D_loss"] += d_losses["total"].item()
            epoch_metrics["G_l1"] += g_losses["l1"].item()
            epoch_metrics["G_ssim"] += g_losses["ssim"].item()
            epoch_metrics["G_adv"] += g_losses["adversarial"].item()
            epoch_metrics["D_real"] += d_losses["real"].item()
            epoch_metrics["D_fake"] += d_losses["fake"].item()
            n_batches += 1

        # Average
        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)

        return epoch_metrics

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation and compute SSIM/PSNR."""
        self.generator.eval()

        ssim_vals = []
        psnr_vals = []

        for batch in val_loader:
            pre = batch["pre"].to(self.device)
            post = batch["post"].to(self.device)

            fake = self.generator(pre)

            # Compute SSIM (per-image in batch)
            for i in range(fake.shape[0]):
                gen_np = ((fake[i, 0].cpu().numpy() + 1) / 2 * 255).clip(0, 255)
                tgt_np = ((post[i, 0].cpu().numpy() + 1) / 2 * 255).clip(0, 255)

                ssim_vals.append(compute_ssim(gen_np, tgt_np, data_range=255.0))

                # PSNR
                mse = np.mean((gen_np - tgt_np) ** 2)
                if mse > 0:
                    psnr_val = 10 * np.log10(255.0 ** 2 / mse)
                else:
                    psnr_val = float("inf")
                psnr_vals.append(psnr_val)

        return {
            "val_ssim": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
            "val_psnr": float(np.mean(psnr_vals)) if psnr_vals else 0.0,
        }

    def _save_samples(self, val_loader: DataLoader, epoch: int) -> None:
        """Save sample images for visual inspection."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.generator.eval()
        batch = next(iter(val_loader))
        pre = batch["pre"].to(self.device)
        post = batch["post"].to(self.device)

        with torch.no_grad():
            fake = self.generator(pre)

        n_show = min(4, pre.shape[0])
        fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
        if n_show == 1:
            axes = axes[np.newaxis, :]

        for i in range(n_show):
            pre_img = (pre[i, 0].cpu().numpy() + 1) / 2
            post_img = (post[i, 0].cpu().numpy() + 1) / 2
            fake_img = (fake[i, 0].cpu().numpy() + 1) / 2

            axes[i, 0].imshow(pre_img, cmap="gray")
            axes[i, 0].set_title("Pre-op (Input)")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(fake_img, cmap="gray")
            axes[i, 1].set_title("Generated")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(post_img, cmap="gray")
            axes[i, 2].set_title("Post-op (Target)")
            axes[i, 2].axis("off")

        fig.suptitle(f"Epoch {epoch}")
        fig.tight_layout()

        sample_dir = self.config.paths.samples_dir / self.model_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(sample_dir / f"epoch_{epoch:04d}.png", dpi=150)
        plt.close(fig)

    def train(self, resume_path: Optional[Path] = None) -> Dict[str, list]:
        """Run full training loop.

        Args:
            resume_path: Optional path to checkpoint to resume from.

        Returns:
            Training history dictionary.
        """
        print(f"\n{'='*60}")
        print(f"PIX2PIX TRAINING ({'with SSIM' if self.use_ssim else 'baseline'})")
        print(f"Device: {self.device}")
        print(f"Experiment: {self.model_name}")
        print(f"{'='*60}\n")

        train_loader, val_loader = self._create_dataloaders()
        self.manifest["status"] = "running"
        self.manifest["train_pairs"] = len(train_loader.dataset)
        self.manifest["val_pairs"] = len(val_loader.dataset)
        self.manifest["resume_path"] = str(resume_path) if resume_path else None
        self._write_manifest()

        # Resume if checkpoint provided
        if resume_path and resume_path.exists():
            ckpt = load_checkpoint(resume_path, device=self.device)
            self.generator.load_state_dict(ckpt["generator"])
            self.discriminator.load_state_dict(ckpt["discriminator"])
            self.opt_G.load_state_dict(ckpt["opt_G"])
            self.opt_D.load_state_dict(ckpt["opt_D"])
            self.epoch = ckpt.get("epoch", 0)
            self.best_ssim = ckpt.get("best_ssim", 0.0)
            if "history" in ckpt:
                self.history = ckpt["history"]
            print(f"Resumed from epoch {self.epoch}, best SSIM: {self.best_ssim:.4f}")

        for epoch in range(self.epoch + 1, self.config.training.epochs + 1):
            self.epoch = epoch
            start_time = time.time()

            # ── Train ───────────────────────────────────────────────────
            train_metrics = self._train_epoch(train_loader)

            # ── Validate ────────────────────────────────────────────────
            val_metrics = self._validate(val_loader)

            # ── LR scheduling (after warmup) ────────────────────────────
            if epoch > self.config.training.warmup_epochs:
                self.sched_G.step()
                self.sched_D.step()

            # ── Log metrics ─────────────────────────────────────────────
            for k, v in train_metrics.items():
                self.history[k].append(v)
            for k, v in val_metrics.items():
                self.history[k].append(v)
            self.history["lr_G"].append(self.opt_G.param_groups[0]["lr"])
            self.history["lr_D"].append(self.opt_D.param_groups[0]["lr"])

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{self.config.training.epochs} "
                f"| G: {train_metrics['G_loss']:.4f} "
                f"| D: {train_metrics['D_loss']:.4f} "
                f"| L1: {train_metrics['G_l1']:.4f} "
                f"| SSIM_loss: {train_metrics['G_ssim']:.4f} "
                f"| Val SSIM: {val_metrics['val_ssim']:.4f} "
                f"| Val PSNR: {val_metrics['val_psnr']:.1f} "
                f"| {elapsed:.1f}s"
            )

            # TensorBoard
            if self.writer:
                for k, v in {**train_metrics, **val_metrics}.items():
                    self.writer.add_scalar(k, v, epoch)

            # ── Checkpointing ───────────────────────────────────────────
            is_best = val_metrics["val_ssim"] > self.best_ssim
            if is_best:
                self.best_ssim = val_metrics["val_ssim"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if epoch % self.config.training.checkpoint_every == 0 or is_best:
                ckpt_path = self.run_dir / f"epoch_{epoch:04d}.pt"
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "generator": self.generator.state_dict(),
                        "discriminator": self.discriminator.state_dict(),
                        "opt_G": self.opt_G.state_dict(),
                        "opt_D": self.opt_D.state_dict(),
                        "best_ssim": self.best_ssim,
                        "history": self.history,
                        "experiment_name": self.model_name,
                        "notes": self.notes,
                        "logic_change_note": self.logic_change_note,
                        "config": asdict(self.config.training),
                    },
                    ckpt_path,
                    is_best=is_best,
                )

            # ── Sample images ───────────────────────────────────────────
            if epoch % self.config.training.sample_every == 0:
                self._save_samples(val_loader, epoch)

            if epoch % self.config.training.checkpoint_every == 0 or is_best:
                prune_epoch_checkpoints(
                    self.run_dir,
                    keep_last=self.config.training.keep_last_checkpoints,
                    keep_paths=[self.run_dir / "best_model.pt"],
                )

            # ── Early stopping ──────────────────────────────────────────
            if self.patience_counter >= self.config.training.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {self.config.training.patience} epochs)")
                break

        print(f"\nTraining complete. Best validation SSIM: {self.best_ssim:.4f}")
        self.manifest["status"] = "completed"
        self.manifest["completed_at"] = utc_timestamp()
        self.manifest["epochs_completed"] = self.epoch
        self.manifest["best_val_ssim"] = self.best_ssim
        self.manifest["best_val_psnr"] = max(self.history["val_psnr"], default=0.0)
        self._write_manifest()
        self._write_history()
        append_jsonl(
            self.config.paths.experiment_registry,
            {
                "timestamp": utc_timestamp(),
                "experiment_name": self.model_name,
                "device": self.device,
                "use_ssim": self.use_ssim,
                "notes": self.notes,
                "logic_change_note": self.logic_change_note,
                "best_val_ssim": self.best_ssim,
                "best_val_psnr": max(self.history["val_psnr"], default=0.0),
                "epochs_completed": self.epoch,
                "dataset_summary": collect_dataset_summary(self.pairs_csv),
                "pairs_csv": str(self.pairs_csv),
                "training_config": asdict(self.config.training),
            },
        )

        if self.writer:
            self.writer.close()

        return self.history
