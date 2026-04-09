#!/usr/bin/env python3
"""Step 6: Run GAN inference on test set."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import pandas as pd

from thapipeline.config import PipelineConfig, get_device
from thapipeline.inference.gan_infer import load_generator, batch_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Generator checkpoint path (default: best model)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional output tag. Defaults to the checkpoint directory name.")
    parser.add_argument("--pairs-csv", type=str, default=None,
                        help="Optional alternate pairing_table.csv path.")
    args = parser.parse_args()

    config = PipelineConfig()
    device = args.device or get_device()
    pairs_csv = Path(args.pairs_csv) if args.pairs_csv else config.paths.pairing_table

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = config.paths.pix2pix_dir / "ssim_l10" / "best_model.pt"
        if not ckpt_path.exists():
            ckpt_path = config.paths.pix2pix_dir / "baseline" / "best_model.pt"

    print(f"Using checkpoint: {ckpt_path}")
    generator = load_generator(ckpt_path, config, device)

    # Get test images
    pairs = pd.read_csv(pairs_csv)
    test_pairs = pairs[pairs["split"] == args.split]
    image_paths = [Path(p) for p in test_pairs["pre_processed_path"].tolist()]

    print(f"Running inference on {len(image_paths)} {args.split} images...")
    output_tag = args.tag or ckpt_path.parent.name
    output_dir = config.paths.outputs_dir / "generated" / output_tag / args.split
    results = batch_inference(generator, image_paths, output_dir, config, device, preprocessed=True)
    print(f"Done! Results saved to {output_dir}")
