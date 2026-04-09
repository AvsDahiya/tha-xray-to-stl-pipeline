#!/usr/bin/env python3
"""Step 7: Segment implants and reconstruct 3D meshes."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from thapipeline.config import PipelineConfig, get_device
from thapipeline.models.segmenter import ImplantSegmenter
from thapipeline.inference.gan_infer import load_generator, infer_single
from thapipeline.inference.segment_and_recon import process_single_case
from thapipeline.utils.io import load_image, get_image_paths

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional output tag. Defaults to the checkpoint directory name.")
    parser.add_argument("--pairs-csv", type=str, default=None,
                        help="Optional alternate pairing_table.csv path.")
    parser.add_argument(
        "--segmentation-mode",
        choices=["combined", "classical", "mlp", "unet"],
        default="combined",
    )
    parser.add_argument(
        "--disable-reprojection-opt",
        action="store_true",
        help="Disable silhouette reprojection optimisation for ablation runs.",
    )
    args = parser.parse_args()

    config = PipelineConfig()
    device = args.device or get_device()
    pairs_csv = Path(args.pairs_csv) if args.pairs_csv else config.paths.pairing_table

    # Init segmenter
    seg_mlp = config.paths.segmenter_dir / "pixel_mlp.pt"
    seg_unet = config.paths.segmenter_dir / "unet_fallback.pt"
    segmenter = ImplantSegmenter(
        device=device,
        mlp_checkpoint=seg_mlp if seg_mlp.exists() else None,
        unet_checkpoint=seg_unet if seg_unet.exists() else None,
    )

    # Load generator
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        ckpt = config.paths.pix2pix_dir / "ssim_l10" / "best_model.pt"
        if not ckpt.exists():
            ckpt = config.paths.pix2pix_dir / "baseline" / "best_model.pt"
    generator = load_generator(ckpt, config, device)

    # Process test images
    pairs = pd.read_csv(pairs_csv)
    test_pairs = pairs[pairs["split"] == args.split]

    output_tag = args.tag or ckpt.parent.name
    output_dir = config.paths.outputs_dir / "reconstruction" / output_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (_, row) in enumerate(test_pairs.iterrows()):
        case_id = Path(row["pre_path"]).stem
        print(f"[{i+1}/{len(test_pairs)}] {case_id}")

        result = infer_single(
            generator,
            Path(row["pre_processed_path"]),
            config,
            device,
            preprocessed=True,
        )

        case_result = process_single_case(
            case_id=case_id,
            generated=result["generated"],
            enhanced=result["enhanced"],
            target=None,
            segmenter=segmenter,
            config=config,
            output_dir=output_dir,
            segmentation_mode=args.segmentation_mode,
            optimize_reprojection=not args.disable_reprojection_opt,
        )
        results.append(case_result)

    success = sum(1 for r in results if r["success"])
    watertight = sum(1 for r in results if r.get("watertight", False))
    print(f"\nDone! {success}/{len(results)} meshes, {watertight} watertight")
