#!/usr/bin/env python3
"""Step 5: Train segmentation models (Pixel MLP + U-Net fallback)."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import argparse

from thapipeline.config import PipelineConfig, get_device
from thapipeline.training.train_segmenter import (
    prepare_training_records,
    train_and_evaluate_segmenter,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = PipelineConfig()
    device = args.device or get_device()

    print(f"Using device: {device}")

    # Prepare training data
    records = prepare_training_records(config)

    if len(records) == 0:
        print("No labeled records found! Segmentation will use classical pipeline only.")
        sys.exit(0)

    report = train_and_evaluate_segmenter(records, config, device)
    print(f"Pixel MLP threshold: {report['mlp_threshold']:.3f}")
    print(f"Segmentation split saved to: {config.paths.segmentation_split_json}")
    print(f"Segmentation report saved to: {config.paths.segmentation_report_json}")
    print(f"Segmentation case metrics saved to: {config.paths.segmentation_case_metrics_csv}")

    print("\nSegmentation training complete!")
