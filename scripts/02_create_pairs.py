#!/usr/bin/env python3
"""Step 2: Create domain-matching synthetic pairs.

Reads catalogue.csv, computes feature vectors, matches pre-op to post-op images,
enforces reuse cap, and splits into train/val/test.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from thapipeline.config import PipelineConfig
from thapipeline.data.pairing import run_pairing_pipeline

if __name__ == "__main__":
    config = PipelineConfig()
    pairs_df = run_pairing_pipeline(config)
    print(f"\nDone! {len(pairs_df)} pairs created.")
