#!/usr/bin/env python3
"""Step 1: Curate all datasets and produce unified catalogue.

Scans FracAtlas, HBFMID (with hip-region filtering), Mendeley Hip (NIfTI extraction),
and HipXNet to produce metadata/catalogue.csv.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from thapipeline.config import PipelineConfig
from thapipeline.data.curate import curate_all_datasets

if __name__ == "__main__":
    config = PipelineConfig()
    catalogue = curate_all_datasets(config)
    print(f"\nDone! Catalogue saved with {len(catalogue)} entries.")
