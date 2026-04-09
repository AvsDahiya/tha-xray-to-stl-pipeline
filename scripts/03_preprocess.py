#!/usr/bin/env python3
"""Step 3: Materialize processed PNGs declared in pairing_table.csv."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from thapipeline.config import PipelineConfig
from thapipeline.data.materialize import preprocess_all


if __name__ == "__main__":
    config = PipelineConfig()
    preprocess_all(config)
