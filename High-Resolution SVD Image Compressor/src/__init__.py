# ==============================================================================
# SVD Image Compressor — src package
# ==============================================================================
#
# This package contains the core modules:
#
#   src.svd_engine      (Phase 1) — Custom SVD math from scratch
#   src.image_pipeline  (Phase 2) — Image loading, channel splitting, block processing
#   src.compressor      (Phase 3) — Compression logic tying Phases 1 & 2 together
#   src.utils           (Helpers) — Timer, progress bar, input validation
#
# HOW TO IMPORT IN THE NOTEBOOK:
#   The notebook (Notebooks/Notebook.ipynb) needs to add the project root
#   to its Python path before importing. Add this as the FIRST cell:
#
#       import sys
#       import os
#       # Add the project root (parent of Notebooks/) to Python's path
#       sys.path.insert(0, os.path.abspath(".."))
#
#   Then you can import like:
#       from src.svd_engine import power_iteration, deflation, custom_svd
#       from src.image_pipeline import load_image, split_channels, image_to_blocks
#       from src.compressor import compress_image
#
# BUILD ORDER:
#   1. svd_engine.py       → test in Notebook Section 1
#   2. image_pipeline.py   → test in Notebook Section 2
#   3. compressor.py       → test in Notebook Section 3
#   4. cli.py (root level) → test in Notebook Section 4
# ==============================================================================
