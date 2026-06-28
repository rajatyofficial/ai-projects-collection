# High-Resolution SVD Image Compressor

A from-scratch SVD-based image compression tool built with pure Python and NumPy.

## Overview

This project compresses images of any size (up to 4K/6K) using a custom Singular Value Decomposition algorithm. Instead of applying SVD to the entire image (computationally impossible in pure Python), it uses **block-based processing** — splitting the image into tiny 32×32 grids, compressing each independently, and stitching them back together.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run via CLI
python cli.py --image samples/sample.jpg --keep 20 --block_size 32
```

## Project Structure

- `src/` — Core modules (SVD engine, image pipeline, compressor, utils)
- `Notebooks/Notebook.ipynb` — Interactive learning lab to test each function
- `cli.py` — Command-line interface
- `samples/` — Test images
- `output/` — Compressed output (auto-generated)
- `docs/` — Project plan and math references

## Learning Approach

This project follows a **notebook-first** workflow. Each function is built in `src/`, then tested and visualized in `Notebook.ipynb` to deeply understand the math before moving on.

See `docs/Project Plan.md` for the full build roadmap.
