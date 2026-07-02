"""
==============================================================================
COMPRESSOR — Phase 3: The Compression Logic
==============================================================================

This is where Phase 1 (SVD math) meets Phase 2 (image pipeline).
This file ties everything together into a working image compressor.

WHAT THIS FILE DOES:
    1. Translate user's "keep percentage" into a mathematical rank k
    2. Compress a single color channel (split → SVD each block → stitch)
    3. Compress a full RGB image (compress all 3 channels → merge → save)

HOW COMPRESSION WORKS (the big picture):
    Original block (32×32) = 32 × 32 = 1024 values stored
    
    SVD decomposes it into: U(32×k) · Σ(k) · Vᵀ(k×32)
    Compressed storage = 32k + k + 32k = 65k values
    
    If k = 8:  stored = 65 × 8 = 520 values  (≈50% compression!)
    If k = 4:  stored = 65 × 4 = 260 values  (≈75% compression!)
    If k = 1:  stored = 65 × 1 = 65 values   (≈94% compression! but very blurry)
    
    The user controls quality by setting "keep percentage" (what % of singular
    values to retain). Higher keep% = better quality but less compression.

BUILD ORDER:
    1. calculate_rank()       — simple math: keep% → rank k
    2. compress_channel()     — compress one color channel
    3. compress_image()       — compress all 3 channels and save

DEPENDENCIES (imports from your other files):
    from src.svd_engine import custom_svd
    from src.image_pipeline import (load_image, split_channels, image_to_blocks,
                                     blocks_to_image, merge_channels, save_image)

NOTEBOOK TEST (after completing this file):
    In Notebook.ipynb, Section 3:
    - Compress a single block at k=1,4,8,16,32 → show visual quality differences
    - Test calculate_rank with different keep percentages
    - Compress a full image → before/after comparison
    - Quality sweep: compress at 5%, 10%, 20%, 50%, 80% and plot PSNR
==============================================================================
"""

# pyrefly: ignore [missing-import]
import numpy as np

from src.svd_engine import custom_svd
from src.image_pipeline import (
    load_image,
    split_channels,
    image_to_blocks,
    blocks_to_image,
    merge_channels,
    save_image,
)


def calculate_rank(block_size, keep_percentage):
    """Convert keep percentage to target SVD rank k."""

    k = round(block_size * keep_percentage / 100)
    k = max(1, min(k, block_size))
    return k


def compress_channel(channel_matrix, block_size, k):
    """Compress a single color channel using block-based SVD."""

    blocks, padded_shape = image_to_blocks(channel_matrix, block_size)
    original_shape = channel_matrix.shape
    compressed_blocks = []

    for i, block in enumerate(blocks):
        U, sigma, Vt = custom_svd(block, k)
        compressed_block = U @ np.diag(sigma) @ Vt
        compressed_blocks.append(compressed_block)
        if i % 100 == 0:
            print(f"Processing block {i}/{len(blocks)}", end="\r")
    compressed_channel = blocks_to_image(compressed_blocks, padded_shape, original_shape, block_size)
    return compressed_channel


def compress_image(image_path, keep=20, block_size=32, output_path="output/compressed.jpg"):
    """Compress a full RGB image using block-based SVD."""

    image = load_image(image_path)
    red, green, blue = split_channels(image)
    k = calculate_rank(block_size, keep)
    print(f"Compressing with rank k={k} (keeping {keep}% of data)")
    print("Compressing Red channel...")
    red_compressed = compress_channel(red, block_size, k)
    print("Compressing Green channel...")
    green_compressed = compress_channel(green, block_size, k)
    print("Compressing Blue channel...")
    blue_compressed = compress_channel(blue, block_size, k)
    compressed = merge_channels(red_compressed, green_compressed, blue_compressed)
    save_image(compressed, output_path)
    print(f"Saved compressed image to: {output_path}")
    return compressed
