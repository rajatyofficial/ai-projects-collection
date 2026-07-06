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
from utils import progress_bar


def calculate_rank(block_size, keep_percentage):
    """Convert keep percentage to target SVD rank k."""

    k = round(block_size * keep_percentage / 100)
    k = max(1, min(k, block_size))
    return k


def compress_channel(channel_matrix, block_size, k, use_numpy=False):
    """Compress a single color channel using block-based SVD.

    Processes blocks in-place to minimize memory usage — avoids
    collecting all blocks into intermediate lists.
    """

    H, W = channel_matrix.shape
    H_padded = H + (block_size - H % block_size) % block_size
    W_padded = W + (block_size - W % block_size) % block_size

    padded = np.zeros((H_padded, W_padded), dtype=channel_matrix.dtype)
    padded[:H, :W] = channel_matrix

    result = np.zeros_like(padded)
    blocks_per_row = W_padded // block_size
    total_blocks = (H_padded // block_size) * blocks_per_row
    block_idx = 0

    for row in range(0, H_padded, block_size):
        for col in range(0, W_padded, block_size):
            block = padded[row:row+block_size, col:col+block_size]
            U, sigma, Vt = custom_svd(block, k, use_numpy=use_numpy)
            result[row:row+block_size, col:col+block_size] = U @ np.diag(sigma) @ Vt
            block_idx += 1
            if block_idx % 10 == 0 or block_idx == total_blocks:
                progress_bar(block_idx, total_blocks, prefix="Compressing blocks")

    del padded
    return result[:H, :W]


def compress_image(image_path, keep=20, block_size=32, output_path="output/compressed.jpg", quality=65, use_numpy=True): # change this to use custom function
    """Compress a full RGB image using block-based SVD."""

    image = load_image(image_path)
    red, green, blue = split_channels(image)
    k = calculate_rank(block_size, keep)
    print(f"Compressing with rank k={k} (keeping {keep}% of data)")
    print("Compressing Red channel...")
    red_compressed = compress_channel(red, block_size, k, use_numpy=use_numpy)
    print("Compressing Green channel...")
    green_compressed = compress_channel(green, block_size, k, use_numpy=use_numpy)
    print("Compressing Blue channel...")
    blue_compressed = compress_channel(blue, block_size, k, use_numpy=use_numpy)
    del image, red, green, blue
    compressed = merge_channels(red_compressed, green_compressed, blue_compressed)
    del red_compressed, green_compressed, blue_compressed
    save_image(compressed, output_path, quality=quality)
    print(f"Saved compressed image to: {output_path}")
    return compressed
