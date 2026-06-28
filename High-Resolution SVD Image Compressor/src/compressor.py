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

import numpy as np

# NOTE: Uncomment these imports once you've completed Phase 1 and Phase 2!
# from src.svd_engine import custom_svd
# from src.image_pipeline import (
#     load_image,
#     split_channels,
#     image_to_blocks,
#     blocks_to_image,
#     merge_channels,
#     save_image,
# )


# ==============================================================================
# STEP 3.1: RANK CALCULATION
# ==============================================================================
#
# WHAT IT DOES:
#     Converts the user's human-friendly "keep percentage" into the
#     mathematical parameter "k" (the number of singular values to keep).
#
# THE MATH:
#     k = round(block_size × (keep_percentage / 100))
#
#     Example: block_size=32, keep_percentage=25%
#     → k = round(32 × 0.25) = 8
#     → We keep the top 8 singular values out of 32
#
# EDGE CASES:
#     - k must be at least 1 (even 1% compression keeps 1 singular value)
#     - k cannot exceed block_size (100% means keep everything)
#
# PARAMETERS:
#     block_size      : int — the size of each block (e.g., 32)
#     keep_percentage : float — percentage of data to keep (1 to 100)
#
# RETURNS:
#     k : int — the target rank for SVD (between 1 and block_size)
#
# HINTS:
#     - k = max(1, round(block_size * keep_percentage / 100))
#     - k = min(k, block_size)   ← clamp to block_size
#
# EXAMPLE:
#     >>> calculate_rank(32, 25)   # → 8
#     >>> calculate_rank(32, 50)   # → 16
#     >>> calculate_rank(32, 100)  # → 32
#     >>> calculate_rank(32, 1)    # → 1  (minimum)
#
def calculate_rank(block_size, keep_percentage):
    """Convert keep percentage to target SVD rank k."""
    
    # YOUR CODE BELOW ↓↓↓
    
    # Step 1: Calculate k = round(block_size * keep_percentage / 100)
    
    # Step 2: Clamp: k must be at least 1 and at most block_size
    # HINT: k = max(1, min(k, block_size))
    
    # Step 3: Return k
    
    pass  # ← Remove this once you write your code


# ==============================================================================
# STEP 3.2: COMPRESS A SINGLE CHANNEL
# ==============================================================================
#
# WHAT IT DOES:
#     Takes one color channel (e.g., the Red channel), splits it into blocks,
#     runs custom_svd on each block, reconstructs each block with rank k,
#     and stitches them back together.
#
# THE ALGORITHM:
#     1. Split the channel into blocks (using image_to_blocks)
#     2. For each block:
#        a) Run custom_svd(block, k) → get U, sigma, Vt
#        b) Reconstruct: compressed_block = U @ np.diag(sigma) @ Vt
#        c) Store the compressed block
#     3. Stitch all compressed blocks back together (using blocks_to_image)
#     4. Return the reconstructed channel
#
# PARAMETERS:
#     channel_matrix : numpy 2D array (H, W) — one color channel (e.g., red)
#     block_size     : int — size of each block (default 32)
#     k              : int — target rank for SVD compression
#
# RETURNS:
#     compressed_channel : numpy 2D array (H, W) — the compressed channel
#
# HINTS:
#     - np.diag(sigma) creates a diagonal matrix from the 1D sigma array
#     - The @ operator does matrix multiplication (same as np.dot for 2D arrays)
#     - Consider printing progress: print(f"Block {i+1}/{len(blocks)}", end="\r")
#
# EXAMPLE:
#     >>> r, g, b = split_channels(load_image("samples/sample.jpg"))
#     >>> r_compressed = compress_channel(r, block_size=32, k=8)
#     >>> print(r_compressed.shape)  # Same shape as r
#
def compress_channel(channel_matrix, block_size, k):
    """Compress a single color channel using block-based SVD."""
    
    # YOUR CODE BELOW ↓↓↓
    
    # Step 1: Split the channel into blocks
    # HINT: blocks, padded_shape = image_to_blocks(channel_matrix, block_size)
    
    # Step 2: Store the original shape for later stitching
    # HINT: original_shape = channel_matrix.shape
    
    # Step 3: Create a list to hold compressed blocks
    # HINT: compressed_blocks = []
    
    # Step 4: Loop through each block:
    #     a) Run SVD: U, sigma, Vt = custom_svd(block, k)
    #     b) Reconstruct: compressed_block = U @ np.diag(sigma) @ Vt
    #     c) Append to compressed_blocks
    #     d) (Optional) Print progress every 100 blocks
    # HINT: for i, block in enumerate(blocks):
    #           U, sigma, Vt = custom_svd(block, k)
    #           compressed_block = U @ np.diag(sigma) @ Vt
    #           compressed_blocks.append(compressed_block)
    
    # Step 5: Stitch blocks back together
    # HINT: compressed_channel = blocks_to_image(compressed_blocks, padded_shape, original_shape, block_size)
    
    # Step 6: Return the compressed channel
    
    pass  # ← Remove this once you write your code


# ==============================================================================
# STEP 3.3: COMPRESS A FULL IMAGE (THE MAIN FUNCTION)
# ==============================================================================
#
# WHAT IT DOES:
#     The top-level function that takes an image path, compresses it,
#     and saves the result. This is what the CLI will call.
#
# THE ALGORITHM:
#     1. Load the image
#     2. Split into R, G, B channels
#     3. Calculate rank k from the keep percentage
#     4. Compress each channel independently
#     5. Merge the compressed channels back into an RGB image
#     6. Save the result
#     7. Return the compressed image (useful for notebook visualization)
#
# PARAMETERS:
#     image_path  : str — path to the input image
#     keep        : float — percentage of data to keep (default 20)
#     block_size  : int — block size for grid processing (default 32)
#     output_path : str — where to save the result (default "output/compressed.jpg")
#
# RETURNS:
#     compressed_image : numpy 3D array (H, W, 3) — the compressed image
#
# NOTEBOOK TEST:
#     >>> original = load_image("samples/sample.jpg")
#     >>> compressed = compress_image("samples/sample.jpg", keep=20, block_size=32)
#     >>> # Display side by side in matplotlib:
#     >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
#     >>> ax1.imshow(original); ax1.set_title("Original")
#     >>> ax2.imshow(compressed); ax2.set_title("Compressed (20% kept)")
#     >>> plt.show()
#
def compress_image(image_path, keep=20, block_size=32, output_path="output/compressed.jpg"):
    """Compress a full RGB image using block-based SVD."""
    
    # YOUR CODE BELOW ↓↓↓
    
    # Step 1: Load the image
    # HINT: image = load_image(image_path)
    
    # Step 2: Split into channels
    # HINT: red, green, blue = split_channels(image)
    
    # Step 3: Calculate the target rank
    # HINT: k = calculate_rank(block_size, keep)
    # HINT: print(f"Compressing with rank k={k} (keeping {keep}% of data)")
    
    # Step 4: Compress each channel
    # HINT: print("Compressing Red channel...")
    # HINT: red_compressed = compress_channel(red, block_size, k)
    # HINT: print("Compressing Green channel...")
    # HINT: green_compressed = compress_channel(green, block_size, k)
    # HINT: print("Compressing Blue channel...")
    # HINT: blue_compressed = compress_channel(blue, block_size, k)
    
    # Step 5: Merge channels back into RGB
    # HINT: compressed = merge_channels(red_compressed, green_compressed, blue_compressed)
    
    # Step 6: Save the result
    # HINT: save_image(compressed, output_path)
    # HINT: print(f"Saved compressed image to: {output_path}")
    
    # Step 7: Return the compressed image (for notebook visualization)
    # HINT: return compressed
    
    pass  # ← Remove this once you write your code
