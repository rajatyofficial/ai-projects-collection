"""
==============================================================================
IMAGE PIPELINE — Phase 2: Grid & Image Pipeline
==============================================================================

This file handles everything between "a .jpg file on disk" and
"a list of tiny NumPy matrices ready for SVD compression."

WHAT THIS FILE DOES:
    1. Load an image file from disk → convert to a NumPy array
    2. Split the image into its 3 color channels: Red, Green, Blue
    3. Chop each channel into small blocks (e.g., 32×32 pixels each)
    4. After compression, stitch those blocks back into a full image

WHY BLOCK PROCESSING?
    A 4K image is 3840×2160 pixels. Running SVD on that full matrix would
    require O(N³) = O(3840³) ≈ 56 BILLION operations. Impossible in Python.
    
    Instead, we split it into 32×32 blocks (1024 pixels each). SVD on a
    32×32 matrix is O(32³) = 32,768 operations. That's 1.7 MILLION times faster.
    
    For a 4K image, that's about 8,100 tiny blocks — each processed in
    microseconds. This is exactly how JPEG compression works under the hood!

BUILD ORDER:
    1. load_image()       — load a .jpg/.png file into a NumPy array
    2. split_channels()   — separate R, G, B into three 2D matrices
    3. image_to_blocks()  — chop a 2D matrix into 32×32 blocks
    4. blocks_to_image()  — stitch blocks back into a full 2D matrix

DEPENDENCIES:
    - numpy (already imported)
    - PIL (Pillow) — for loading image files
    You may also use matplotlib.image if you prefer, but Pillow is recommended.

NOTEBOOK TEST (after completing this file):
    In Notebook.ipynb, Section 2:
    - Load samples/sample.jpg → display it inline
    - Show R, G, B channels as separate colored heatmaps
    - Split a channel into blocks → display a grid of blocks
    - Stitch blocks back → verify pixel-perfect roundtrip with np.array_equal()
==============================================================================
"""

import numpy as np
from PIL import Image


# ==============================================================================
# STEP 2.1a: LOAD IMAGE
# ==============================================================================
#
# WHAT IT DOES:
#     Takes a file path to an image (.jpg, .png, etc.) and returns it as
#     a NumPy array with shape (height, width, 3) and dtype float64.
#
# THE ALGORITHM:
#     1. Open the image file using PIL.Image.open()
#     2. Convert it to RGB mode (in case it's RGBA, grayscale, etc.)
#     3. Convert the PIL Image to a NumPy array
#     4. Convert pixel values from integers (0-255) to floats (0.0-1.0)
#        by dividing by 255.0
#
# WHY CONVERT TO FLOAT?
#     SVD works with real numbers, not integers. Also, when we reconstruct
#     the image after compression, values might go slightly below 0 or above 1.
#     Working in float-space makes the math cleaner.
#
# PARAMETERS:
#     path : str — file path to the image (e.g., "samples/sample.jpg")
#
# RETURNS:
#     image : numpy 3D array of shape (H, W, 3) — normalized to [0.0, 1.0]
#             H = height in pixels, W = width in pixels, 3 = RGB channels
#
# HINTS:
#     - PIL.Image.open(path) returns a PIL Image object
#     - .convert("RGB") ensures 3 channels (handles RGBA, grayscale, etc.)
#     - np.array(pil_image) converts PIL → NumPy
#     - Divide by 255.0 to normalize to [0, 1]
#     - Use .astype(np.float64) for precision
#
# EXAMPLE:
#     >>> img = load_image("samples/sample.jpg")
#     >>> print(img.shape)   # e.g., (1080, 1920, 3)
#     >>> print(img.dtype)   # float64
#     >>> print(img.min(), img.max())  # 0.0 ... 1.0
#
def load_image(path):
    """Load an image file and return as a normalized NumPy array."""
    
    # YOUR CODE BELOW ↓↓↓
    
    # Step 1: Open the image with PIL
    # HINT: pil_image = Image.open(path)
    
    # Step 2: Convert to RGB (handles RGBA, grayscale, palette modes)
    # HINT: pil_image = pil_image.convert("RGB")
    
    # Step 3: Convert to NumPy array
    # HINT: image_array = np.array(pil_image, dtype=np.float64)
    
    # Step 4: Normalize pixel values from [0, 255] to [0.0, 1.0]
    # HINT: image_array = image_array / 255.0
    
    # Step 5: Return the normalized array
    
    pass  # ← Remove this once you write your code


# ==============================================================================
# STEP 2.1b: SPLIT CHANNELS
# ==============================================================================
#
# WHAT IT DOES:
#     Takes an RGB image array (H, W, 3) and splits it into three
#     separate 2D matrices: one for Red, one for Green, one for Blue.
#
# WHY SPLIT?
#     SVD operates on 2D matrices, not 3D arrays. We need to compress
#     each color channel independently, then reassemble them at the end.
#
# PARAMETERS:
#     image : numpy 3D array of shape (H, W, 3) — the loaded image
#
# RETURNS:
#     A tuple of three 2D arrays: (red, green, blue)
#     - red   : numpy 2D array of shape (H, W) — red channel values [0.0, 1.0]
#     - green : numpy 2D array of shape (H, W) — green channel values [0.0, 1.0]
#     - blue  : numpy 2D array of shape (H, W) — blue channel values [0.0, 1.0]
#
# HINTS:
#     - NumPy slicing: image[:, :, 0] gives ALL rows, ALL columns, channel 0 (Red)
#     - Channel indices: 0 = Red, 1 = Green, 2 = Blue
#
# EXAMPLE:
#     >>> img = load_image("samples/sample.jpg")
#     >>> r, g, b = split_channels(img)
#     >>> print(r.shape)  # (1080, 1920) — just the red values
#
def split_channels(image):
    """Split an RGB image into its three color channels."""
    
    # YOUR CODE BELOW ↓↓↓
    
    # Step 1: Extract the Red channel (index 0)
    # HINT: red = image[:, :, 0]
    
    # Step 2: Extract the Green channel (index 1)
    # HINT: green = image[:, :, 1]
    
    # Step 3: Extract the Blue channel (index 2)
    # HINT: blue = image[:, :, 2]
    
    # Step 4: Return as a tuple: (red, green, blue)
    
    pass  # ← Remove this once you write your code


# ==============================================================================
# STEP 2.2: THE BLOCK SPLITTER
# ==============================================================================
#
# WHAT IT DOES:
#     Takes a 2D matrix (one color channel) and slices it into a grid of
#     small square blocks (e.g., 32×32 pixels each).
#
# THE ALGORITHM:
#     1. Get the height (H) and width (W) of the matrix
#     2. PAD the matrix so H and W are exactly divisible by block_size
#        (add rows/columns of zeros if needed)
#     3. Loop through the padded matrix in steps of block_size:
#        - For row in range(0, H_padded, block_size):
#            For col in range(0, W_padded, block_size):
#                Extract the block: matrix[row:row+block_size, col:col+block_size]
#                Add it to the list
#     4. Return the list of blocks AND the padded shape (needed for stitching)
#
# WHY PAD?
#     If the image is 1080×1920 and block_size is 32:
#     - 1080 / 32 = 33.75 → not evenly divisible!
#     - We pad to 1088×1920 (1088 = 34 × 32) so all blocks are the same size
#     - The padding is zeros (black pixels) that get trimmed during stitching
#
# PARAMETERS:
#     matrix     : numpy 2D array of shape (H, W) — one color channel
#     block_size : int — the size of each square block (default 32)
#
# RETURNS:
#     blocks       : list of numpy 2D arrays, each of shape (block_size, block_size)
#     padded_shape : tuple (H_padded, W_padded) — needed by blocks_to_image()
#
# HINTS:
#     - To calculate padded dimension:
#       import math
#       H_padded = math.ceil(H / block_size) * block_size
#       OR equivalently: H_padded = H + (block_size - H % block_size) % block_size
#     - To create the padded matrix:
#       padded = np.zeros((H_padded, W_padded))
#       padded[:H, :W] = matrix    (copy original into top-left corner)
#     - Use nested for loops with range(0, H_padded, block_size)
#
# EXAMPLE:
#     >>> r, g, b = split_channels(load_image("samples/sample.jpg"))
#     >>> blocks, padded_shape = image_to_blocks(r, block_size=32)
#     >>> print(len(blocks))        # e.g., 2040 blocks for a 1080×1920 image
#     >>> print(blocks[0].shape)    # (32, 32)
#     >>> print(padded_shape)       # (1088, 1920)
#
def image_to_blocks(matrix, block_size=32):
    """Split a 2D matrix into a list of block_size × block_size sub-matrices."""
    
    # YOUR CODE BELOW ↓↓↓
    
    # Step 1: Get the original height and width
    # HINT: H, W = matrix.shape
    
    # Step 2: Calculate the padded dimensions (must be divisible by block_size)
    # HINT: H_padded = H + (block_size - H % block_size) % block_size
    # HINT: W_padded = W + (block_size - W % block_size) % block_size
    # NOTE: The "% block_size" at the end handles the case where H is already divisible
    
    # Step 3: Create a padded matrix filled with zeros
    # HINT: padded = np.zeros((H_padded, W_padded))
    # HINT: padded[:H, :W] = matrix   (copy original data into top-left)
    
    # Step 4: Extract blocks using nested loops
    # HINT: blocks = []
    # HINT: for row in range(0, H_padded, block_size):
    #           for col in range(0, W_padded, block_size):
    #               block = padded[row:row+block_size, col:col+block_size]
    #               blocks.append(block)
    
    # Step 5: Return (blocks, (H_padded, W_padded))
    
    pass  # ← Remove this once you write your code


# ==============================================================================
# STEP 2.3: THE BLOCK STITCHER
# ==============================================================================
#
# WHAT IT DOES:
#     The REVERSE of image_to_blocks(). Takes a list of compressed blocks
#     and reassembles them into a full 2D matrix.
#
# THE ALGORITHM:
#     1. Create an empty matrix of size padded_shape (H_padded × W_padded)
#     2. Figure out how many columns of blocks there are:
#        blocks_per_row = W_padded / block_size
#     3. Loop through blocks and place each one at the correct position:
#        - Block index i → row = (i // blocks_per_row) * block_size
#                        → col = (i % blocks_per_row) * block_size
#     4. Trim the result back to original_shape (remove the padding)
#
# PARAMETERS:
#     blocks         : list of numpy 2D arrays (each block_size × block_size)
#     padded_shape   : tuple (H_padded, W_padded) — from image_to_blocks()
#     original_shape : tuple (H, W) — the original image channel dimensions
#     block_size     : int — the size of each block (default 32)
#
# RETURNS:
#     matrix : numpy 2D array of shape (H, W) — the reconstructed channel
#
# HINTS:
#     - blocks_per_row = padded_shape[1] // block_size
#     - For block index i:
#       row_idx = (i // blocks_per_row)    ← which row of blocks
#       col_idx = (i % blocks_per_row)     ← which column of blocks
#       row_start = row_idx * block_size
#       col_start = col_idx * block_size
#     - Place block: matrix[row_start:row_start+block_size, col_start:col_start+block_size] = blocks[i]
#     - Trim at the end: result = matrix[:original_shape[0], :original_shape[1]]
#
# IMPORTANT TEST:
#     After implementing both image_to_blocks and blocks_to_image, run this in the notebook:
#     >>> blocks, padded_shape = image_to_blocks(red_channel, 32)
#     >>> reconstructed = blocks_to_image(blocks, padded_shape, red_channel.shape, 32)
#     >>> assert np.array_equal(red_channel, reconstructed)  # MUST be True!
#     If this fails, you have a bug in your indexing.
#
def blocks_to_image(blocks, padded_shape, original_shape, block_size=32):
    """Reassemble a list of blocks into a full 2D matrix."""
    
    # YOUR CODE BELOW ↓↓↓
    
    # Step 1: Create an empty matrix of padded_shape
    # HINT: matrix = np.zeros(padded_shape)
    
    # Step 2: Calculate how many blocks fit in one row
    # HINT: blocks_per_row = padded_shape[1] // block_size
    
    # Step 3: Loop through blocks and place each one
    # HINT: for i, block in enumerate(blocks):
    #           row_idx = i // blocks_per_row
    #           col_idx = i % blocks_per_row
    #           row_start = row_idx * block_size
    #           col_start = col_idx * block_size
    #           matrix[row_start:row_start+block_size, col_start:col_start+block_size] = block
    
    # Step 4: Trim padding and return
    # HINT: return matrix[:original_shape[0], :original_shape[1]]
    
    pass  # ← Remove this once you write your code


# ==============================================================================
# HELPER: MERGE CHANNELS
# ==============================================================================
#
# WHAT IT DOES:
#     Takes three separate R, G, B 2D matrices and stacks them back
#     into a single (H, W, 3) image array ready for saving/display.
#
# ALSO:
#     Clips values to [0.0, 1.0] because SVD reconstruction can produce
#     values slightly outside this range due to numerical precision.
#
# PARAMETERS:
#     red   : numpy 2D array (H, W)
#     green : numpy 2D array (H, W)
#     blue  : numpy 2D array (H, W)
#
# RETURNS:
#     image : numpy 3D array (H, W, 3) — clipped to [0.0, 1.0]
#
# HINTS:
#     - np.stack([red, green, blue], axis=2)  — stacks along the 3rd axis
#     - np.clip(array, 0.0, 1.0)             — clamps values to range
#
def merge_channels(red, green, blue):
    """Merge R, G, B channels back into an RGB image array."""
    
    # YOUR CODE BELOW ↓↓↓
    
    # Step 1: Stack the three channels along axis=2
    # HINT: image = np.stack([red, green, blue], axis=2)
    
    # Step 2: Clip values to [0, 1] (SVD can produce values slightly out of range)
    # HINT: image = np.clip(image, 0.0, 1.0)
    
    # Step 3: Return the merged image
    
    pass  # ← Remove this once you write your code


# ==============================================================================
# HELPER: SAVE IMAGE
# ==============================================================================
#
# WHAT IT DOES:
#     Takes a float image array [0.0, 1.0] and saves it as a .jpg or .png file.
#
# PARAMETERS:
#     image_array : numpy 3D array (H, W, 3) — float values in [0.0, 1.0]
#     output_path : str — where to save (e.g., "output/compressed.jpg")
#
# HINTS:
#     - Convert back to uint8: (image_array * 255).astype(np.uint8)
#     - Use PIL: Image.fromarray(uint8_array).save(output_path)
#
def save_image(image_array, output_path):
    """Save a float image array as an image file."""
    
    # YOUR CODE BELOW ↓↓↓
    
    # Step 1: Convert from float [0, 1] back to uint8 [0, 255]
    # HINT: uint8_array = (image_array * 255).astype(np.uint8)
    
    # Step 2: Create PIL Image and save
    # HINT: pil_image = Image.fromarray(uint8_array)
    # HINT: pil_image.save(output_path)
    
    pass  # ← Remove this once you write your code
