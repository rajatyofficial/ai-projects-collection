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

# pyrefly: ignore [missing-import]
import numpy as np
from PIL import Image

def load_image(path:str) -> np.ndarray:
    """Load an image file and return as a normalized NumPy array.""" 
    pil_image = Image.open(path).convert("RGB")
    image_array = np.array(pil_image,dtype=np.float64)
    image_array /=255.0
    return image_array


def split_channels(image:np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Split an RGB image into its three color channels."""
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    return r,g,b

def image_to_blocks(matrix, block_size=32):
    """Split a 2D matrix into a list of block_size × block_size sub-matrices."""
    H, W = matrix.shape
    H_padded = H + (block_size - H % block_size) % block_size
    W_padded = W + (block_size - W % block_size) % block_size
    padded = np.zeros((H_padded, W_padded))
    padded[:H, :W] = matrix
    
    blocks = []
    for row in range(0, H_padded, block_size):
        for col in range(0, W_padded, block_size):
            block = padded[row:row+block_size, col:col+block_size]
            blocks.append(block)
    return blocks,(H_padded,W_padded)

def blocks_to_image(blocks, padded_shape, original_shape, block_size=32):
    """Reassemble a list of blocks into a full 2D matrix."""
    matrix = np.zeros(padded_shape)
    blocks_per_row = padded_shape[1] // block_size
    for i, block in enumerate(blocks):
        row_idx = i // blocks_per_row
        col_idx = i % blocks_per_row
        row_start = row_idx * block_size
        col_start = col_idx * block_size
        matrix[row_start:row_start+block_size, col_start:col_start+block_size] = block
    return matrix[:original_shape[0], :original_shape[1]]


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
