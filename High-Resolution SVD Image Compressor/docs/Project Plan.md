# Project Plan: High-Resolution SVD Image Compressor

## Project Objective

Build a Command Line Interface (CLI) tool that compresses images of any size (up to 4K/6K) using a completely custom, from-scratch Singular Value Decomposition (SVD) algorithm in pure Python and NumPy.

## Architectural Bottleneck & Solution

The Problem: Calculating SVD from scratch on a full 6K image matrix ($6000 \times 4000$) requires $O(N^3)$ time complexity. In pure Python, this would take days to process and likely crash the computer due to RAM limitations.

The Engineering Solution (Block Processing): We will split the massive image into tiny, independent grids (e.g., $16 \times 16$ or $32 \times 32$ pixels). We apply our custom SVD to each tiny block individually. This keeps memory usage flat, makes the math incredibly fast, and mimics how real-world codecs like JPEG actually work under the hood.

## Phase 1: The Core Math Engine (Custom SVD)

We will write the SVD algorithm from scratch using Eigendecomposition, Power Iteration, and Deflation.

Step 1.1: Power Iteration Function

Write a pure NumPy loop that takes a matrix, multiplies it by a random vector repeatedly, and normalizes it to find the dominant (largest) eigenvalue and its corresponding eigenvector.

Step 1.2: Deflation Logic

Once the dominant eigenvector is found, mathematically subtract its influence from the matrix so we can run Power Iteration again to find the second largest, then the third, up to our target rank $k$.

Step 1.3: The SVD Wrapper

Create a custom_svd(A, k) function.

Use $A^T A$ to find $V$ (Right Singular Vectors).

Use $A A^T$ to find $U$ (Left Singular Vectors).

Take the square roots of the eigenvalues to find $\Sigma$ (Singular Values).

## Phase 2: The Grid & Image Pipeline

We need to handle the conversion between standard image files and our mathematical matrices.

Step 2.1: Image Loading & Channel Splitting

Load an image (using Pillow/matplotlib.image) and split it into its Red, Green, and Blue (RGB) matrices.

Step 2.2: The Block Splitter

Write a function image_to_blocks(matrix, block_size) that slices a 2D matrix (like the Red channel) into a list of $32 \times 32$ sub-matrices.

Step 2.3: The Block Stitcher

Write the reverse function blocks_to_image(blocks, original_shape) that perfectly reassembles the compressed blocks back into a full-resolution 2D matrix.

## Phase 3: The Compression Logic

This connects the Math (Phase 1) to the Pipeline (Phase 2).

Step 3.1: Rank Calculation

Translate the user's requested "Compression Percentage" into an actual target rank $k$ for the blocks. (e.g., If the block is $32 \times 32$ and the user wants 25% data retention, $k = 8$).

Step 3.2: The Main Compression Loop

Iterate through all blocks for R, G, and B.

Pass each block to custom_svd(block, k).

Reconstruct the compressed block: $U_k \cdot \Sigma_k \cdot V_k^T$.

Stitch the image back together and stack the RGB channels.

## Phase 4: The Command Line Interface (CLI)

We will wrap the entire pipeline into a professional, easy-to-use terminal command.

Step 4.1: Argument Parsing

Use argparse to accept --image (path), --keep (percentage of data to keep, default 20%), and --block_size (default 32).

Step 4.2: File I/O & Feedback

Ensure the script safely loads the file, processes it, prints a progress bar (optional but highly recommended for 4K images), and saves the compressed output locally (e.g., compressed_output.jpg).