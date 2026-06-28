# Project Plan: High-Resolution SVD Image Compressor

## Project Objective

Build a Command Line Interface (CLI) tool that compresses images of any size (up to 4K/6K) using a completely custom, from-scratch Singular Value Decomposition (SVD) algorithm in pure Python and NumPy.

## Architectural Bottleneck & Solution

The Problem: Calculating SVD from scratch on a full 6K image matrix ($6000 \times 4000$) requires $O(N^3)$ time complexity. In pure Python, this would take days to process and likely crash the computer due to RAM limitations.

The Engineering Solution (Block Processing): We will split the massive image into tiny, independent grids (e.g., $16 \times 16$ or $32 \times 32$ pixels). We apply our custom SVD to each tiny block individually. This keeps memory usage flat, makes the math incredibly fast, and mimics how real-world codecs like JPEG actually work under the hood.

---

## The Notebook-First Learning Approach

**You are NOT building a black-box CLI.** You are building each function one at a time, testing it interactively in the notebook, visualizing the results, and *understanding* the math — then the CLI is just a wrapper at the end.

### Workflow: Code in `src/` → Test & Learn in `Notebook.ipynb`

The rhythm is: **write a function → jump to notebook → test it → visualize it → understand it → move on.** Never write the next function until the previous one passes your notebook tests.

---

## Project Structure

```
High-Resolution SVD Image Compressor/
│
├── docs/
│   ├── Project Plan.md                        # this file
│   └── Singular Value Decomposition ... .pdf  # math reference
│
├── Notebooks/
│   ├── Notebook.ipynb                   # 🧪 THE MAIN LEARNING LAB
|
├── src/                                 # all production source code (importable by notebook)
│   ├── __init__.py
│   ├── svd_engine.py                    # Phase 1: power_iteration, deflation, custom_svd
│   ├── image_pipeline.py               # Phase 2: load, channel split, block split/stitch
│   ├── compressor.py                   # Phase 3: rank calc, compression loop, RGB merge
│   └── utils.py                        # Shared helpers (progress bar, timing, validation)
│
├── cli.py                              # Phase 4 — CLI entry point (argparse)
│
├── samples/                            # sample images for testing
│   └── (place test images here)
│
├── output/                             # default output directory (gitignored)
│
└── README.md                           # project overview & usage instructions
```

---

## Phase 1: The Core Math Engine (Custom SVD)

We will write the SVD algorithm from scratch using Eigendecomposition, Power Iteration, and Deflation.

**Target file:** `src/svd_engine.py`

### Step 1.1: Power Iteration Function

Write a pure NumPy loop that takes a matrix, multiplies it by a random vector repeatedly, and normalizes it to find the dominant (largest) eigenvalue and its corresponding eigenvector.

### Step 1.2: Deflation Logic

Once the dominant eigenvector is found, mathematically subtract its influence from the matrix so we can run Power Iteration again to find the second largest, then the third, up to our target rank $k$.

### Step 1.3: The SVD Wrapper

Create a `custom_svd(A, k)` function.

Use $A^T A$ to find $V$ (Right Singular Vectors).

Use $A A^T$ to find $U$ (Left Singular Vectors).

Take the square roots of the eigenvalues to find $\Sigma$ (Singular Values).

### 📓 Notebook Section 1: Test the Math

| Cell Group | What You'll Do | What You'll See |
|------------|---------------|-----------------|
| **1.1 — Power Iteration** | Import `power_iteration` from `src.svd_engine`. Run it on a small 4×4 matrix. | Print eigenvalue per iteration → watch it converge. Plot convergence curve. |
| **1.2 — Deflation** | Call `deflation` to strip the dominant eigenvector. Run power iteration again. | Print the 2nd eigenvalue. Verify it's smaller than the 1st. |
| **1.3 — Custom SVD** | Call `custom_svd(A, k)` on the same 4×4 matrix. | Print U, Σ, Vᵀ. Reconstruct A ≈ UΣVᵀ. Compare with `np.linalg.svd` — see how close your version is. |
| **1.4 — Sanity Check** | Test on a 32×32 random matrix (actual block size). | Compute reconstruction error. Verify it's near-zero when k = full rank. |

---

## Phase 2: The Grid & Image Pipeline

We need to handle the conversion between standard image files and our mathematical matrices.

**Target file:** `src/image_pipeline.py`

### Step 2.1: Image Loading & Channel Splitting

Load an image (using Pillow/matplotlib.image) and split it into its Red, Green, and Blue (RGB) matrices.

### Step 2.2: The Block Splitter

Write a function `image_to_blocks(matrix, block_size)` that slices a 2D matrix (like the Red channel) into a list of $32 \times 32$ sub-matrices.

### Step 2.3: The Block Stitcher

Write the reverse function `blocks_to_image(blocks, original_shape)` that perfectly reassembles the compressed blocks back into a full-resolution 2D matrix.

### 📓 Notebook Section 2: Test the Pipeline

| Cell Group | What You'll Do | What You'll See |
|------------|---------------|-----------------|
| **2.1 — Load & Display** | Import `load_image` from `src.image_pipeline`. Load `samples/sample.jpg`. | Display the image inline with `matplotlib`. Print shape and dtype. |
| **2.2 — Channel Split** | Call `split_channels`. | Display R, G, B as separate heatmaps side-by-side (use `cmap='Reds'`, `'Greens'`, `'Blues'`). |
| **2.3 — Block Splitting** | Call `image_to_blocks(red_channel, 32)`. | Print number of blocks. Display a grid of the first 16 blocks as tiny images. |
| **2.4 — Block Stitching** | Call `blocks_to_image` on the blocks you just split. | Display the reconstructed channel. Assert `np.array_equal(original, reconstructed)` — must be pixel-perfect. |

---

## Phase 3: The Compression Logic

This connects the Math (Phase 1) to the Pipeline (Phase 2).

**Target file:** `src/compressor.py`

### Step 3.1: Rank Calculation

Translate the user's requested "Compression Percentage" into an actual target rank $k$ for the blocks. (e.g., If the block is $32 \times 32$ and the user wants 25% data retention, $k = 8$).

### Step 3.2: The Main Compression Loop

Iterate through all blocks for R, G, and B.

Pass each block to `custom_svd(block, k)`.

Reconstruct the compressed block: $U_k \cdot \Sigma_k \cdot V_k^T$.

Stitch the image back together and stack the RGB channels.

### 📓 Notebook Section 3: Test the Compression

| Cell Group | What You'll Do | What You'll See |
|------------|---------------|-----------------|
| **3.1 — Single Block Compression** | Take one 32×32 block. Run `custom_svd(block, k)` for k = 1, 4, 8, 16, 32. | Show 5 images side-by-side: watch the block go from blurry → crisp as k increases. |
| **3.2 — Rank Calculator** | Call `calculate_rank(32, keep_pct)` for various percentages. | Print a table: keep% → rank k. Verify the math makes sense. |
| **3.3 — Full Image Compression** | Call `compress_image(path, keep=20, block_size=32)`. | Display original vs compressed side-by-side. Print file sizes & compression ratio. |
| **3.4 — Quality Sweep** | Loop: compress at keep = 5%, 10%, 20%, 50%, 80%. | Plot a chart: keep% on x-axis, visual quality (PSNR or SSIM) on y-axis. Display a carousel of results. |

---

## Phase 4: The Command Line Interface (CLI)

We will wrap the entire pipeline into a professional, easy-to-use terminal command.

**Target file:** `cli.py`

### Step 4.1: Argument Parsing

Use `argparse` to accept `--image` (path), `--keep` (percentage of data to keep, default 20%), and `--block_size` (default 32).

### Step 4.2: File I/O & Feedback

Ensure the script safely loads the file, processes it, prints a progress bar (optional but highly recommended for 4K images), and saves the compressed output locally (e.g., `compressed_output.jpg`).

### 📓 Notebook Section 4: Final Demo & Benchmarks

| Cell Group | What You'll Do | What You'll See |
|------------|---------------|-----------------|
| **4.1 — High-Res Test** | Run on a large (2K+) image. | Before/after at full resolution. Timing stats. |
| **4.2 — CLI Dry Run** | Simulate CLI args and call the same pipeline. | Confirm the CLI will work identically to the notebook. |

---

## Module Responsibility Map

| File | Phase | Key Functions |
|------|-------|---------------|
| `src/svd_engine.py` | 1 | `power_iteration(M, iters)`, `deflation(M, eigvec, eigval)`, `custom_svd(A, k)` → U, Σ, Vᵀ |
| `src/image_pipeline.py` | 2 | `load_image(path)`, `split_channels(img)`, `image_to_blocks(matrix, size)`, `blocks_to_image(blocks, shape)` |
| `src/compressor.py` | 3 | `calculate_rank(block_size, keep_pct)`, `compress_image(path, keep, block_size)` |
| `cli.py` | 4 | `argparse` entry: `--image`, `--keep` (default 20%), `--block_size` (default 32) |
| `src/utils.py` | — | Progress bar, timer decorator, input validation |
| `Notebooks/Notebook.ipynb` | ALL | 🧪 Interactive lab — test every function, visualize results, learn the math |

---

## Build Order

1. **`src/svd_engine.py`** → 📓 Notebook Section 1 (test on tiny matrices, compare with NumPy)
2. **`src/image_pipeline.py`** → 📓 Notebook Section 2 (visualize channels, verify roundtrip)
3. **`src/compressor.py`** → 📓 Notebook Section 3 (compress at various ranks, quality sweep)
4. **`cli.py`** → 📓 Notebook Section 4 (final demo, benchmarks)

> **The rhythm is: write a function → jump to notebook → test it → visualize it → understand it → move on.**