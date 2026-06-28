"""
==============================================================================
CLI ENTRY POINT — Phase 4: Command Line Interface
==============================================================================

This is the LAST file you build. It wraps your entire compression pipeline
into a professional terminal command that anyone can use.

USAGE:
    python cli.py --image samples/sample.jpg --keep 20 --block_size 32

ARGUMENTS:
    --image       : (REQUIRED) Path to the input image file
    --keep        : (OPTIONAL) Percentage of data to keep (default: 20)
    --block_size  : (OPTIONAL) Block size for grid processing (default: 32)
    --output      : (OPTIONAL) Output file path (default: output/compressed.jpg)

WHAT THIS FILE DOES:
    1. Parse command-line arguments using Python's argparse module
    2. Validate the inputs (file exists? percentage in range? etc.)
    3. Call compress_image() from src/compressor.py
    4. Print helpful feedback (timing, file sizes, compression ratio)

DEPENDENCIES (imports from your completed files):
    from src.compressor import compress_image
    from src.utils import validate_image_path, validate_percentage, validate_block_size

NOTEBOOK TEST (before building this, test in Notebook.ipynb Section 4):
    # Simulate what the CLI does:
    >>> from src.compressor import compress_image
    >>> compressed = compress_image("samples/sample.jpg", keep=20, block_size=32)
    # If this works in the notebook, the CLI is just a thin wrapper around it.
==============================================================================
"""

import argparse
import os
import time

# NOTE: Uncomment these imports once you've completed Phases 1-3!
# from src.compressor import compress_image
# from src.utils import validate_image_path, validate_percentage, validate_block_size


# ==============================================================================
# STEP 4.1: ARGUMENT PARSING
# ==============================================================================
#
# WHAT IT DOES:
#     Sets up argparse to accept command-line arguments and parse them
#     into clean Python variables.
#
# ABOUT ARGPARSE (if you're new to it):
#     argparse is Python's built-in library for creating CLI tools.
#     It automatically:
#     - Parses command-line arguments (--image, --keep, etc.)
#     - Validates types (ensures --keep is a number)
#     - Generates --help text
#     - Gives friendly error messages for bad input
#
# HOW TO USE ARGPARSE:
#     1. Create a parser:    parser = argparse.ArgumentParser(description="...")
#     2. Add arguments:      parser.add_argument("--name", type=str, ...)
#     3. Parse:              args = parser.parse_args()
#     4. Access values:      args.image, args.keep, etc.
#
# EXAMPLE OUTPUT:
#     $ python cli.py --help
#     usage: cli.py [-h] --image IMAGE [--keep KEEP] [--block_size BLOCK_SIZE]
#     
#     High-Resolution SVD Image Compressor
#     
#     optional arguments:
#       -h, --help            show this help message and exit
#       --image IMAGE         Path to the input image file
#       --keep KEEP           Percentage of data to keep (default: 20)
#       --block_size BLOCK_SIZE
#                             Block size for grid processing (default: 32)
#
def create_parser():
    """Create and configure the argument parser."""
    
    # YOUR CODE BELOW ↓↓↓
    
    # Step 1: Create the parser with a description
    # HINT: parser = argparse.ArgumentParser(
    #           description="High-Resolution SVD Image Compressor — "
    #                       "Compress images using custom SVD from scratch"
    #       )
    
    # Step 2: Add --image argument (REQUIRED)
    # HINT: parser.add_argument(
    #           "--image",
    #           type=str,
    #           required=True,
    #           help="Path to the input image file (e.g., samples/sample.jpg)"
    #       )
    
    # Step 3: Add --keep argument (OPTIONAL, default 20)
    # HINT: parser.add_argument(
    #           "--keep",
    #           type=float,
    #           default=20,
    #           help="Percentage of data to keep, 1-100 (default: 20)"
    #       )
    
    # Step 4: Add --block_size argument (OPTIONAL, default 32)
    # HINT: parser.add_argument(
    #           "--block_size",
    #           type=int,
    #           default=32,
    #           help="Block size for grid processing, must be power of 2 (default: 32)"
    #       )
    
    # Step 5: Add --output argument (OPTIONAL)
    # HINT: parser.add_argument(
    #           "--output",
    #           type=str,
    #           default="output/compressed.jpg",
    #           help="Output file path (default: output/compressed.jpg)"
    #       )
    
    # Step 6: Return the parser
    
    pass  # ← Remove this once you write your code


# ==============================================================================
# STEP 4.2: THE MAIN FUNCTION
# ==============================================================================
#
# WHAT IT DOES:
#     The main entry point. Parses arguments, validates inputs,
#     runs compression, and prints results.
#
# THE FLOW:
#     1. Parse arguments
#     2. Validate inputs (file exists? keep% in range? block_size valid?)
#     3. Create output directory if it doesn't exist
#     4. Print what we're about to do
#     5. Start timer
#     6. Call compress_image()
#     7. Stop timer
#     8. Print results (time taken, file sizes, compression ratio)
#
# HINTS FOR FILE SIZE:
#     - os.path.getsize(path) → file size in bytes
#     - To convert bytes to KB: size_bytes / 1024
#     - Compression ratio = original_size / compressed_size
#
def main():
    """Main CLI entry point."""
    
    # YOUR CODE BELOW ↓↓↓
    
    # Step 1: Create parser and parse arguments
    # HINT: parser = create_parser()
    # HINT: args = parser.parse_args()
    
    # Step 2: Validate inputs
    # HINT: validate_image_path(args.image)
    # HINT: validate_percentage(args.keep)
    # HINT: validate_block_size(args.block_size)
    
    # Step 3: Create output directory if it doesn't exist
    # HINT: os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Step 4: Print a summary of what we're about to do
    # HINT: print("=" * 60)
    # HINT: print("  SVD Image Compressor")
    # HINT: print("=" * 60)
    # HINT: print(f"  Input:      {args.image}")
    # HINT: print(f"  Output:     {args.output}")
    # HINT: print(f"  Keep:       {args.keep}%")
    # HINT: print(f"  Block size: {args.block_size}×{args.block_size}")
    # HINT: print("=" * 60)
    
    # Step 5: Run compression with timing
    # HINT: start = time.time()
    # HINT: compress_image(args.image, args.keep, args.block_size, args.output)
    # HINT: elapsed = time.time() - start
    
    # Step 6: Print results
    # HINT: original_size = os.path.getsize(args.image) / 1024  # KB
    # HINT: compressed_size = os.path.getsize(args.output) / 1024  # KB
    # HINT: print(f"\n  Time:        {elapsed:.2f}s")
    # HINT: print(f"  Original:    {original_size:.1f} KB")
    # HINT: print(f"  Compressed:  {compressed_size:.1f} KB")
    # HINT: print(f"  Ratio:       {original_size/compressed_size:.1f}x")
    # HINT: print("=" * 60)
    # HINT: print("  Done! ✓")
    
    pass  # ← Remove this once you write your code


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
# This block runs main() when you execute: python cli.py --image ...
# It does NOT run when you import this file from the notebook.
#
if __name__ == "__main__":
    main()
