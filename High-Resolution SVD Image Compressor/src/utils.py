"""
==============================================================================
UTILITIES — Shared Helper Functions
==============================================================================

This file contains helper functions used across the project.
Build these as you need them — they're not critical to the core algorithm
but make your code cleaner and your CLI more user-friendly.

==============================================================================
"""

import sys

def progress_bar(current, total, prefix="Progress", length=40):
    """Print a visual progress bar in the terminal."""
    percent = current / total * 100
    filled = int(length * current / total)
    bar = '█' * filled + '░' * (length - filled)
    print(f"\r{prefix}: [{bar}] {percent:.1f}%", end="")
    sys.stdout.flush()
    if current == total:
        print()
