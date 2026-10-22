"""Expose the source package to direct and discovered tests."""

from pathlib import Path
import sys

SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)
