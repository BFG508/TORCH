"""
TORCH Plotting Utility (Python)
Exports simulation CSV datasets to high-resolution PNG and SVG figures.
Optionally overlays Toro's exact Riemann solver solution for benchmark comparison.

Author: BFG508
"""

import argparse
from cfdSolver import plot_csv


def main() -> None:
    """CLI entry point for exporting high-resolution PNG & SVG figures non-interactively."""
    parser = argparse.ArgumentParser(
        description="Export 1D CFD CSV Data to PNG & SVG at 600 DPI with exact Toro overlay"
    )
    parser.add_argument(
        "--file", type=str, default="out/sod_data.csv",
        help="Path to CSV data file"
    )
    parser.add_argument(
        "--save", type=str, default="out/sod_plot",
        help="Output filepath prefix (saves both .png and .svg at 600 DPI)"
    )
    parser.add_argument(
        "--exact", action="store_true", default=False,
        help="Overlay Toro's exact Riemann solution for Sod shock tube"
    )
    parser.add_argument(
        "--t_exact", type=float, default=0.2,
        help="Time evaluated for exact Riemann solver overlay"
    )

    args = parser.parse_args()
    plot_csv(args.file, args.save, overlay_exact=args.exact, t_exact=args.t_exact)


if __name__ == "__main__":
    main()
