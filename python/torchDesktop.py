"""Launch the session-based 2D/3D desktop application."""

import argparse

from desktop.app import SimulationApp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", choices=("2d", "3d"), default="2d")
    args = parser.parse_args()
    SimulationApp(int(args.dim[0])).mainloop()


if __name__ == "__main__":
    main()
