"""Coordinate-aware slices and centerline profiles from complete 3D CSV
grids.
"""

import csv
from pathlib import Path
import numpy as np


def plot_volume_csv(csv_path, save_prefix, slice_axis="z", slice_index=None):
    if slice_axis not in ("x", "y", "z"):
        raise ValueError("slice_axis must be x, y or z")
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = ("x", "y", "z", "p", "rho", "u", "v", "w")
        if not set(required).issubset(fields):
            raise ValueError("3D CSV requires x,y,z,p,rho,u,v,w")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no simulation data")
    columns = {
        key: np.array([float(row[key]) for row in rows]) for key in fields
    }
    if any(not np.isfinite(value).all() for value in columns.values()):
        raise ValueError("CSV data must be finite")
    coordinates = {axis: np.unique(columns[axis]) for axis in ("x", "y", "z")}
    shape = tuple(len(coordinates[axis]) for axis in ("z", "y", "x"))
    triples = np.column_stack([columns[axis] for axis in ("z", "y", "x")])
    if len(rows) != np.prod(shape) or len(np.unique(triples, axis=0)) != len(
        rows
    ):
        raise ValueError(
            "3D CSV must contain a complete rectangular grid with unique cells"
        )
    order = np.lexsort((columns["x"], columns["y"], columns["z"]))
    volumes = {
        key: values[order].reshape(shape) for key, values in columns.items()
    }
    dimension = ("z", "y", "x").index(slice_axis)
    index = shape[dimension] // 2 if slice_index is None else slice_index
    if (
        not isinstance(index, (int, np.integer))
        or not 0 <= index < shape[dimension]
    ):
        raise ValueError("slice_index is outside the volume")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    remaining = [axis for axis in ("z", "y", "x") if axis != slice_axis]
    vertical, horizontal = remaining
    xx, yy = np.meshgrid(coordinates[horizontal], coordinates[vertical])
    speed = np.sqrt(sum(volumes[axis] ** 2 for axis in ("u", "v", "w")))
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    selected = [
        ("rho", "Density [kg/m³]"),
        ("p", "Pressure [Pa]"),
        (
            ("lambda", "Reactant fraction")
            if "lambda" in volumes
            else ("speed", "Speed [m/s]")
        ),
    ]
    volumes["speed"] = speed
    for col, (key, title) in enumerate(selected):
        plane = np.take(volumes[key], index, axis=dimension)
        mesh = axes[0, col].pcolormesh(xx, yy, plane, shading="auto")
        figure.colorbar(mesh, ax=axes[0, col])
        axes[0, col].set(
            xlabel=f"{horizontal} [m]", ylabel=f"{vertical} [m]", title=title
        )
        # Profile through the middle row of the selected slice.
        axes[1, col].plot(coordinates[horizontal], plane[plane.shape[0] // 2])
        axes[1, col].set(
            xlabel=f"{horizontal} [m]", ylabel=title, title="Slice centerline"
        )
        axes[1, col].grid(True)
    figure.suptitle(
        f"3D slice: {slice_axis}={coordinates[slice_axis][index]:.6g} m"
    )
    figure.tight_layout()
    prefix = Path(save_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = (str(prefix) + ".png", str(prefix) + ".svg")
    try:
        figure.savefig(paths[0], dpi=200)
        figure.savefig(paths[1])
    finally:
        plt.close(figure)
    return paths
