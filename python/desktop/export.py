"""Export a displayed session without changing its state."""

import csv
from pathlib import Path

import numpy as np

from cfdSolver.torchGeometry import axes, coordinates
from cfdSolver.torchIO import write_vtk, write_xdmf


def export_session(path, session):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".vtr", ".vts"):
        return write_vtk(path, session)
    if suffix in (".xdmf", ".xmf"):
        return write_xdmf(path, session)
    if suffix == ".h5":
        return session.save(path)
    if suffix != ".csv":
        raise ValueError("Choose CSV, HDF5, VTK or XDMF output")
    labels = coordinates(session.config)
    centers = [(a[:-1] + a[1:]) / 2 for a in axes(session.config)]
    grids = np.meshgrid(*centers[::-1], indexing="ij")[::-1]
    primitive = session.primitive
    names = labels + ["p", "rho"] + ["u_" + a for a in labels]
    values = (
        list(grids) + [primitive[-1], primitive[0]] + list(primitive[1:-1])
    )
    for name, value in (("lambda", session.lam), ("beta", session.beta)):
        if value is not None:
            names.append(name)
            values.append(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(zip(*(v.ravel() for v in values)))
    return str(path)
