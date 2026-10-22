"""Portable schema-3 leaf checkpoints and changing-topology outputs."""

import json
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np

from .torchAdaptive import AdaptiveSimulation
from .torchAdaptiveGeometry import LeafMesh, settings
from .torchGeometry import component_names, coordinates, map_points
from .torchSolver import physical_state


def fields(session):
    c, p = session.config, session.primitive
    result = {
        name: row for name, row in zip(component_names(c), session.state)
    }
    result["pressure"] = p[-1]
    for a, value in zip(coordinates(c), p[1:-1]):
        result["velocity_" + a] = value
    result["refinement_level"] = session.mesh.levels.astype(float)
    result["cell_volume"] = session.mesh.volumes
    if session.lam is not None:
        result["lambda"] = session.lam
    if session.beta is not None:
        result["beta"] = session.beta
    if c["geometry"] != "cartesian":
        for a, value in zip("xyz", cartesian_velocity(session)):
            result["velocity_cartesian_" + a] = value
    return result


def cartesian_velocity(session):
    c, p, x = session.config, session.primitive, session.mesh.centers
    d, g = c["dimension"], c["geometry"]
    zero = np.zeros(len(x))
    v = list(p[1:-1]) + [zero] * (3 - d)
    if g == "cartesian" or d == 1:
        return v
    if g == "cylindrical" and d == 2:
        return [v[0], zero, v[1]]
    theta = x[:, 1]
    if g == "spherical":
        phi = x[:, 2] if d == 3 else 0
        r, t, a = v
        return [
            (r * np.sin(theta) + t * np.cos(theta)) * np.cos(phi)
            - a * np.sin(phi),
            (r * np.sin(theta) + t * np.cos(theta)) * np.sin(phi)
            + a * np.cos(phi),
            r * np.cos(theta) - t * np.sin(theta),
        ]
    return [
        v[0] * np.cos(theta) - v[1] * np.sin(theta),
        v[0] * np.sin(theta) + v[1] * np.cos(theta),
        v[2],
    ]


def mesh_data(session):
    mesh, d = session.mesh, session.config["dimension"]
    offsets = np.array(
        {
            1: [(0,), (1,)],
            2: [(0, 0), (1, 0), (1, 1), (0, 1)],
            3: [
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1),
            ],
        }[d]
    )
    points = np.zeros((len(mesh.keys), len(offsets), 3))
    points[:, :, :d] = mesh.low[:, None] + offsets[None] * mesh.widths[:, None]
    points = map_points(session.config, points.reshape(-1, 3))
    cells = np.arange(len(points), dtype=np.int64).reshape(-1, len(offsets))
    return points, cells


def metadata(session):
    session._validate_config()
    if not physical_state(session.state) or session.state.shape != (
        session.config["dimension"] + 2,
        len(session.mesh.keys),
    ):
        raise ValueError("Invalid adaptive state")
    if (
        not np.isfinite(session.time)
        or not 0 <= session.time <= session.config["t_end"]
        or not isinstance(session.steps, int)
        or not 0 <= session.steps <= session.config["max_steps"]
    ):
        raise ValueError("Invalid adaptive counters")
    expected_lam = session.config["reactive"]
    expected_beta = expected_lam and session.config["kinetic_model"] == "znd"
    if (session.lam is not None) != expected_lam or (
        session.beta is not None
    ) != expected_beta:
        raise ValueError("Adaptive composition/model mismatch")
    for key in ("steps", "regrids", "mesh_revision"):
        value = getattr(session, key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("Invalid adaptive counters")
    if session.mesh_revision > session.regrids or not isinstance(
        session.budget_limited, bool
    ):
        raise ValueError("Invalid adaptive regrid metadata")
    if (
        not isinstance(session.run_id, str)
        or not session.run_id
        or session.initial_integral[0] <= 0
    ):
        raise ValueError("Invalid adaptive run identity")
    if session.lam is not None:
        session._fraction(session.lam, "lambda")
    if session.beta is not None:
        session._fraction(session.beta, "beta")
    for value in (
        session.initial_integral,
        session.boundary_integral,
        session.source_integral,
    ):
        if (
            value.shape != (session.state.shape[0],)
            or not np.isfinite(value).all()
        ):
            raise ValueError("Invalid adaptive conservation balances")
    return dict(
        format="TORCH checkpoint",
        schema_version=3,
        solver_version="2.0",
        geometry=session.config["geometry"],
        units="SI",
        layout="leaf-level-z-y-x",
        config=session.config,
        run_id=session.run_id,
        time=session.time,
        steps=session.steps,
        initial_integral=session.initial_integral,
        boundary_integral=session.boundary_integral,
        source_integral=session.source_integral,
        current_integral=session.integral,
        cell_count=len(session.mesh.keys),
        regrids=session.regrids,
        mesh_revision=session.mesh_revision,
        budget_limited=bool(session.budget_limited),
    )


def hdf_content(handle, session, visualization):
    from .torchIO import _json

    handle.create_dataset(
        "metadata",
        data=np.frombuffer(_json(metadata(session)).encode(), dtype=np.uint8),
    )
    for name, row in zip(component_names(session.config), session.state):
        handle.create_dataset(
            "state/" + name, data=row.astype("<f8"), fletcher32=True
        )
    for name, value in (("lambda", session.lam), ("beta", session.beta)):
        if value is not None:
            handle.create_dataset(
                "state/" + name, data=value.astype("<f8"), fletcher32=True
            )
    handle.create_dataset(
        "mesh/levels", data=session.mesh.levels.astype("<i8"), fletcher32=True
    )
    handle.create_dataset(
        "mesh/indices",
        data=session.mesh.keys[:, 1:].ravel().astype("<i8"),
        fletcher32=True,
    )
    if visualization:
        points, cells = mesh_data(session)
        handle.create_dataset("mesh/points", data=points.astype("<f8"))
        handle.create_dataset("mesh/cells", data=cells)
        for name, value in fields(session).items():
            handle.create_dataset("fields/" + name, data=value.astype("<f8"))


def read(handle, meta):
    required = {
        "cell_count",
        "regrids",
        "mesh_revision",
        "budget_limited",
        "config",
        "time",
        "steps",
        "run_id",
        "initial_integral",
        "boundary_integral",
        "source_integral",
        "current_integral",
    }
    if not required <= set(meta) or any(
        meta.get(k) != v
        for k, v in dict(
            format="TORCH checkpoint",
            solver_version="2.0",
            units="SI",
            layout="leaf-level-z-y-x",
        ).items()
    ):
        raise ValueError("Unsupported adaptive checkpoint metadata")
    result = AdaptiveSimulation.__new__(AdaptiveSimulation)
    result.config = c = meta["config"]
    if (
        meta.get("geometry") != c.get("geometry")
        or "amr" not in c
        or "face_boundaries" not in c
    ):
        raise ValueError("Adaptive geometry/config mismatch")
    c["amr"] = settings(**c["amr"])
    for key in ("counts_xyz", "lengths_xyz", "origins_xyz", "boundaries_xyz"):
        c[key] = list(np.atleast_1d(c[key]))
    result._validate_config()
    n, d = meta["cell_count"], c["dimension"]
    if (
        not isinstance(n, int)
        or isinstance(n, bool)
        or not 0 < n <= c["amr"]["max_cells"]
    ):
        raise ValueError("Invalid adaptive cell count")

    def array(path, length, kind, bytesize=8):
        data = handle[path]
        if (
            data.shape != (length,)
            or data.dtype.kind != kind
            or data.dtype.itemsize != bytesize
        ):
            raise ValueError("Invalid adaptive dataset layout or precision")
        value = np.asarray(data)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite adaptive dataset")
        return value

    levels = array("mesh/levels", n, "i")
    indices = array("mesh/indices", n * d, "i").reshape(n, d)
    keys = np.column_stack((levels, indices))
    result.mesh = LeafMesh(c, keys)
    if not np.array_equal(result.mesh.keys, keys) or np.any(
        levels < c["amr"]["min_level"]
    ):
        raise ValueError("Noncanonical or undersized adaptive leaves")
    names = component_names(c)
    expected = (
        set(names)
        | ({"lambda"} if c["reactive"] else set())
        | (
            {"beta"}
            if c["reactive"] and c["kinetic_model"] == "znd"
            else set()
        )
    )
    if set(handle["state"]) != expected:
        raise ValueError("Unexpected adaptive state fields")
    result.state = np.stack([array("state/" + name, n, "f") for name in names])
    result.lam = array("state/lambda", n, "f") if c["reactive"] else None
    result.beta = array("state/beta", n, "f") if "beta" in expected else None
    for key in ("steps", "regrids", "mesh_revision"):
        value = meta[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("Invalid adaptive step/regrid counters")
        setattr(result, key, value)
    result.time, result.run_id = meta["time"], meta["run_id"]
    result.budget_limited = meta["budget_limited"]
    if (
        not isinstance(result.budget_limited, bool)
        or result.mesh_revision > result.regrids
    ):
        raise ValueError("Invalid adaptive regrid metadata")
    for key in ("initial_integral", "boundary_integral", "source_integral"):
        value = np.asarray(meta[key], dtype=float)
        if value.shape != (d + 2,) or not np.isfinite(value).all():
            raise ValueError("Invalid adaptive balances")
        setattr(result, key, value)
    if (
        not isinstance(result.run_id, str)
        or not result.run_id
        or result.initial_integral[0] <= 0
    ):
        raise ValueError("Invalid adaptive run identity")
    metadata(result)
    current = np.asarray(meta["current_integral"], dtype=float)
    if (
        current.shape != (d + 2,)
        or not np.isfinite(current).all()
        or not np.allclose(current, result.integral, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("Adaptive state/integral mismatch")
    return result


def write_vtu(path, session):
    from .torchIO import _xml

    if Path(path).suffix.lower() != ".vtu":
        raise ValueError("Adaptive VTK requires .vtu")
    points, cells = mesh_data(session)
    root = ET.Element(
        "VTKFile",
        type="UnstructuredGrid",
        version="1.0",
        byte_order="LittleEndian",
    )
    grid = ET.SubElement(root, "UnstructuredGrid")
    data = ET.SubElement(grid, "FieldData")
    ET.SubElement(
        data,
        "DataArray",
        type="Float64",
        Name="TimeValue",
        NumberOfTuples="1",
        format="ascii",
    ).text = format(session.time, ".17g")
    piece = ET.SubElement(
        grid,
        "Piece",
        NumberOfPoints=str(len(points)),
        NumberOfCells=str(len(cells)),
    )
    field = ET.SubElement(piece, "CellData", Scalars="rho")
    for name, value in fields(session).items():
        ET.SubElement(
            field, "DataArray", type="Float64", Name=name, format="ascii"
        ).text = " ".join(format(v, ".17g") for v in value)
    node = ET.SubElement(piece, "Points")
    ET.SubElement(
        node,
        "DataArray",
        type="Float64",
        NumberOfComponents="3",
        format="ascii",
    ).text = " ".join(format(v, ".17g") for v in points.ravel())
    node = ET.SubElement(piece, "Cells")
    for name, value, dtype in (
        ("connectivity", cells.ravel(), "Int64"),
        ("offsets", np.arange(1, len(cells) + 1) * cells.shape[1], "Int64"),
        (
            "types",
            np.full(
                len(cells), {1: 3, 2: 9, 3: 12}[session.config["dimension"]]
            ),
            "UInt8",
        ),
    ):
        ET.SubElement(
            node, "DataArray", type=dtype, Name=name, format="ascii"
        ).text = " ".join(str(int(v)) for v in value)
    _xml(path, root)
    return str(path)
