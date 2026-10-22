"""Versioned, non-executable HDF5 checkpoints and streaming VTK/XDMF
outputs.

Schema 2 (and legacy schema 1) stores each conservative component as a
flat float64
vector with x varying fastest. Metadata is a UTF-8 JSON byte vector,
avoiding
MATLAB/HDF5 dimension reversal and implementation-specific object
serialization.
"""

from .torchGeometry import (
    component_names,
    axes as grid_axes,
    map_points,
    cartesian_velocity,
    coordinates,
)
from pathlib import Path
import json
import os
import tempfile
from xml.etree import ElementTree as ET
import h5py
import numpy as np

SCHEMA_VERSION = 2
FIELD_NAMES = {
    1: ["rho", "momentum_x", "energy"],
    2: ["rho", "momentum_x", "momentum_y", "energy"],
    3: ["rho", "momentum_x", "momentum_y", "momentum_z", "energy"],
}


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _json(value):
    return json.dumps(
        value, default=_plain, allow_nan=False, separators=(",", ":")
    )


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        write(Path(temp))
        with open(temp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def metadata(session):
    from .torchSolver import physical_state

    session._validate_config()
    dim = session.config["dimension"]
    if session.state.shape != (
        dim + 2,
        *session.config["counts_xyz"][::-1],
    ) or not physical_state(session.state):
        raise ValueError("Cannot save a nonphysical or mismatched state")
    if (
        not np.isfinite(session.time)
        or not 0 <= session.time <= session.config["t_end"]
        or not isinstance(session.steps, int)
        or not 0 <= session.steps <= session.config["max_steps"]
    ):
        raise ValueError("Cannot save invalid integration counters")
    if session.config["reactive"]:
        if (
            session.lam is None
            or session.config["kinetic_model"] == "znd"
            and session.beta is None
        ):
            raise ValueError("Missing reactive composition")
        session._fraction(session.lam, "lambda")
        if session.config["kinetic_model"] == "znd":
            session._fraction(session.beta, "beta")
    for value in (
        session.initial_integral,
        session.boundary_integral,
        session.source_integral,
    ):
        if np.shape(value) != (dim + 2,) or not np.isfinite(value).all():
            raise ValueError("Cannot save invalid conservation balances")
    return dict(
        format="TORCH checkpoint",
        schema_version=SCHEMA_VERSION,
        solver_version="1.4",
        geometry=session.config.get("geometry", "cartesian"),
        units="SI",
        layout="x-fastest",
        config=session.config,
        run_id=session.run_id,
        time=session.time,
        steps=session.steps,
        initial_integral=session.initial_integral,
        boundary_integral=session.boundary_integral,
        source_integral=session.source_integral,
        current_integral=session.integral,
    )


def _mesh(session):
    c = session.config
    dim = c["dimension"]
    counts = c["counts_xyz"]
    axes = grid_axes(c)
    axes += [np.array([0.0])] * (3 - dim)
    zz, yy, xx = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    nx = counts[0]
    ny = counts[1] if dim >= 2 else 1
    nz = counts[2] if dim == 3 else 1
    k, j, i = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    base = (
        i + (nx + 1) * (j + (counts[1] + 1 if dim >= 2 else 1) * k)
    ).ravel()
    if dim == 1:
        offsets = [0, 1]
    elif dim == 2:
        offsets = [0, 1, nx + 2, nx + 1]
    else:
        plane = (nx + 1) * (ny + 1)
        offsets = [
            0,
            1,
            nx + 2,
            nx + 1,
            plane,
            plane + 1,
            plane + nx + 2,
            plane + nx + 1,
        ]
    cells = base[:, None] + offsets
    return axes, map_points(c, points), cells.astype(np.int64)


def fields(session):
    dim = session.config["dimension"]
    prim = session.primitive
    result = {
        name: row.ravel()
        for name, row in zip(component_names(session.config), session.state)
    }
    result["pressure"] = prim[-1].ravel()
    for direction in range(dim):
        result["velocity_" + coordinates(session.config)[direction]] = prim[
            direction + 1
        ].ravel()
    if session.config.get("geometry", "cartesian") != "cartesian":
        for name, value in zip(
            "xyz", cartesian_velocity(session.config, prim)
        ):
            result["velocity_cartesian_" + name] = value.ravel()
    if session.lam is not None:
        result["lambda"] = session.lam.ravel()
    if session.beta is not None:
        result["beta"] = session.beta.ravel()
    return result


def _hdf_content(handle, session, visualization=False):
    encoded = _json(metadata(session)).encode("utf-8")
    handle.create_dataset(
        "metadata", data=np.frombuffer(encoded, dtype=np.uint8)
    )
    dim = session.config["dimension"]
    for name, row in zip(component_names(session.config), session.state):
        handle.create_dataset(
            "state/" + name, data=row.ravel().astype("<f8"), fletcher32=True
        )
    for name, value in (("lambda", session.lam), ("beta", session.beta)):
        if value is not None:
            handle.create_dataset(
                "state/" + name,
                data=value.ravel().astype("<f8"),
                fletcher32=True,
            )
    if visualization:
        _, points, cells = _mesh(session)
        handle.create_dataset("mesh/points", data=points.astype("<f8"))
        handle.create_dataset("mesh/cells", data=cells)
        for name, value in fields(session).items():
            handle.create_dataset("fields/" + name, data=value.astype("<f8"))


def write_checkpoint(path, session, *, visualization=False):
    """Replace a checkpoint atomically on the same filesystem."""

    def write(temp):
        with h5py.File(temp, "w") as handle:
            _hdf_content(handle, session, visualization)
            handle.flush()

    _atomic(path, write)
    return str(path)


def read_checkpoint(path):
    """
    Validate schema, configuration and state before returning a session.
    """
    from .torchSession import Simulation
    from .torchSolver import physical_state

    try:
        with h5py.File(path, "r") as handle:
            raw = np.asarray(handle["metadata"])
            if (
                raw.ndim != 1
                or raw.dtype != np.dtype("uint8")
                or raw.size > 1_000_000
            ):
                raise ValueError("Invalid checkpoint metadata encoding")
            meta = json.loads(raw.tobytes().decode("utf-8"))
            required = {
                "config",
                "time",
                "steps",
                "run_id",
                "initial_integral",
                "boundary_integral",
                "source_integral",
                "current_integral",
            }
            if not isinstance(meta, dict) or not required.issubset(meta):
                raise ValueError("Incomplete checkpoint metadata")
            expected = dict(
                format="TORCH checkpoint",
                units="SI",
                layout="x-fastest",
            )
            if any(meta.get(k) != v for k, v in expected.items()):
                raise ValueError(
                    (
                        "Unsupported checkpoint schema, solver version, "
                        "geometry or units"
                    )
                )
            version = (meta.get("schema_version"), meta.get("solver_version"))
            if version not in ((1, "1.1"), (2, "1.2"), (2, "1.4")):
                raise ValueError("Unsupported checkpoint schema/solver family")
            if version[0] == 1 and (
                meta.get("geometry") != "cartesian"
                or any(
                    k in meta["config"]
                    for k in ("geometry", "origins_xyz", "face_boundaries")
                )
            ):
                raise ValueError(
                    "Schema 1 only supports legacy Cartesian configurations"
                )
            if version[0] == 2 and meta.get("geometry") != meta["config"].get(
                "geometry", "cartesian"
            ):
                raise ValueError("Checkpoint geometry/configuration mismatch")
            result = Simulation.__new__(Simulation)
            result.config = meta["config"]
            for key in (
                "counts_xyz",
                "lengths_xyz",
                "boundaries_xyz",
                "origins_xyz",
            ):
                if key not in result.config:
                    continue
                result.config[key] = list(np.atleast_1d(result.config[key]))
            result._validate_config()
            dim = result.config["dimension"]
            shape = tuple(result.config["counts_xyz"][::-1])
            size = int(np.prod(shape))

            def array(name):
                dataset = handle["state/" + name]
                if (
                    dataset.shape != (size,)
                    or dataset.dtype.kind != "f"
                    or dataset.dtype.itemsize != 8
                ):
                    raise ValueError(
                        (
                            "Checkpoint dataset has incompatible shape or "
                            "precision"
                        )
                    )
                return np.asarray(dataset, dtype=float).reshape(shape)

            result.state = np.stack(
                [array(name) for name in component_names(result.config)]
            )
            result.lam = array("lambda") if result.config["reactive"] else None
            result.beta = (
                array("beta")
                if result.config["reactive"]
                and result.config["kinetic_model"] == "znd"
                else None
            )
            expected_fields = (
                set(component_names(result.config))
                | ({"lambda"} if result.lam is not None else set())
                | ({"beta"} if result.beta is not None else set())
            )
            if set(handle["state"]) != expected_fields:
                raise ValueError("Unexpected or missing composition datasets")
    except (
        OSError,
        KeyError,
        TypeError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            f"Invalid or incomplete TORCH checkpoint: {exc}"
        ) from exc
    result.time = meta["time"]
    result.steps = meta["steps"]
    result.run_id = meta["run_id"]
    if (
        not isinstance(result.run_id, str)
        or not result.run_id
        or not isinstance(result.time, (int, float))
        or isinstance(result.time, bool)
        or not np.isfinite(result.time)
        or not 0 <= result.time <= result.config["t_end"]
    ):
        raise ValueError("Invalid checkpoint run identity or time")
    if (
        not isinstance(result.steps, int)
        or isinstance(result.steps, bool)
        or result.steps < 0
        or result.steps > result.config["max_steps"]
    ):
        raise ValueError("Invalid checkpoint step counter")
    for name in (
        "initial_integral",
        "boundary_integral",
        "source_integral",
        "current_integral",
    ):
        value = np.asarray(meta[name], dtype=float)
        if value.shape != (dim + 2,) or not np.isfinite(value).all():
            raise ValueError("Invalid integrated diagnostics")
        if name != "current_integral":
            setattr(result, name, value)
        elif not np.allclose(value, result.integral, rtol=1e-12, atol=1e-12):
            raise ValueError("Checkpoint state/integral mismatch")
    if not physical_state(result.state) or result.initial_integral[0] <= 0:
        raise ValueError("Nonphysical checkpoint state")
    if result.lam is not None:
        result._fraction(result.lam, "lambda")
    if result.beta is not None:
        result._fraction(result.beta, "beta")
    return result


def _xml(path, root):
    ET.indent(root)
    _atomic(
        path,
        lambda temp: ET.ElementTree(root).write(
            temp, encoding="utf-8", xml_declaration=True
        ),
    )


def write_vtk(path, session):
    """Write cell-centered VTK XML RectilinearGrid, including inactive
    axes.
    """
    curved = session.config.get("geometry", "cartesian") != "cartesian"
    if curved:
        return _write_structured_vtk(path, session)
    if Path(path).suffix.lower() != ".vtr":
        raise ValueError("VTK rectilinear output requires a .vtr path")
    c = session.config
    dim = c["dimension"]
    axes = grid_axes(c) + [np.array([0.0])] * (3 - dim)
    extent = " ".join(f"0 {n}" for n in c["counts_xyz"]) + " 0 0" * (3 - dim)
    root = ET.Element(
        "VTKFile",
        type="RectilinearGrid",
        version="1.0",
        byte_order="LittleEndian",
    )
    grid = ET.SubElement(root, "RectilinearGrid", WholeExtent=extent)
    piece = ET.SubElement(grid, "Piece", Extent=extent)
    cell = ET.SubElement(piece, "CellData", Scalars="rho")
    for name, value in fields(session).items():
        ET.SubElement(
            cell, "DataArray", type="Float64", Name=name, format="ascii"
        ).text = " ".join(format(v, ".17g") for v in value)
    data = ET.SubElement(grid, "FieldData")
    grid.remove(data)
    grid.insert(0, data)
    ET.SubElement(
        data,
        "DataArray",
        type="Float64",
        Name="TimeValue",
        NumberOfTuples="1",
        format="ascii",
    ).text = format(session.time, ".17g")
    coordinates = ET.SubElement(piece, "Coordinates")
    for axis, value in zip("xyz", axes):
        ET.SubElement(
            coordinates, "DataArray", type="Float64", Name=axis, format="ascii"
        ).text = " ".join(format(v, ".17g") for v in value)
    _xml(path, root)
    return str(path)


def _write_structured_vtk(path, session):
    if Path(path).suffix.lower() != ".vts":
        raise ValueError("Curved grids require .vts structured VTK output")
    c = session.config
    extent = " ".join(f"0 {n}" for n in c["counts_xyz"]) + " 0 0" * (
        3 - c["dimension"]
    )
    root = ET.Element(
        "VTKFile",
        type="StructuredGrid",
        version="1.0",
        byte_order="LittleEndian",
    )
    grid = ET.SubElement(root, "StructuredGrid", WholeExtent=extent)
    data = ET.SubElement(grid, "FieldData")
    ET.SubElement(
        data,
        "DataArray",
        type="Float64",
        Name="TimeValue",
        NumberOfTuples="1",
        format="ascii",
    ).text = format(session.time, ".17g")
    piece = ET.SubElement(grid, "Piece", Extent=extent)
    cell = ET.SubElement(piece, "CellData", Scalars="rho")
    for name, value in fields(session).items():
        ET.SubElement(
            cell, "DataArray", type="Float64", Name=name, format="ascii"
        ).text = " ".join(format(v, ".17g") for v in value)
    points = ET.SubElement(piece, "Points")
    ET.SubElement(
        points,
        "DataArray",
        type="Float64",
        NumberOfComponents="3",
        format="ascii",
    ).text = " ".join(format(v, ".17g") for v in _mesh(session)[1].ravel())
    _xml(path, root)
    return str(path)


def _xdmf_grid(parent, filename, time, dimension, counts, names):
    n = int(np.prod(counts))
    points = int(np.prod(np.array(counts) + 1))
    nodes = {1: 2, 2: 4, 3: 8}[dimension]
    grid = ET.SubElement(
        parent, "Grid", Name=Path(filename).stem, GridType="Uniform"
    )
    ET.SubElement(grid, "Time", Value=format(time, ".17g"))
    topology = ET.SubElement(
        grid,
        "Topology",
        TopologyType={1: "Polyline", 2: "Quadrilateral", 3: "Hexahedron"}[
            dimension
        ],
        NumberOfElements=str(n),
        NodesPerElement=str(nodes),
    )
    ET.SubElement(
        topology,
        "DataItem",
        Dimensions=f"{n} {nodes}",
        NumberType="Int",
        Precision="8",
        Format="HDF",
    ).text = (
        filename + ":/mesh/cells"
    )
    geometry = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
    ET.SubElement(
        geometry,
        "DataItem",
        Dimensions=f"{points} 3",
        NumberType="Float",
        Precision="8",
        Format="HDF",
    ).text = (
        filename + ":/mesh/points"
    )
    for name in names:
        attribute = ET.SubElement(
            grid, "Attribute", Name=name, AttributeType="Scalar", Center="Cell"
        )
        ET.SubElement(
            attribute,
            "DataItem",
            Dimensions=str(n),
            NumberType="Float",
            Precision="8",
            Format="HDF",
        ).text = (
            filename + ":/fields/" + name
        )
    return grid


def write_xdmf(path, session):
    """Write an XDMF/HDF5 snapshot that also serves as a checkpoint."""
    path = Path(path)
    if path.suffix.lower() not in (".xdmf", ".xmf"):
        raise ValueError("XDMF output requires .xdmf or .xmf")
    hdf = path.with_suffix(".h5")
    write_checkpoint(hdf, session, visualization=True)
    root = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(root, "Domain")
    grid = _xdmf_grid(
        domain,
        hdf.name,
        session.time,
        session.config["dimension"],
        session.config["counts_xyz"],
        fields(session),
    )
    grid.remove(
        grid.find("Time")
        # Static readers need no temporal element; time remains in HDF5
        # metadata.
    )
    _xml(path, root)
    return str(path), str(hdf)


class TemporalWriter:
    """Append immutable frames; manifest stores metadata, never solution
    arrays.
    """

    def __init__(self, prefix, session, formats=("vtk", "xdmf")):
        self.vtk_extension = (
            ".vts"
            if session.config.get("geometry", "cartesian") != "cartesian"
            else ".vtr"
        )
        self.prefix = Path(prefix)
        self.manifest = Path(str(prefix) + ".series.json")
        self.formats = tuple(formats)
        if not self.formats or set(self.formats) - {"vtk", "xdmf"}:
            raise ValueError("Formats must be vtk and/or xdmf")
        if self.manifest.exists():
            self.data = json.loads(self.manifest.read_text())
            if self.data["run_id"] != session.run_id or self.data[
                "formats"
            ] != list(self.formats):
                raise ValueError(
                    "Output prefix belongs to another run or format selection"
                )
            if self.data.get("schema_version") != 1:
                raise ValueError("Unsupported output manifest version")
            for key in (
                "counts_xyz",
                "lengths_xyz",
                "boundaries_xyz",
                "origins_xyz",
            ):
                if key not in self.data["config"]:
                    continue
                self.data["config"][key] = list(
                    np.atleast_1d(self.data["config"][key])
                )
            previous_time = -1.0
            previous_step = -1
            for frame in self.data["frames"]:
                if (
                    frame["time"] <= previous_time
                    or frame["step"] <= previous_step
                ):
                    raise ValueError("Output series has nonmonotone frames")
                previous_time, previous_step = frame["time"], frame["step"]
                for fmt in self.formats:
                    suffix = self.vtk_extension if fmt == "vtk" else ".h5"
                    if not (
                        self.prefix.parent / (frame["stem"] + suffix)
                    ).is_file():
                        raise ValueError(
                            "Output series references a missing frame"
                        )
            for key in session.config:
                if key not in ("t_end", "max_steps") and self.data["config"][
                    key
                ] != json.loads(_json(session.config[key])):
                    raise ValueError("Output series configuration mismatch")
            if (
                self.data["frames"]
                and self.data["frames"][-1]["time"] > session.time
            ):
                raise ValueError(
                    "Series is ahead of checkpoint; choose a new output prefix"
                )
        else:
            self.data = dict(
                schema_version=1,
                run_id=session.run_id,
                formats=list(self.formats),
                config=json.loads(_json(session.config)),
                frames=[],
            )

    def append(self, session):
        frames = self.data["frames"]
        if frames and frames[-1]["time"] == session.time:
            if frames[-1]["step"] != session.steps:
                raise ValueError("Duplicate series time at a different step")
            return
        stem = self.prefix.name + f"_{session.steps:08d}"
        parent = self.prefix.parent
        parent.mkdir(parents=True, exist_ok=True)
        if "vtk" in self.formats:
            write_vtk(parent / (stem + self.vtk_extension), session)
        if "xdmf" in self.formats:
            write_checkpoint(
                parent / (stem + ".h5"), session, visualization=True
            )
        frames.append(
            dict(
                time=session.time,
                step=session.steps,
                stem=stem,
                fields=list(fields(session)),
            )
        )
        if "vtk" in self.formats:
            root = ET.Element(
                "VTKFile",
                type="Collection",
                version="1.0",
                byte_order="LittleEndian",
            )
            collection = ET.SubElement(root, "Collection")
            for frame in frames:
                ET.SubElement(
                    collection,
                    "DataSet",
                    timestep=format(frame["time"], ".17g"),
                    group="",
                    part="0",
                    file=frame["stem"] + self.vtk_extension,
                )
            _xml(str(self.prefix) + ".pvd", root)
        if "xdmf" in self.formats:
            root = ET.Element("Xdmf", Version="3.0")
            domain = ET.SubElement(root, "Domain")
            collection = ET.SubElement(
                domain,
                "Grid",
                Name="TORCH",
                GridType="Collection",
                CollectionType="Temporal",
            )
            for frame in frames:
                _xdmf_grid(
                    collection,
                    frame["stem"] + ".h5",
                    frame["time"],
                    session.config["dimension"],
                    session.config["counts_xyz"],
                    frame["fields"],
                )
            _xml(str(self.prefix) + ".xdmf", root)
        _atomic(
            self.manifest,
            lambda temp: temp.write_text(_json(self.data), encoding="utf-8"),
        )
