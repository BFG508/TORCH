"""Serializable inputs for desktop sessions; no GUI dependency."""

from pathlib import Path
import json
import uuid

import numpy as np

from cfdSolver import Simulation
from cfdSolver.torchBoundaries import normalize_faces
from cfdSolver.torchGeometry import coordinates
from cfdSolver.torchSolver2D import get_2d_preset
from cfdSolver.torchSolver3D import PRESETS_3D

PRESETS_2D = (
    "2d_explosion",
    "2d_implosion",
    "2d_riemann_quadrant",
    "2d_shock_tube",
    "2d_rde_chamber",
)


def presets(dimension, geometry):
    """Offer presets appropriate to the selected grid."""
    if geometry == "cartesian":
        return ("Uniform",) + (PRESETS_2D if dimension == 2 else PRESETS_3D)
    choices = ("Uniform", "Radial blast")
    if geometry == "annular" and dimension >= 2 or dimension == 3:
        choices += ("Rotating equilibrium",)
    return choices


def defaults(dimension=2, geometry="cartesian"):
    """Return interactive mesh sizes and valid coordinate domains."""
    lengths = [1.0] * dimension
    origins = [0.0] * dimension
    if geometry == "annular" or dimension == 3 and geometry != "cartesian":
        lengths[0], origins[0] = 0.5, 0.5
    if (
        geometry == "annular"
        and dimension >= 2
        or (geometry == "cylindrical" and dimension == 3)
    ):
        lengths[1] = 2 * np.pi
    if geometry == "spherical" and dimension >= 2:
        lengths[1] = np.pi if dimension == 2 else np.pi - 0.4
        origins[1] = 0.0 if dimension == 2 else 0.2
        if dimension == 3:
            lengths[2] = 2 * np.pi
    return dict(
        dimension=dimension,
        geometry=geometry,
        counts=[40, 32] if dimension == 2 else [20, 16, 12],
        lengths=lengths,
        origins=origins,
        preset=(
            ("2d_explosion" if dimension == 2 else "3d_explosion")
            if geometry == "cartesian"
            else "Uniform"
        ),
        t_end=0.05 if geometry == "cartesian" else 1e-4,
        cfl=0.4,
        gamma=1.4,
        limiter="VanLeer",
        reactive=False,
        kinetic_model="arrhenius",
        density=1.0,
        pressure=101325.0,
        velocities=[0.0] * dimension,
        rotation=20.0,
        fraction=1.0,
        beta=1.0,
        face_edits={},
        workers=1,
        adaptive=False,
        amr={},
    )


def face_defaults(spec):
    """Translate preset and geometry conditions for the face editor."""
    dim, geometry = spec["dimension"], spec["geometry"]
    bc = ["Open"] * dim
    if geometry == "cartesian" and spec["preset"] in PRESETS_2D:
        p = get_2d_preset(spec["preset"], 4, 4)
        bc = [
            "Open" if p[k] == "Transmissive" else p[k]
            for k in ("bc_x", "bc_y")
        ]
    c = dict(
        dimension=dim,
        boundaries_xyz=bc,
        inlet_primitive=[1.0, 0.0, 100.0, 101325.0],
    )
    result = normalize_faces(c)
    if geometry != "cartesian":
        origins, lengths = spec["origins"], spec["lengths"]
        if origins[0] == 0:
            result[0] = dict(type="wall")
        if geometry == "spherical" and dim >= 2:
            if origins[1] == 0:
                result[2] = dict(type="wall")
            if np.isclose(origins[1] + lengths[1], np.pi):
                result[3] = dict(type="wall")
        azimuth = (
            2
            if geometry == "spherical" and dim == 3
            else (
                1
                if geometry == "annular"
                and dim >= 2
                or (geometry == "cylindrical" and dim == 3)
                else None
            )
        )
        if azimuth is not None and np.isclose(lengths[azimuth], 2 * np.pi):
            result[2 * azimuth: 2 * azimuth + 2] = [dict(type="periodic")] * 2
    edits = spec.get("face_edits", {})
    for i, name in enumerate(face_keys(dim)):
        if name in edits:
            result[i] = dict(edits[name])
    return result


def face_keys(dimension):
    return [a + side for a in "xyz"[:dimension] for side in ("_min", "_max")]


def create_session(spec):
    session = _create_uniform(spec)
    if spec.get("adaptive", False):
        adaptive = session.to_adaptive(**spec.get("amr", {}))
        session.close()
        return adaptive
    return session


def _create_uniform(spec):
    """Create a session from validated, widget-independent inputs."""
    dim = spec["dimension"]
    if dim not in (2, 3):
        raise ValueError("The session desktop creates 2D or 3D simulations")
    if spec["preset"] not in presets(dim, spec["geometry"]):
        raise ValueError("Preset does not match the geometry/dimension")
    counts = list(spec["counts"])
    if len(counts) != dim or any(
        not isinstance(n, int) or n < 2 for n in counts
    ):
        raise ValueError("Enter at least two integer cells per active axis")
    common = dict(
        t_end=spec["t_end"],
        cfl=spec["cfl"],
        gamma=spec["gamma"],
        limiter=spec["limiter"],
        reactive=spec["reactive"],
        kinetic_model=spec["kinetic_model"],
        geometry=spec["geometry"],
        origins=spec["origins"],
        workers=spec.get("workers", 1),
    )
    if spec["reactive"]:
        common["lambda_init"] = np.full(counts[::-1], spec["fraction"])
        if spec["kinetic_model"] == "znd":
            common["beta_init"] = np.full(counts[::-1], spec["beta"])
    if spec["geometry"] != "cartesian" or spec.get("face_edits"):
        common["face_boundaries"] = face_defaults(spec)
    name = spec["preset"]
    if name not in ("Uniform", "Radial blast", "Rotating equilibrium"):
        return Simulation.preset(
            name, counts, lengths=spec["lengths"], **common
        )
    if len(spec["velocities"]) != dim:
        raise ValueError("Enter one local velocity per active axis")
    p = np.empty((dim + 2, *counts[::-1]))
    p[0], p[-1] = spec["density"], spec["pressure"]
    for k, value in enumerate(spec["velocities"], 1):
        p[k] = value
    r = (
        spec["origins"][0]
        + (np.arange(counts[0]) + 0.5) * spec["lengths"][0] / counts[0]
    )
    if name == "Radial blast":
        p[-1] *= np.where(
            r < spec["origins"][0] + 0.25 * spec["lengths"][0], 10.0, 1.0
        )
    if name == "Rotating equilibrium":
        speed = spec["rotation"] * r
        p[1:-1] = 0
        direction = 2
        if spec["geometry"] == "spherical":
            theta = (
                spec["origins"][1]
                + (np.arange(counts[1]) + 0.5) * spec["lengths"][1] / counts[1]
            )
            speed = speed[None, :] * np.sin(theta[:, None])
            direction = 3
        p[direction] = speed
        p[-1] += 0.5 * p[0] * speed**2
    return Simulation(p, spec["lengths"], **common)


def new_prefix(folder):
    """Use a fresh collection for every new run or checkpoint branch."""
    if not str(folder).strip():
        raise ValueError("Choose a results folder")
    folder = Path(folder).expanduser()
    return folder / ("run_" + uuid.uuid4().hex[:12]) / "history"


def history_frames(prefix):
    data = json.loads(Path(str(prefix) + ".series.json").read_text())
    if data.get("schema_version") != 1 or "xdmf" not in data["formats"]:
        raise ValueError("Playback requires a series with HDF5/XDMF frames")
    frames = data["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("This series contains no frames")
    parent = Path(prefix).parent.resolve()
    previous_time = -1.0
    previous_step = -1
    for frame in frames:
        stem = frame["stem"]
        if Path(stem).name != stem:
            raise ValueError("Invalid series frame path")
        path = parent / (stem + ".h5")
        if not path.is_file():
            raise ValueError("A series frame is missing")
        if not np.isfinite(frame["time"]) or not isinstance(
            frame["step"], int
        ):
            raise ValueError("Invalid frame counters")
        if frame["time"] <= previous_time or frame["step"] <= previous_step:
            raise ValueError("Series frames are not ordered")
        previous_time, previous_step = frame["time"], frame["step"]
    return data


def config_spec(session):
    """Show saved mesh and physics without rebuilding initial data."""
    c = session.config
    dim = c["dimension"]
    result = defaults(max(2, dim), c.get("geometry", "cartesian"))
    result.update(
        dimension=dim,
        counts=c["counts_xyz"],
        lengths=c["lengths_xyz"],
        origins=c.get("origins_xyz", [0.0] * dim),
        t_end=c["t_end"],
        cfl=c["cfl"],
        gamma=c["gamma"],
        limiter=c["limiter"],
        reactive=c["reactive"],
        kinetic_model=c["kinetic_model"],
        preset="Uniform",
        workers=session.workers,
        adaptive=hasattr(session, "mesh"),
        amr=c.get("amr", {}),
    )
    saved_faces = c.get("face_boundaries") or normalize_faces(c)
    result["face_edits"] = dict(zip(face_keys(dim), saved_faces))
    result["coordinates"] = coordinates(c)
    return result
