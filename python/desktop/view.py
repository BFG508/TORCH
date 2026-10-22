"""Cell fields, physical maps and interactive volume slices."""

from dataclasses import dataclass

import numpy as np
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from cfdSolver.torchGeometry import axes, coordinates, map_points


@dataclass
class Plane:
    values: np.ndarray
    vertices: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    profile_x: np.ndarray
    profile: np.ndarray
    x_label: str
    y_label: str
    field_label: str
    slice_label: str


def fields(session):
    names = ["Pressure", "Density", "Speed"]
    names += ["Velocity " + a for a in coordinates(session.config)]
    if session.lam is not None:
        names.append("Reactant fraction")
    if session.beta is not None:
        names.append("Induction fraction")
    return names


def field_values(session, name):
    p = session.primitive
    if name == "Pressure":
        return p[-1], "Pressure [Pa]"
    if name == "Density":
        return p[0], "Density [kg/m³]"
    if name == "Speed":
        return np.sqrt(np.sum(p[1:-1] ** 2, axis=0)), "Speed [m/s]"
    if name.startswith("Velocity "):
        direction = coordinates(session.config).index(name[9:])
        return p[direction + 1], name + " [m/s]"
    if name == "Reactant fraction" and session.lam is not None:
        return session.lam, "Unburned fraction"
    if name == "Induction fraction" and session.beta is not None:
        return session.beta, "Induction fraction"
    raise ValueError("This field is unavailable in the current simulation")


def coordinate_label(name):
    return name + " [rad]" if name in ("theta", "phi") else name + " [m]"


def session_plane(session, name="Pressure", normal=2, index=None):
    c = session.config
    dim = c["dimension"]
    names = coordinates(c)
    edges = axes(c)
    centers = [(a[:-1] + a[1:]) / 2 for a in edges]
    values, field_label = field_values(session, name)
    if dim == 1:
        return Plane(
            values,
            np.empty((0, 3)),
            edges[0],
            np.zeros_like(edges[0]),
            centers[0],
            values,
            coordinate_label(names[0]),
            "",
            field_label,
            "Radial profile" if names[0] == "r" else "Profile",
        )
    remaining = list(range(dim))
    selected = "2D field"
    if dim == 3:
        if normal not in (0, 1, 2):
            raise ValueError("Select a valid slice axis")
        count = c["counts_xyz"][normal]
        index = count // 2 if index is None else index
        if not isinstance(index, (int, np.integer)) or not 0 <= index < count:
            raise ValueError("Slice index is outside the mesh")
        values = np.take(values, index, axis=2 - normal)
        remaining.remove(normal)
        selected = f"{names[normal]} = {centers[normal][index]:.6g} " + (
            "rad" if names[normal] in ("theta", "phi") else "m"
        )
    horizontal, vertical = remaining
    xx, yy = np.meshgrid(edges[horizontal], edges[vertical])
    logical = np.zeros((*xx.shape, 3))
    logical[..., horizontal] = xx
    logical[..., vertical] = yy
    if dim == 3:
        logical[..., normal] = centers[normal][index]
    points = map_points(c, logical.reshape(-1, 3)).reshape(logical.shape)
    map_x, map_y = xx, yy
    x_label = coordinate_label(names[horizontal])
    y_label = coordinate_label(names[vertical])
    if dim == 2 and c.get("geometry", "cartesian") != "cartesian":
        pair = (
            (0, 2) if c["geometry"] in ("cylindrical", "spherical") else (0, 1)
        )
        map_x, map_y = points[..., pair[0]], points[..., pair[1]]
        x_label, y_label = "xyz"[pair[0]] + " [m]", "xyz"[pair[1]] + " [m]"
    middle = values.shape[0] // 2
    return Plane(
        values,
        points,
        map_x,
        map_y,
        centers[horizontal],
        values[middle],
        x_label,
        y_label,
        field_label,
        selected,
    )


def draw_session(
    figure, session, name="Pressure", normal=2, index=None, mode="Map"
):
    """Replace Figure artists without changing the global backend."""
    plane = session_plane(session, name, normal, index)
    figure.clear()
    dim = session.config["dimension"]
    if dim == 1:
        axis = figure.add_subplot(111)
        axis.plot(plane.profile_x, plane.profile)
        axis.set(xlabel=plane.x_label, ylabel=plane.field_label)
        axis.grid(True)
    else:
        lower = float(np.min(plane.values))
        upper = float(np.max(plane.values))
        if lower == upper:
            delta = max(1.0, abs(lower)) * 1e-9
            lower, upper = lower - delta, upper + delta
        norm = Normalize(lower, upper)
        if mode == "3D plane":
            axis = figure.add_subplot(211, projection="3d")
            p = plane.vertices
            axis.plot_surface(
                p[..., 0],
                p[..., 1],
                p[..., 2],
                facecolors=cm.viridis(norm(plane.values)),
                rstride=1,
                cstride=1,
                shade=False,
            )
            axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
            mappable = ScalarMappable(norm=norm, cmap="viridis")
        else:
            axis = figure.add_subplot(211)
            mappable = axis.pcolormesh(
                plane.map_x,
                plane.map_y,
                plane.values,
                shading="flat",
                cmap="viridis",
                norm=norm,
            )
            axis.set(xlabel=plane.x_label, ylabel=plane.y_label)
            if dim == 2:
                axis.set_aspect("equal", adjustable="box")
        figure.colorbar(mappable, ax=axis, label=plane.field_label, pad=0.08)
        profile = figure.add_subplot(212)
        profile.plot(plane.profile_x, plane.profile)
        profile.set(
            xlabel=coordinate_label(
                coordinates(session.config)[
                    next(d for d in range(dim) if dim == 2 or d != normal)
                ]
            ),
            ylabel=plane.field_label,
            title="Slice centreline",
        )
        profile.grid(True)
    figure.suptitle(
        f"{plane.slice_label} | t={session.time:.6g} s | step {session.steps}"
    )
    figure.tight_layout(pad=1.8)
    return plane
