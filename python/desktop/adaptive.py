"""Leaf polygon maps and slices without finest-grid arrays.
"""

import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize

from cfdSolver.torchGeometry import coordinates, map_points


def slice_count(session, normal):
    count = session.config["counts_xyz"][normal]
    if hasattr(session, "mesh"):
        count *= 2 ** session.config["amr"]["max_level"]
    return count


def plane(session, name, normal, index):
    from .view import Plane, coordinate_label, field_values

    c, mesh = session.config, session.mesh
    d, names = c["dimension"], coordinates(c)
    values, label = field_values(session, name)
    if d == 1:
        order = np.argsort(mesh.centers[:, 0])
        return Plane(
            values[order],
            np.empty((0, 3)),
            mesh.centers[order, 0],
            np.zeros(len(order)),
            mesh.centers[order, 0],
            values[order],
            coordinate_label(names[0]),
            "",
            label,
            "Adaptive profile",
        )
    remaining = list(range(d))
    mask = np.ones(len(values), dtype=bool)
    selected, position = "Adaptive 2D field", 0.0
    if d == 3:
        if normal not in (0, 1, 2):
            raise ValueError("Select a valid slice axis")
        count = slice_count(session, normal)
        index = count // 2 if index is None else index
        if not isinstance(index, (int, np.integer)) or not 0 <= index < count:
            raise ValueError("Slice index is outside the mesh")
        position = (
            c["origins_xyz"][normal]
            + (index + 0.5) * c["lengths_xyz"][normal] / count
        )
        mask = (mesh.low[:, normal] <= position) & (
            position < mesh.high[:, normal]
        )
        remaining.remove(normal)
        selected = f"Adaptive slice: {names[normal]} = {position:.6g}"
    h, v = remaining
    low, high = mesh.low[mask], mesh.high[mask]
    logical = np.zeros((len(low), 4, 3))
    logical[:, :, h] = np.column_stack(
        (low[:, h], high[:, h], high[:, h], low[:, h])
    )
    logical[:, :, v] = np.column_stack(
        (low[:, v], low[:, v], high[:, v], high[:, v])
    )
    if d == 3:
        logical[:, :, normal] = position
    points = map_points(c, logical.reshape(-1, 3)).reshape(-1, 4, 3)
    mx, my = logical[:, :, h], logical[:, :, v]
    xl, yl = coordinate_label(names[h]), coordinate_label(names[v])
    if d == 2 and c["geometry"] != "cartesian":
        pair = (
            (0, 2) if c["geometry"] in ("cylindrical", "spherical") else (0, 1)
        )
        mx, my = points[:, :, pair[0]], points[:, :, pair[1]]
        xl, yl = "xyz"[pair[0]] + " [m]", "xyz"[pair[1]] + " [m]"
    middle = c["origins_xyz"][v] + 0.5 * c["lengths_xyz"][v]
    profile = mask & (mesh.low[:, v] <= middle) & (middle < mesh.high[:, v])
    ids = np.flatnonzero(profile)
    ids = ids[np.argsort(mesh.centers[ids, h])]
    return Plane(
        values[mask],
        points,
        mx,
        my,
        mesh.centers[ids, h],
        values[ids],
        xl,
        yl,
        label,
        selected,
    )


def draw(figure, session, name, normal, index, mode):
    p = plane(session, name, normal, index)
    figure.clear()
    if session.config["dimension"] == 1:
        ax = figure.add_subplot(111)
        ax.plot(p.profile_x, p.profile, ".-")
        ax.set(xlabel=p.x_label, ylabel=p.field_label)
        ax.grid(True)
    else:
        lo, hi = float(np.min(p.values)), float(np.max(p.values))
        if lo == hi:
            delta = max(1.0, abs(lo)) * 1e-9
            lo, hi = lo - delta, hi + delta
        norm = Normalize(lo, hi)
        if mode == "3D plane":
            ax = figure.add_subplot(211, projection="3d")
            artist = Poly3DCollection(
                p.vertices,
                cmap="viridis",
                norm=norm,
                edgecolor=(0, 0, 0, 0.15),
                linewidth=0.2,
            )
            ax.add_collection3d(artist)
            pts = p.vertices.reshape(-1, 3)
            ax.auto_scale_xyz(pts[:, 0], pts[:, 1], pts[:, 2])
            span = np.ptp(pts, axis=0)
            ax.set_box_aspect(np.maximum(span, max(span) * 1e-6))
            ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        else:
            ax = figure.add_subplot(211)
            polys = np.stack((p.map_x, p.map_y), axis=-1)
            artist = PolyCollection(
                polys,
                cmap="viridis",
                norm=norm,
                edgecolor=(0, 0, 0, 0.15),
                linewidth=0.2,
            )
            ax.add_collection(artist)
            ax.autoscale_view()
            ax.set(xlabel=p.x_label, ylabel=p.y_label)
            if session.config["dimension"] == 2:
                ax.set_aspect("equal", adjustable="box")
        artist.set_array(p.values)
        figure.colorbar(artist, ax=ax, label=p.field_label, pad=0.08)
        ax = figure.add_subplot(212)
        ax.plot(p.profile_x, p.profile, ".-")
        names = coordinates(session.config)
        h = next(
            k for k in range(len(names)) if len(names) == 2 or k != normal
        )
        from .view import coordinate_label

        ax.set(
            xlabel=coordinate_label(names[h]),
            ylabel=p.field_label,
            title="Slice centreline (leaf centres)",
        )
        ax.grid(True)
    figure.suptitle(
        f"{p.slice_label} | t={session.time:.6g} s | "
        f"{len(session.mesh.keys)} leaves | step {session.steps}"
    )
    figure.tight_layout(pad=1.8)
    return p
