"""Uniform orthogonal grids with exact finite-volume metric factors.

Coordinate order: Cartesian x,y,z; cylindrical r,z (2D), r,theta,z (3D);
annular r,theta[,z]; spherical r,theta[,phi]. Angles are in radians.
"""

import numpy as np

GEOMETRIES = ("cartesian", "cylindrical", "annular", "spherical")


def coordinates(c):
    dim = c["dimension"]
    g = c.get("geometry", "cartesian")
    return (
        list("xyz"[:dim])
        if g == "cartesian"
        else (
            ["r", "theta", "phi"][:dim]
            if g == "spherical"
            else (
                ["r", "z"][:dim]
                if g == "cylindrical" and dim == 2
                else ["r", "theta", "z"][:dim]
            )
        )
    )


def component_names(c):
    return ["rho"] + ["momentum_" + a for a in coordinates(c)] + ["energy"]


def axes(c):
    return [
        np.linspace(o, o + l, n + 1)
        for o, l, n in zip(
            c.get("origins_xyz", [0.0] * c["dimension"]),
            c["lengths_xyz"],
            c["counts_xyz"],
        )
    ]


def validate_geometry(c):
    dim, g = c["dimension"], c.get("geometry", "cartesian")
    if g not in GEOMETRIES:
        raise ValueError("Unknown geometry")
    origin = np.asarray(c.get("origins_xyz", [0.0] * dim), dtype=float)
    if origin.shape != (dim,) or not np.isfinite(origin).all():
        raise ValueError("One finite origin per logical coordinate is required")
    if g == "cartesian":
        return
    if origin[0] < 0 or (g == "annular" and origin[0] <= 0):
        raise ValueError(
            "Radius must be nonnegative; an annulus needs a positive inner radius"
        )
    if g == "cylindrical" and dim == 3 and origin[0] == 0:
        raise ValueError("3D cylindrical grids require a positive inner radius")
    if g == "spherical" and dim >= 2:
        if origin[1] < 0 or origin[1] + c["lengths_xyz"][1] > np.pi + 1e-14:
            raise ValueError("Spherical theta must lie in [0, pi]")
        # A true 3D pole/origin needs connectivity across the coordinate singularity.
        if dim == 3 and (
            origin[0] == 0 or origin[1] <= 0 or origin[1] + c["lengths_xyz"][1] >= np.pi
        ):
            raise ValueError(
                "3D spherical grids require r>0 and 0<theta<pi (shell/wedge)"
            )
    angular = (
        [2]
        if g == "spherical" and dim == 3
        else (
            [1]
            if (g == "annular" and dim >= 2 or g == "cylindrical" and dim == 3)
            else []
        )
    )
    if any(c["lengths_xyz"][a] > 2 * np.pi + 1e-14 for a in angular):
        raise ValueError("Azimuthal extent cannot exceed 2*pi")
    if any(c["lengths_xyz"][a] / c["counts_xyz"][a] >= np.pi for a in angular):
        raise ValueError("Azimuthal cells must span less than pi radians")


class Metrics:
    """Cell volumes and face areas in reversed spatial order, matching states."""

    def __init__(self, c):
        validate_geometry(c)
        self.config = c
        dim, g = c["dimension"], c.get("geometry", "cartesian")
        self.axes = axes(c)
        widths = [np.diff(a) for a in self.axes]
        n = c["counts_xyz"]

        def product(factors):
            result = np.ones(tuple(len(a) for a in factors)[::-1])
            for direction, factor in enumerate(factors):
                shape = [1] * dim
                shape[dim - 1 - direction] = len(factor)
                result *= np.asarray(factor).reshape(shape)
            return result

        if g == "cartesian":
            volume_factors = widths
            face_factors = [
                [np.ones(n[a] + 1) if a == d else widths[a] for a in range(dim)]
                for d in range(dim)
            ]
        else:
            r = self.axes[0]
            dr, shell2 = np.diff(r), np.diff(r**2) / 2
            if g == "spherical":
                shell3 = np.diff(r**3) / 3
                if dim == 1:
                    volume_factors = [4 * np.pi * shell3]
                    face_factors = [[4 * np.pi * r**2]]
                else:
                    theta = self.axes[1]
                    solid = -np.diff(np.cos(theta))
                    azimuth = widths[2] if dim == 3 else np.array([2 * np.pi])
                    factor = 1 if dim == 3 else 2 * np.pi
                    volume_factors = [factor * shell3, solid] + (
                        [azimuth] if dim == 3 else []
                    )
                    face_factors = [
                        [factor * r**2, solid] + ([azimuth] if dim == 3 else []),
                        [factor * shell2, np.sin(theta)]
                        + ([azimuth] if dim == 3 else []),
                    ]
                    if dim == 3:
                        face_factors += [[shell2, widths[1], np.ones(n[2] + 1)]]
            elif dim == 1:
                volume_factors = [2 * np.pi * shell2]
                face_factors = [[2 * np.pi * r]]
            elif g == "cylindrical" and dim == 2:
                volume_factors = [2 * np.pi * shell2, widths[1]]
                face_factors = [
                    [2 * np.pi * r, widths[1]],
                    [2 * np.pi * shell2, np.ones(n[1] + 1)],
                ]
            else:
                volume_factors = [shell2, widths[1]] + widths[2:]
                face_factors = [
                    [r, widths[1]] + widths[2:],
                    [dr, np.ones(n[1] + 1)] + widths[2:],
                ]
                if dim == 3:
                    face_factors += [[shell2, widths[1], np.ones(n[2] + 1)]]
        self.volumes = product(volume_factors)
        self.areas = [product(f) for f in face_factors]
        self.divergence = [
            np.diff(a, axis=dim - 1 - d) / self.volumes
            for d, a in enumerate(self.areas)
        ]
        if not np.isfinite(self.volumes).all() or np.any(self.volumes <= 0):
            raise ValueError("Invalid cell volumes")

    def source(self, primitive):
        c = self.config
        dim, g = c["dimension"], c.get("geometry", "cartesian")
        out = np.zeros_like(primitive)
        if g == "cartesian":
            return out
        rho, p = primitive[0], primitive[-1]
        radial = self.divergence[0]
        out[1] = p * radial
        if g == "spherical":
            invr = radial / 2
            out[1] += rho * np.sum(primitive[2:-1] ** 2, axis=0) * invr
            if dim >= 2:
                cotr = self.divergence[1]
                out[2] = p * cotr - rho * primitive[1] * primitive[2] * invr
                if dim == 3:
                    out[2] += rho * primitive[3] ** 2 * cotr
                    out[3] = (
                        -rho
                        * primitive[3]
                        * (primitive[1] * invr + primitive[2] * cotr)
                    )
        elif dim >= 2 and not (g == "cylindrical" and dim == 2):
            out[1] += rho * primitive[2] ** 2 * radial
            out[2] = -rho * primitive[1] * primitive[2] * radial
        return out


def map_points(c, points):
    """Map logical coordinate triples to physical Cartesian positions."""
    g, dim = c.get("geometry", "cartesian"), c["dimension"]
    out = points.copy()
    if g == "cartesian" or dim == 1:
        return out
    r, theta = points[:, 0], points[:, 1]
    if g == "spherical":
        phi = points[:, 2] if dim == 3 else np.zeros(len(points))
        out[:, 0] = r * np.sin(theta) * np.cos(phi)
        out[:, 1] = r * np.sin(theta) * np.sin(phi)
        out[:, 2] = r * np.cos(theta)
    elif g == "cylindrical" and dim == 2:
        out[:, 0], out[:, 1], out[:, 2] = r, 0.0, theta
    else:
        out[:, 0], out[:, 1] = r * np.cos(theta), r * np.sin(theta)
    return out


def cartesian_velocity(c, primitive):
    dim, g = c["dimension"], c.get("geometry", "cartesian")
    local = [primitive[k + 1] for k in range(dim)] + [np.zeros_like(primitive[0])] * (
        3 - dim
    )
    if g == "cartesian" or dim == 1:
        return local
    centers = [(a[:-1] + a[1:]) / 2 for a in axes(c)]
    grids = np.meshgrid(*centers[::-1], indexing="ij")[::-1]
    if g == "cylindrical" and dim == 2:
        return [local[0], np.zeros_like(local[0]), local[1]]
    theta = grids[1]
    if g == "spherical":
        phi = grids[2] if dim == 3 else 0.0
        vr, vt, vp = local
        return [
            (vr * np.sin(theta) + vt * np.cos(theta)) * np.cos(phi) - vp * np.sin(phi),
            (vr * np.sin(theta) + vt * np.cos(theta)) * np.sin(phi) + vp * np.cos(phi),
            vr * np.cos(theta) - vt * np.sin(theta),
        ]
    vr, vt, vz = local
    return [
        vr * np.cos(theta) - vt * np.sin(theta),
        vr * np.sin(theta) + vt * np.cos(theta),
        vz,
    ]
