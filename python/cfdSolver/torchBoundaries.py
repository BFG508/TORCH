"""Serializable, per-face Euler boundary data shared by Cartesian/metric
sessions.
"""

import numpy as np
from .torchSolver import gamma_calc

TYPES = (
    "outflow",
    "wall",
    "periodic",
    "inflow",
    "pressure_outlet",
    "stagnation_inlet",
)


def normalize_faces(c, specification=None, _base=False):
    dim = c["dimension"]
    if specification is None:
        result = []
        for d, name in enumerate(c["boundaries_xyz"]):
            kind = {
                "Open": "outflow",
                "Reflective": "wall",
                "Periodic": "periodic",
                "Inflow": "inflow",
            }[name]
            low = dict(type=kind)
            if kind == "inflow":
                low["primitive"] = list(c["inlet_primitive"])
            result.extend(
                [low, dict(type="outflow") if kind == "inflow" else low.copy()]
            )
    elif isinstance(specification, dict):
        expected = [a + side for a in "xyz"[:dim] for side in ("_min", "_max")]
        if set(specification) - set(expected):
            raise ValueError(
                "Face keys must use logical x_min,x_max,y_min,..."
            )
        result = normalize_faces(c, _base=True)
        for i, name in enumerate(expected):
            if name in specification:
                value = specification[name]
                result[i] = (
                    dict(type=value) if isinstance(value, str) else dict(value)
                )
    else:
        result = [dict(v) for v in specification]
    if _base:
        return result
    if len(result) != 2 * dim:
        raise ValueError("Require one boundary per face")
    for i, face in enumerate(result):
        kind = face.get("type")
        if kind not in TYPES:
            raise ValueError("Unknown face boundary type")
        allowed = {"type", "lambda", "beta"}
        allowed |= {"primitive"} if kind == "inflow" else set()
        allowed |= (
            {"pressure", "backflow"} if kind == "pressure_outlet" else set()
        )
        allowed |= (
            {"total_pressure", "total_temperature", "gas_constant"}
            if kind == "stagnation_inlet"
            else set()
        )
        if set(face) - allowed:
            raise ValueError("Unknown boundary parameters")
        for key in ("lambda", "beta"):
            value = face.get(key, 1.0)
            if (
                not np.isscalar(value)
                or not np.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Boundary fractions must lie in [0,1]")
        for key in ("primitive", "backflow"):
            if key in face:
                p = np.asarray(face[key], dtype=float)
                if (
                    p.shape != (dim + 2,)
                    or not np.isfinite(p).all()
                    or np.any(p[[0, -1]] <= 0)
                ):
                    raise ValueError("Invalid boundary primitive state")
                sign = -1 if i % 2 == 0 else 1
                if p[i // 2 + 1] * sign >= 0:
                    raise ValueError(
                        "Inlet/backflow velocity must point into the domain"
                    )
                face[key] = p.tolist()
        if kind == "inflow" and "primitive" not in face:
            raise ValueError("Inflow requires primitive state")
        for key in (
            {"pressure"}
            if kind == "pressure_outlet"
            else (
                {"total_pressure", "total_temperature", "gas_constant"}
                if kind == "stagnation_inlet"
                else set()
            )
        ):
            if key == "gas_constant":
                face.setdefault(key, 287.0)
            if (
                key not in face
                or not np.isscalar(face[key])
                or not np.isfinite(face[key])
                or face[key] <= 0
            ):
                raise ValueError(
                    (
                        "Boundary pressures, temperatures and gas constant "
                        "must be positive"
                    )
                )
    for d in range(dim):
        if (result[2 * d]["type"] == "periodic") != (
            result[2 * d + 1]["type"] == "periodic"
        ):
            raise ValueError("Periodic boundaries must be paired on an axis")
    g = c.get("geometry", "cartesian")
    origin = c.get("origins_xyz", [0.0] * dim)
    if g != "cartesian":
        if any(f["type"] == "periodic" for f in result[:2]):
            raise ValueError("Radius cannot be periodic")
        if origin[0] == 0 and result[0]["type"] != "wall":
            raise ValueError("The radial origin requires a reflective wall")
        if g == "spherical" and dim >= 2:
            for i, at_pole in (
                (2, origin[1] == 0),
                (
                    3,
                    np.isclose(
                        origin[1] + c["lengths_xyz"][1],
                        np.pi,
                        atol=1e-14,
                        rtol=0,
                    ),
                ),
            ):
                if at_pole and result[i]["type"] != "wall":
                    raise ValueError(
                        "A spherical pole requires a reflective wall"
                    )
            if any(f["type"] == "periodic" for f in result[2:4]):
                raise ValueError("Polar theta cannot be periodic")
        azimuth = (
            2
            if g == "spherical" and dim == 3
            else (
                1
                if (
                    g == "annular"
                    and dim >= 2
                    or g == "cylindrical"
                    and dim == 3
                )
                else None
            )
        )
        if (
            azimuth is not None
            and result[2 * azimuth]["type"] == "periodic"
            and not np.isclose(
                c["lengths_xyz"][azimuth], 2 * np.pi, atol=1e-14, rtol=0
            )
        ):
            raise ValueError(
                "Azimuthal periodicity requires a complete 2*pi domain"
            )
    return result


def ghost(primitive, gamma, fractions, face, direction, side, reactive):
    """Build a face state; side=-1 is low and side=+1 is high."""
    p = primitive.copy()
    g = np.array(gamma, copy=True)
    qs = [np.array(q, copy=True) for q in fractions]
    kind = face["type"]
    normal = direction + 1
    if kind == "wall":
        p[normal] *= -1
    elif kind in ("inflow", "stagnation_inlet"):
        g[...] = gamma_calc(face.get("lambda", 1.0)) if reactive else gamma
        for q, key in zip(qs, ("lambda", "beta")):
            q[...] = face.get(key, 1.0)
        if kind == "inflow":
            p[...] = np.asarray(face["primitive"]).reshape(
                (-1,) + (1,) * (p.ndim - 1)
            )
        else:
            sound = np.sqrt(gamma * primitive[-1] / primitive[0])
            invariant = -side * primitive[normal] - 2 * sound / (gamma - 1)
            a0sq = g * face["gas_constant"] * face["total_temperature"]
            coefficient = (g + 1) / (g - 1)
            discriminant = invariant**2 - coefficient * (
                0.5 * (g - 1) * invariant**2 - a0sq
            )
            if np.any(discriminant < 0):
                raise ValueError("No subsonic reservoir-inlet solution")
            a = (-invariant + np.sqrt(discriminant)) / coefficient
            speed = invariant + 2 * a / (g - 1)
            choked = speed >= a
            a = np.where(choked, np.sqrt(2 * a0sq / (g + 1)), a)
            speed = np.where(choked, a, speed)
            if np.any(speed < 0) or np.any(a <= 0):
                raise ValueError(
                    (
                        "Reservoir cannot supply inflow against the interior "
                        "pressure"
                    )
                )
            p[1:-1] = 0
            p[normal] = -side * speed
            p[-1] = face["total_pressure"] * (a * a / a0sq) ** (g / (g - 1))
            p[0] = g * p[-1] / (a * a)
    elif kind == "pressure_outlet":
        a = np.sqrt(g * p[-1] / p[0])
        un = side * p[normal]
        subsonic = un < a
        rho = p[0] * (face["pressure"] / p[-1]) ** (1 / g)
        anew = np.sqrt(g * face["pressure"] / rho)
        velocity = un + 2 * (a - anew) / (g - 1)
        backflow = subsonic & ((velocity < 0) | (un < 0))
        if np.any(backflow) and "backflow" not in face:
            raise ValueError(
                (
                    "Pressure outlet backflow requires an explicit backflow "
                    "primitive state"
                )
            )
        p[0] = np.where(subsonic, rho, p[0])
        p[-1] = np.where(subsonic, face["pressure"], p[-1])
        p[normal] = np.where(subsonic, side * velocity, p[normal])
        if np.any(backflow):
            prescribed = np.asarray(face["backflow"]).reshape(
                (-1,) + (1,) * (p.ndim - 1)
            )
            p = np.where(backflow[None], prescribed, p)
            if reactive:
                g = np.where(backflow, gamma_calc(face.get("lambda", 1.0)), g)
            for q, key in zip(qs, ("lambda", "beta")):
                q[...] = np.where(backflow, face.get(key, 1.0), q)
    return p, g, qs
