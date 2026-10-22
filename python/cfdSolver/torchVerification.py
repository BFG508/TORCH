"""Analytic Euler references independent of the numerical flux kernels.

Vortex: Spiegel, Huynh and DeBonis, NASA document 20150018403.
Normal shock: NASA Glenn normal-shock relations (NACA Report 1135).
Acoustics: a right-running mode of linearized ideal-gas Euler.
"""

from itertools import product
import platform

import numpy as np


def conservative_average(primitive, gamma=1.4):
    """Convert reference point primitives, without calling a solver."""
    rho = primitive[0]
    result = primitive.copy()
    result[1:-1] = rho * primitive[1:-1]
    result[-1] = primitive[-1] / (gamma - 1) + (
        0.5 * rho * np.sum(primitive[1:-1] ** 2, axis=0)
    )
    return result


def as_primitive(average, gamma=1.4):
    """Primitives corresponding to conservative cell averages."""
    result = average.copy()
    result[1:-1] /= average[0]
    result[-1] = (gamma - 1) * (
        average[-1] - 0.5 * np.sum(average[1:-1] ** 2, axis=0) / average[0]
    )
    return result


def vortex_average(n, time=0.0, dimension=2):
    """Gauss cell averages of a translating beta=5 isentropic vortex.

    The 10x10 domain is centered at zero; the 3D case is a four-cell
    extrusion with constant axial velocity. Two-point Gauss quadrature
    in each varying direction has fourth-order spatial accuracy.
    """
    if dimension not in (2, 3):
        raise ValueError("The vortex requires two or three dimensions")
    h = 10.0 / n
    x = -5 + (np.arange(n) + 0.5) * h
    y, x = np.meshgrid(x, x, indexing="ij")
    out = np.zeros((dimension + 2, n, n))
    for sx, sy in product((-1, 1), repeat=2):
        xx = (x + sx * h / np.sqrt(12) - time + 5) % 10 - 5
        yy = (y + sy * h / np.sqrt(12) - time + 5) % 10 - 5
        radius2 = xx * xx + yy * yy
        swirl = 5 / (2 * np.pi) * np.exp(0.5 * (1 - radius2))
        temperature = 1 - 0.4 * 25 / (8 * 1.4 * np.pi**2) * np.exp(1 - radius2)
        p = np.empty_like(out)
        p[0] = temperature ** (1 / 0.4)
        p[1], p[2] = 1 - yy * swirl, 1 + xx * swirl
        if dimension == 3:
            p[3] = 0.25
        p[-1] = p[0] ** 1.4
        out += 0.25 * conservative_average(p)
    if dimension == 3:
        out = np.repeat(out[:, None], 4, axis=1)
    return out


def acoustic_average(n, time=0.0, dimension=1, amplitude=1e-6):
    """Exact cell averages of a linear acoustic mode along the diagonal.

    rho0=p0=1, gamma=1.4, uniform velocity 0.2 in every direction.
    Nonlinear Euler departs from this reference at O(amplitude**2).
    """
    if dimension not in (1, 2, 3):
        raise ValueError("Dimension must be one, two or three")
    grids = np.meshgrid(
        *([(np.arange(n) + 0.5) / n] * dimension), indexing="ij"
    )
    phase = (
        2
        * np.pi
        * (sum(grids) - (0.2 * dimension + np.sqrt(1.4 * dimension)) * time)
    )
    wave = amplitude * np.sin(phase) * np.sinc(1 / n) ** dimension
    result = np.empty((dimension + 2, *([n] * dimension)))
    result[0] = 1 + wave
    sound_normal = np.sqrt(1.4 / dimension)
    for k in range(dimension):
        result[k + 1] = 0.2 + (0.2 + sound_normal) * wave
    result[-1] = (
        2.5
        + 0.5 * dimension * 0.2**2
        + (
            1.4 / 0.4
            + 0.5 * dimension * 0.2**2
            + dimension * 0.2 * sound_normal
        )
        * wave
    )
    return result


def normal_shock_states(mach=2.0, gamma=1.4):
    """Up/downstream primitives from the Rankine-Hugoniot relations."""
    if not np.isfinite([mach, gamma]).all() or mach <= 1 or gamma <= 1:
        raise ValueError("A normal shock requires Mach>1 and gamma>1")
    density_ratio = (gamma + 1) * mach**2 / ((gamma - 1) * mach**2 + 2)
    pressure_ratio = (2 * gamma * mach**2 - (gamma - 1)) / (gamma + 1)
    speed = mach * np.sqrt(gamma)
    return (
        np.array([1.0, speed, 1.0]),
        np.array([density_ratio, speed / density_ratio, pressure_ratio]),
    )


def make_session(kind, n, dimension, kernel):
    """Initialize verification using independent reference data."""
    from .torchSession import Simulation

    options = dict(t_end=0.05, boundaries=["Periodic"] * dimension)
    if kind == "vortex":
        state = vortex_average(n, dimension=dimension)
        lengths = [10.0, 10.0] + ([1.0] if dimension == 3 else [])
        options.update(
            t_end=0.5, origins=[-5.0, -5.0] + ([0.0] if dimension == 3 else [])
        )
    elif kind == "acoustic":
        state = acoustic_average(n, dimension=dimension)
        lengths = [1.0] * dimension
    elif kind == "shock":
        upstream, downstream = normal_shock_states()
        p = np.ones((dimension + 2, *([4] * (dimension - 1)), n))
        p[1:-1] = 0
        for row, component in ((0, 0), (1, 1), (-1, 2)):
            p[row] = np.where(
                np.arange(n) < n // 2,
                upstream[component],
                downstream[component],
            )
        state = conservative_average(p)
        lengths = [1.0] * dimension
        options["boundaries"] = ["Open"] + ["Periodic"] * (dimension - 1)
    else:
        raise ValueError("Unknown verification case")
    if kernel not in ("hllc", "metric"):
        raise ValueError("Choose hllc or metric")
    if kernel == "metric":
        faces = {
            a + edge: "periodic"
            for a in "xyz"[:dimension]
            for edge in ("_min", "_max")
        }
        if kind == "shock":
            inlet = [float(upstream[0]), float(upstream[1])]
            inlet += [0.0] * (dimension - 1) + [float(upstream[-1])]
            faces["x_min"] = dict(type="inflow", primitive=inlet)
            faces["x_max"] = dict(
                type="pressure_outlet", pressure=float(downstream[-1])
            )
        options["face_boundaries"] = faces
    return Simulation(as_primitive(state), lengths, **options)


def measure_case(kind, n, dimension, kernel):
    """Record analytic errors and conservation balance for one mesh."""
    session = make_session(kind, n, dimension, kernel)
    initial = session.state.copy()
    session.run()
    if kind == "vortex":
        exact = vortex_average(n, session.time, dimension)
    elif kind == "acoustic":
        exact = acoustic_average(n, session.time, dimension)
    else:
        exact = initial
    error = np.abs(session.state - exact)
    scale = np.maximum(np.abs(session.initial_integral), 1.0)
    record = dict(
        case=kind,
        dimension=dimension,
        kernel=kernel,
        counts=session.config["counts_xyz"],
        time=session.time,
        steps=session.steps,
        density_l1=float(np.mean(error[0])),
        conservative_l1=float(np.mean(error)),
        balance_linf=float(np.max(np.abs(session.balance_residual) / scale)),
    )
    if kind == "shock":
        rho = session.primitive[0].reshape(-1, n).mean(axis=0)
        x = (np.arange(n) + 0.5) / n
        mask = (x < 0.3) | (x > 0.7)
        record["plateau_linf"] = float(np.max(error[0, ..., mask]))
        record["shock_position_error"] = abs(
            float((np.argmax(np.abs(np.diff(rho))) + 1) / n) - 0.5
        )
    return record


def verification_report(quick=False):
    """Mesh studies for both kernels, with no timing criteria."""
    rows = []
    checks = []
    for kernel in ("hllc", "metric"):
        for kind, dimensions, meshes in (
            ("vortex", (2, 3), (16, 32) if quick else (24, 48, 96)),
            ("acoustic", (1, 2, 3), (8, 16) if quick else (12, 24, 48)),
            ("shock", (1, 2, 3), (40, 80) if quick else (80, 160, 320)),
        ):
            for dimension in dimensions:
                group = [
                    measure_case(kind, n, dimension, kernel) for n in meshes
                ]
                errors = [r["density_l1"] for r in group]
                floor = 5e-13
                orders = [
                    float(np.log2(a / b)) if min(a, b) > floor else None
                    for a, b in zip(errors[:-1], errors[1:])
                ]
                minimum_order = 0.6 if kind == "shock" else 1.0
                if not quick and kind == "acoustic" and dimension == 1:
                    minimum_order = 1.6
                ok = all(
                    b <= floor or (order is not None and order > minimum_order)
                    for b, order in zip(errors[1:], orders)
                )
                for row in group:
                    ok &= row["balance_linf"] < 2e-10
                    if kind == "shock":
                        ok &= row["plateau_linf"] < 0.03
                        ok &= row["shock_position_error"] < 2 / meshes[0]
                checks.append(
                    dict(
                        case=kind,
                        dimension=dimension,
                        kernel=kernel,
                        observed_density_orders=orders,
                        passed=bool(ok),
                    )
                )
                rows.extend(group)
    return dict(
        schema_version=1,
        suite="TORCH analytic verification",
        quick=quick,
        python=platform.python_version(),
        numpy=np.__version__,
        passed=all(check["passed"] for check in checks),
        checks=checks,
        results=rows,
    )
