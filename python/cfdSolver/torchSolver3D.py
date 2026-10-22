"""Cartesian 3D Euler/reactive finite-volume solver.

States use (5,nz,ny,nx): rho, rho*u, rho*v, rho*w, E. Only the final
state is retained. Chemistry and HLLC are shared with the lower dimensions.
"""

import numpy as np
from .torchFlux import hllc_flux
from .torchSolver import gamma_calc, new_lambda_n_e_calc, physical_state, validate_grid

PRESETS_3D = ("3d_sod_x", "3d_sod_y", "3d_sod_z", "3d_explosion", "3d_ignition")
LIMITERS = ("VanLeer", "Albada", "MINBEE", "Superbee", "Unos")


def uprim_to_uconsv_3d(primitive, gamma=1.4):
    primitive = np.asarray(primitive, dtype=float)
    state = primitive.copy()
    state[1:4] *= primitive[0]
    state[4] = primitive[4] / (gamma - 1) + 0.5 * primitive[0] * np.sum(
        primitive[1:4] ** 2, axis=0
    )
    return state


def uconsv_to_uprim_3d(state, gamma=1.4):
    state = np.asarray(state, dtype=float)
    primitive = state.copy()
    primitive[1:4] /= state[0]
    primitive[4] = (gamma - 1) * (
        state[4] - 0.5 * np.sum(state[1:4] ** 2, axis=0) / state[0]
    )
    return primitive


def _pad(values, boundaries, vector=False):
    """Two ghost layers; scalar reflection never changes a sign."""
    out = values
    for spatial_axis, boundary in enumerate(boundaries):  # z,y,x
        axis = spatial_axis + int(vector)
        mode = boundary.lower()
        if mode in ("periodic", "wrap"):
            padding = "wrap"
        elif mode in ("open", "transmissive", "outflow"):
            padding = "edge"
        elif mode in ("reflective", "wall", "slip"):
            padding = "symmetric"
        else:
            raise ValueError(f"Unknown 3D boundary: {boundary}")
        widths = [(0, 0)] * out.ndim
        widths[axis] = (2, 2)
        out = np.pad(out, widths, mode=padding)
        if vector and padding == "symmetric":
            for edge in (slice(0, 2), slice(-2, None)):
                index = [slice(None)] * out.ndim
                index[0] = 3 - spatial_axis
                index[axis] = edge
                out[tuple(index)] *= -1
    return out


def apply_boundary_conditions_3d(state, bc_x="Open", bc_y="Open", bc_z="Open"):
    return _pad(state, (bc_z, bc_y, bc_x), vector=True)


def _flux(state, gamma, momentum):
    prim = uconsv_to_uprim_3d(state, gamma)
    velocity, pressure = prim[momentum], prim[4]
    flux = state * velocity
    flux[momentum] += pressure
    flux[4] = velocity * (state[4] + pressure)
    return flux


def _slope(padded, axis, limiter):
    active = (slice(None),) + (slice(1, -1),) * 3
    minus = (padded - np.roll(padded, 1, axis=axis))[active]
    plus = (np.roll(padded, -1, axis=axis) - padded)[active]
    r = minus / np.where(abs(plus) < 1e-14, 1e-14, plus)
    positive = np.maximum(r, 0)
    if limiter == "VanLeer":
        phi = 2 * positive / (1 + positive)
    elif limiter == "Albada":
        phi = (positive**2 + positive) / (positive**2 + 1)
    elif limiter == "MINBEE":
        phi = np.maximum(0, np.minimum(r, 1))
    elif limiter == "Superbee":
        phi = np.maximum(0, np.maximum(np.minimum(2 * r, 1), np.minimum(r, 2)))
    elif limiter == "Unos":
        # Bound both biased MUSCL slopes to preserve reflection symmetry in 3D.
        phi = np.maximum(
            0,
            np.minimum(np.minimum(np.minimum(2, 2 * r), (2 + r) / 3), (1 + 2 * r) / 3),
        )
    else:
        raise ValueError(f"Unknown limiter: {limiter}")
    return phi * plus


def _faces(array, axis, active):
    left = [slice(None)] + [slice(1, -1) if active else slice(2, -2)] * 3
    right = left.copy()
    left[axis] = slice(None, -1) if active else slice(1, -2)
    right[axis] = slice(1, None) if active else slice(2, -1)
    return array[tuple(left)], array[tuple(right)]


def _hydro(state, gamma, spacing, dt, limiter, boundaries, first_order=False):
    padded = _pad(state, boundaries, True)
    gp = _pad(np.broadcast_to(gamma, state.shape[1:]), boundaries)
    active = padded[:, 1:-1, 1:-1, 1:-1]
    ga = gp[1:-1, 1:-1, 1:-1]
    axes = (3, 2, 1)  # x,y,z
    slopes = [_slope(padded, a, limiter) for a in axes] if not first_order else []
    if not first_order:
        predicted = active.copy()
        for momentum, (slope, h) in enumerate(zip(slopes, spacing), 1):
            predicted -= (
                0.5
                * dt
                / h
                * (
                    _flux(active + 0.5 * slope, ga, momentum)
                    - _flux(active - 0.5 * slope, ga, momentum)
                )
            )
        if any(
            not physical_state(predicted + sign * 0.5 * s)
            for s in slopes
            for sign in (-1, 1)
        ):
            return _hydro(state, gamma, spacing, dt, limiter, boundaries, True)
    fluxes = []
    for momentum, axis in enumerate(axes, 1):
        if first_order:
            left, right = _faces(padded, axis, False)
            gl, gr = _faces(gp[None], axis, False)
            pl, pr = uconsv_to_uprim_3d(left, gl[0]), uconsv_to_uprim_3d(right, gr[0])
            speed = np.maximum(
                abs(pl[momentum]) + np.sqrt(gl[0] * pl[4] / pl[0]),
                abs(pr[momentum]) + np.sqrt(gr[0] * pr[4] / pr[0]),
            )
            flux = 0.5 * (
                _flux(left, gl[0], momentum)
                + _flux(right, gr[0], momentum)
                - speed * (right - left)
            )
        else:
            left, _ = _faces(predicted + 0.5 * slopes[momentum - 1], axis, True)
            _, right = _faces(predicted - 0.5 * slopes[momentum - 1], axis, True)
            gl, gr = _faces(ga[None], axis, True)
            permutation = [0, momentum] + [m for m in (1, 2, 3) if m != momentum] + [4]
            shape = left.shape
            rotated = hllc_flux(
                left[permutation].reshape(5, -1),
                right[permutation].reshape(5, -1),
                gl.ravel(),
                gamma_right=gr.ravel(),
            ).reshape(shape)
            flux = rotated[np.argsort(permutation)]
        fluxes.append(flux)
    result = state.copy()
    for flux, h, axis in zip(fluxes, spacing, axes):
        result -= dt / h * np.diff(flux, axis=axis)
    if not physical_state(result):
        if not first_order:
            return _hydro(state, gamma, spacing, dt, limiter, boundaries, True)
        raise RuntimeError("3D hydrodynamic update produced a nonphysical state")
    return result, fluxes


def muscl_hancock_step_3d(
    state,
    dx,
    dy,
    dz,
    dt,
    gamma=1.4,
    limiter="VanLeer",
    bc_x="Open",
    bc_y="Open",
    bc_z="Open",
    return_fluxes=False,
):
    result, fluxes = _hydro(state, gamma, (dx, dy, dz), dt, limiter, (bc_z, bc_y, bc_x))
    return (result, *fluxes) if return_fluxes else result


def reaction_step_3d(state, lam, beta, dt, model="arrhenius"):
    flat = state.reshape(5, -1)
    normal = np.stack(
        (flat[0], flat[1], flat[4] - 0.5 * np.sum(flat[2:4] ** 2, axis=0) / flat[0])
    )
    new_lam, reacted, new_beta = new_lambda_n_e_calc(
        lam.ravel(), dt, normal, model, None if beta is None else beta.ravel()
    )
    result = state.copy()
    result[4] += (reacted[2] - normal[2]).reshape(lam.shape)
    return (
        result,
        new_lam.reshape(lam.shape),
        None if beta is None else new_beta.reshape(lam.shape),
    )


def _advect(q, old, new, fluxes, dt, spacing, boundaries):
    qp = _pad(q, boundaries)
    rhoq = old[0] * q
    for flux, h, axis in zip(fluxes, spacing, (3, 2, 1)):
        left, right = _faces(qp[None], axis, False)
        scalar_flux = flux[0] * np.where(flux[0] >= 0, left[0], right[0])
        rhoq -= dt / h * np.diff(scalar_flux, axis=axis - 1)
    fraction = rhoq / new[0]
    if np.any(fraction < -1e-10) or np.any(fraction > 1 + 1e-10):
        raise RuntimeError("3D scalar transport left the physical interval")
    return np.clip(fraction, 0, 1)


def get_3d_preset(name, nx=20, ny=20, nz=20):
    if name not in PRESETS_3D:
        raise ValueError(f"Unknown 3D preset: {name}")
    for n in (nx, ny, nz):
        validate_grid(1, n, 0.4, 0)
    z, y, x = np.meshgrid(
        (np.arange(nz) + 0.5) / nz,
        (np.arange(ny) + 0.5) / ny,
        (np.arange(nx) + 0.5) / nx,
        indexing="ij",
    )
    prim = np.zeros((5, nz, ny, nx))
    if name.startswith("3d_sod_"):
        coordinate = {"x": x, "y": y, "z": z}[name[-1]]
        prim[0] = np.where(coordinate <= 0.5, 1, 0.125)
        prim[4] = np.where(coordinate <= 0.5, 1, 0.1)
    elif name == "3d_explosion":
        inside = (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 <= 0.2**2
        prim[0] = np.where(inside, 1, 0.125)
        prim[4] = np.where(inside, 1, 0.1)
    else:
        prim[0] = 1
        prim[4] = np.where(x <= 0.5, 1e6, 101325)
    return {
        "primitive": prim,
        "is_reactive": name == "3d_ignition",
        "t_end": 1e-5 if name == "3d_ignition" else 0.05,
    }


def run_torch_solver_3d(
    preset_name="3d_explosion",
    *,
    primitive_init=None,
    nx=20,
    ny=20,
    nz=20,
    lx=1.0,
    ly=1.0,
    lz=1.0,
    t_end=None,
    cfl=0.4,
    gamma=1.4,
    limiter="VanLeer",
    bc_x="Open",
    bc_y="Open",
    bc_z="Open",
    is_reactive=False,
    kinetic_model="arrhenius",
    lambda_init=None,
    beta_init=None,
    max_steps=100000,
):
    """Return final 3D state, coordinates and integrated five-equation balances.

    Custom primitives override the preset. Arrays have (5,nz,ny,nx) shape;
    lambda/beta have (nz,ny,nx) shape. Spacings/lengths are ordered x,y,z.
    """
    if t_end is None:
        t_end = (
            1e-5 if preset_name == "3d_ignition" and primitive_init is None else 0.05
        )
    for length, n in zip((lx, ly, lz), (nx, ny, nz)):
        validate_grid(length, n, cfl, t_end)
    if not np.isscalar(gamma) or not np.isfinite(gamma) or gamma <= 1:
        raise ValueError("gamma must be finite and greater than one")
    if limiter not in LIMITERS or kinetic_model not in ("arrhenius", "znd"):
        raise ValueError("Invalid 3D limiter or kinetic model")
    if not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if primitive_init is None:
        cfg = get_3d_preset(preset_name, nx, ny, nz)
        primitive_init, is_reactive = cfg["primitive"], cfg["is_reactive"]
    primitive = np.array(primitive_init, dtype=float, copy=True)
    shape = (nz, ny, nx)
    if (
        primitive.shape != (5, *shape)
        or not np.isfinite(primitive).all()
        or np.any(primitive[[0, 4]] <= 0)
    ):
        raise ValueError(
            "Require finite (5,nz,ny,nx) primitives with positive density/pressure"
        )
    boundaries = (bc_z, bc_y, bc_x)
    _pad(np.zeros((2, 2, 2)), boundaries)  # Validate options even for t_end=0.

    def fraction(values, name):
        out = (
            np.ones(shape)
            if values is None
            else np.array(values, dtype=float, copy=True)
        )
        if (
            out.shape != shape
            or not np.isfinite(out).all()
            or np.any((out < 0) | (out > 1))
        ):
            raise ValueError(f"{name} must have shape (nz,ny,nx) and lie in [0,1]")
        return out

    if not is_reactive and (lambda_init is not None or beta_init is not None):
        raise ValueError("Composition requires is_reactive=True")
    lam = fraction(lambda_init, "lambda_init") if is_reactive else None
    beta = (
        fraction(beta_init, "beta_init")
        if is_reactive and kinetic_model == "znd"
        else None
    )
    if beta_init is not None and beta is None:
        raise ValueError("beta_init requires reactive ZND kinetics")
    gamma = gamma_calc(lam) if is_reactive else gamma
    state = uprim_to_uconsv_3d(primitive, gamma)
    if not physical_state(state):
        raise ValueError("Initial conservative state is nonphysical")
    spacing = (lx / nx, ly / ny, lz / nz)
    volume = np.prod(spacing)
    initial = state.sum(axis=(1, 2, 3)) * volume
    boundary, source = np.zeros(5), np.zeros(5)
    time, steps = 0.0, 0

    def cfl_dt(u, g):
        p = uconsv_to_uprim_3d(u, g)
        sound = np.sqrt(g * p[4] / p[0])
        return cfl / sum(
            np.max(abs(p[m]) + sound) / h for m, h in enumerate(spacing, 1)
        )

    while time < t_end and steps < max_steps:
        g = gamma_calc(lam) if is_reactive else gamma
        dt = min(cfl_dt(state, g), t_end - time)
        if not np.isfinite(dt) or dt <= 0 or time + dt == time:
            raise RuntimeError("3D integration stalled before final time")
        start = state
        if is_reactive:
            for _ in range(60):
                heated, lh, bh = reaction_step_3d(
                    state, lam, beta, 0.5 * dt, kinetic_model
                )
                allowed = cfl_dt(heated, gamma_calc(lh))
                if dt <= allowed * (1 + 1e-12):
                    break
                dt = min(dt * 0.5, allowed)
            else:
                raise RuntimeError("Could not find a stable 3D reactive step")
            old, lam, beta = heated, lh, bh
        else:
            old = state
        g = gamma_calc(lam) if is_reactive else gamma
        state, fluxes = _hydro(old, g, spacing, dt, limiter, boundaries)
        for flux, h, axis in zip(fluxes, spacing, (3, 2, 1)):
            boundary += (
                dt
                * volume
                / h
                * (np.take(flux, -1, axis=axis) - np.take(flux, 0, axis=axis)).sum(
                    axis=(1, 2)
                )
            )
        if is_reactive:
            lam = _advect(lam, old, state, fluxes, dt, spacing, boundaries)
            if beta is not None:
                beta = _advect(beta, old, state, fluxes, dt, spacing, boundaries)
            hydro_energy = state[4].sum()
            state, lam, beta = reaction_step_3d(
                state, lam, beta, 0.5 * dt, kinetic_model
            )
            source[4] += volume * (
                (old[4] - start[4]).sum() + state[4].sum() - hydro_energy
            )
        time += dt
        steps += 1
    if time < t_end:
        raise RuntimeError(f"Maximum steps reached at t={time}; requested {t_end}")
    final = state.sum(axis=(1, 2, 3)) * volume
    gamma = gamma_calc(lam) if is_reactive else gamma
    x, y, z = [(np.arange(n) + 0.5) * h for n, h in zip((nx, ny, nz), spacing)]
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    residual = final - initial + boundary - source
    return dict(
        u_final=state,
        u_prim_final=uconsv_to_uprim_3d(state, gamma),
        lambda_final=lam,
        beta_final=beta,
        time=time,
        steps=steps,
        x=x,
        y=y,
        z=z,
        X=X,
        Y=Y,
        Z=Z,
        initial_integral=initial,
        final_integral=final,
        boundary_integral=boundary,
        source_integral=source,
        balance_residual=residual,
        mass_conservation_error=abs(residual[0]) / initial[0],
        mass_change_rel=(final[0] - initial[0]) / initial[0],
    )
