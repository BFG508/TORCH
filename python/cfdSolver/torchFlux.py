"""Directional Euler HLLC flux shared by the 2D and 3D solvers."""

import numpy as np


def hllc_flux_grid(
    left, right, gamma, gamma_right=None, *, direction=0, batch_size=8192
):
    """Evaluate grid faces with bounded HLLC scratch storage.

    Components keep Cartesian order. Each batch rotates into the normal
    basis. Strided inputs are gathered a batch at a time, avoiding full
    copies of large 3D face arrays.
    """
    if (
        left.shape != right.shape
        or left.ndim < 2
        or left.shape[0] not in (4, 5)
    ):
        raise ValueError("HLLC requires matching four/five-component grids")
    if direction not in range(left.shape[0] - 2):
        raise ValueError("Invalid normal direction")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("HLLC batch size must be a positive integer")
    nv = left.shape[0]
    normal = direction + 1
    order = [0, normal] + [m for m in range(1, nv - 1) if m != normal]
    order += [nv - 1]
    gl = np.broadcast_to(gamma, left.shape[1:])
    gr = (
        gl
        if gamma_right is None
        else np.broadcast_to(gamma_right, left.shape[1:])
    )
    result = np.empty_like(left, order="C")
    flat = result.reshape(nv, -1)
    rows_l = [
        left[m].reshape(-1) if left[m].flags.c_contiguous else left[m].flat
        for m in order
    ]
    rows_r = [
        right[m].reshape(-1) if right[m].flags.c_contiguous else right[m].flat
        for m in order
    ]
    for start in range(0, flat.shape[1], batch_size):
        stop = min(start + batch_size, flat.shape[1])
        ul = np.array([row[start:stop] for row in rows_l])
        ur = np.array([row[start:stop] for row in rows_r])
        flat[order, start:stop] = hllc_flux(
            ul, ur, gl.flat[start:stop], gamma_right=gr.flat[start:stop]
        )
    return result


def hllc_flux(
    u_l: np.ndarray,
    u_r: np.ndarray,
    gamma: float = 1.4,
    wave_speed_mode: str = "Pressure Based",
    gamma_right=None,
) -> np.ndarray:
    """
    Normal HLLC flux with one or two tangential momenta (4 or 5 rows).
    """
    if u_l.ndim != 2 or u_l.shape != u_r.shape or u_l.shape[0] not in (4, 5):
        raise ValueError("HLLC requires matching (4,m) or (5,m) states")
    gamma_l = gamma
    gamma_r = gamma if gamma_right is None else gamma_right
    gamma = 0.5 * (gamma_l + gamma_r)
    rho_l = np.maximum(u_l[0], 1e-12)
    rho_r = np.maximum(u_r[0], 1e-12)

    un_l = u_l[1] / rho_l
    un_r = u_r[1] / rho_r

    ut_l = u_l[2:-1] / rho_l
    ut_r = u_r[2:-1] / rho_r

    ke_l = 0.5 * rho_l * (un_l**2 + np.sum(ut_l**2, axis=0))
    ke_r = 0.5 * rho_r * (un_r**2 + np.sum(ut_r**2, axis=0))

    p_l = np.maximum((gamma_l - 1.0) * (u_l[-1] - ke_l), 1e-12)
    p_r = np.maximum((gamma_r - 1.0) * (u_r[-1] - ke_r), 1e-12)

    a_l = np.sqrt(gamma_l * p_l / rho_l)
    a_r = np.sqrt(gamma_r * p_r / rho_r)

    # Physical normal fluxes
    f_l = np.empty_like(u_l)
    f_l[0] = rho_l * un_l
    f_l[1] = rho_l * un_l**2 + p_l
    f_l[2:-1] = rho_l * un_l * ut_l
    f_l[-1] = un_l * (u_l[-1] + p_l)

    f_r = np.empty_like(u_r)
    f_r[0] = rho_r * un_r
    f_r[1] = rho_r * un_r**2 + p_r
    f_r[2:-1] = rho_r * un_r * ut_r
    f_r[-1] = un_r * (u_r[-1] + p_r)

    # Roe averages
    sqrt_l = np.sqrt(rho_l)
    sqrt_r = np.sqrt(rho_r)
    inv_denom = 1.0 / (sqrt_l + sqrt_r)

    u_roe = (sqrt_l * un_l + sqrt_r * un_r) * inv_denom
    v_roe = (sqrt_l * ut_l + sqrt_r * ut_r) * inv_denom
    h_l = (u_l[-1] + p_l) / rho_l
    h_r = (u_r[-1] + p_r) / rho_r
    h_roe = (sqrt_l * h_l + sqrt_r * h_r) * inv_denom
    ke_roe = 0.5 * (u_roe**2 + np.sum(v_roe**2, axis=0))
    a_roe = np.sqrt(np.maximum((gamma - 1.0) * (h_roe - ke_roe), 1e-12))

    # Roe-Einfeldt acoustic wave speeds
    s_l = np.minimum(un_l - a_l, u_roe - a_roe)
    s_r = np.maximum(un_r + a_r, u_roe + a_roe)

    # Middle wave speed S_*
    denom_star = rho_l * (s_l - un_l) - rho_r * (s_r - un_r)
    denom_star_safe = np.where(np.abs(denom_star) < 1e-14, 1e-14, denom_star)
    s_star = (
        p_r - p_l + rho_l * un_l * (s_l - un_l) - rho_r * un_r * (s_r - un_r)
    ) / denom_star_safe

    # Star states U_*L and U_*R
    denom_sl_ss = np.where(np.abs(s_l - s_star) < 1e-14, 1e-14, s_l - s_star)
    denom_sr_ss = np.where(np.abs(s_r - s_star) < 1e-14, 1e-14, s_r - s_star)

    fac_l = (s_l - un_l) / denom_sl_ss
    fac_r = (s_r - un_r) / denom_sr_ss

    rho_sl = np.maximum(rho_l * fac_l, 1e-12)
    rho_sr = np.maximum(rho_r * fac_r, 1e-12)

    denom_sl_un = np.where(np.abs(s_l - un_l) < 1e-14, 1e-14, s_l - un_l)
    denom_sr_un = np.where(np.abs(s_r - un_r) < 1e-14, 1e-14, s_r - un_r)

    u_star_l = np.empty_like(u_l)
    u_star_l[0] = rho_sl
    u_star_l[1] = rho_sl * s_star
    u_star_l[2:-1] = rho_sl * ut_l  # Passively advected tangential velocity
    u_star_l[-1] = rho_sl * (
        u_l[-1] / rho_l
        + (s_star - un_l) * (s_star + p_l / (rho_l * denom_sl_un))
    )

    u_star_r = np.empty_like(u_r)
    u_star_r[0] = rho_sr
    u_star_r[1] = rho_sr * s_star
    u_star_r[2:-1] = rho_sr * ut_r
    u_star_r[-1] = rho_sr * (
        u_r[-1] / rho_r
        + (s_star - un_r) * (s_star + p_r / (rho_r * denom_sr_un))
    )

    # Star fluxes
    f_star_l = f_l + s_l * (u_star_l - u_l)
    f_star_r = f_r + s_r * (u_star_r - u_r)

    # Bypass narrow star states
    mask_close_l = np.abs(s_l - s_star) < 1e-12
    mask_close_r = np.abs(s_r - s_star) < 1e-12
    f_star_l[:, mask_close_l] = f_l[:, mask_close_l]
    f_star_r[:, mask_close_r] = f_r[:, mask_close_r]

    # Assemble piecewise HLLC flux
    flux = np.empty_like(u_l)
    mask_l = 0.0 <= s_l
    mask_star_l = (s_l <= 0.0) & (0.0 <= s_star)
    mask_star_r = (s_star <= 0.0) & (0.0 <= s_r)
    mask_r = 0.0 >= s_r

    flux[:, mask_l] = f_l[:, mask_l]
    flux[:, mask_star_l] = f_star_l[:, mask_star_l]
    flux[:, mask_star_r] = f_star_r[:, mask_star_r]
    flux[:, mask_r] = f_r[:, mask_r]

    return flux
