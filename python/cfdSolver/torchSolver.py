"""TORCH Core Solver Module (Python)
Unified finite-volume solver engine for 1D Euler / Reactive gas
dynamics.
Contains thermodynamics, boundary conditions, slope limiters, Riemann
solver,
MUSCL reconstruction, HLLC flux, CFL time stepping, Arrhenius/ZND
kinetics,
presets, JIT acceleration kernels, and simulation runners.

Author: BFG508 (Python implementation based on the formulation by Paul
Augustus Aliaga Brown)
"""

from typing import Union, Optional, Dict, Any, NamedTuple
import numpy as np

# ======================================================================
# JIT ACCELERATION KERNELS (NUMBA FALLBACK)
# ======================================================================

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator


@njit(fastmath=True)
def jit_vanleer_limiter(r: np.ndarray) -> np.ndarray:
    """JIT-accelerated Van Leer slope limiter kernel."""
    out = np.zeros_like(r)
    n = len(r)
    for i in range(n):
        if r[i] > 0.0:
            out[i] = (r[i] + abs(r[i])) / (1.0 + r[i])
        else:
            out[i] = 0.0
    return out


@njit(fastmath=True)
def jit_state_update(
    u_inner: np.ndarray,
    f_plus: np.ndarray,
    f_minus: np.ndarray,
    dt_over_dx: float,
) -> np.ndarray:
    """JIT-accelerated flux conservative update kernel."""
    n_vars, n_cells = u_inner.shape
    u_out = np.zeros_like(u_inner)
    for v in range(n_vars):
        for i in range(n_cells):
            u_out[v, i] = u_inner[v, i] - dt_over_dx * (
                f_plus[v, i] - f_minus[v, i]
            )
    return u_out


# ======================================================================
# THERMODYNAMICS & PHYSICS
# ======================================================================


def gamma_calc(
    lambda_val: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculates specific heat ratio gamma(lambda) = (4.5 - lambda) / (3.5
    - lambda).
    """
    return (4.5 - lambda_val) / (3.5 - lambda_val)


def sound_speed_calc(
    u_properties: np.ndarray, gamma: Union[float, np.ndarray]
) -> np.ndarray:
    """Calculates speed of sound a = sqrt(gamma * p / rho)."""
    rho = np.maximum(u_properties[0, :], 1e-12)
    p = np.maximum(u_properties[2, :], 0.0)
    return np.sqrt(gamma * p / rho)


def temp_calc(
    gamma: Union[float, np.ndarray],
    lambda_val: np.ndarray,
    u_0_wo_bc: np.ndarray,
) -> np.ndarray:
    """
    Calculates temperature T (Kelvin) using ideal gas law for reacting
    mixture.
    """
    r_univ = 8314.46  # J/(kmol K), with molar masses in kg/kmol
    u_prim = uconsv_to_uprim(u_0_wo_bc, gamma)
    rho = np.maximum(u_prim[0, :], 1e-12)
    p = u_prim[2, :]
    p_pa = p  # All reactive states use SI: pressure in Pa, energy in J/m^3.

    m_h2 = 2.016
    m_o2 = 32.0
    m_h2o = 18.016

    x_h2 = lambda_val * (2.0 / 3.0)
    x_o2 = lambda_val * (1.0 / 3.0)
    x_h2o = 1.0 - lambda_val

    m_m = x_h2 * m_h2 + x_o2 * m_o2 + x_h2o * m_h2o
    return (p_pa * m_m) / (rho * r_univ)


def uconsv_to_uprim(
    u_consv: np.ndarray, gamma: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Converts conservative state [rho, rho*u, E] to primitive [rho, u, p]
    with positivity floor.
    """
    rho = np.maximum(u_consv[0, :], 1e-12)
    u = u_consv[1, :] / rho
    p = np.maximum((u_consv[2, :] - 0.5 * rho * (u**2)) * (gamma - 1.0), 1e-12)
    return np.vstack((rho, u, p))


def uprim_to_uconsv(
    u_prim: np.ndarray, gamma: Union[float, np.ndarray]
) -> np.ndarray:
    """Converts primitive state [rho, u, p] to conservative [rho, rho*u,
    E].
    """
    rho = u_prim[0, :]
    u = u_prim[1, :]
    p = u_prim[2, :]
    rho_safe = np.where(np.abs(rho) < 1e-14, 1e-14, rho)
    e_int = p / ((gamma - 1.0) * rho_safe)
    e_tot = rho * (0.5 * (u**2) + e_int)
    return np.vstack((rho, rho * u, e_tot))


def u_to_fu_calc(
    u_consv: np.ndarray, gamma: Union[float, np.ndarray] = 1.4
) -> np.ndarray:
    """Calculates Euler physical flux vector F(U) = [rho*u, rho*u^2 + p,
    u*(E + p)].
    """
    u_prim = uconsv_to_uprim(u_consv, gamma)
    rho = u_prim[0, :]
    u = u_prim[1, :]
    p = u_prim[2, :]
    e_tot = u_consv[2, :]
    return np.vstack((rho * u, rho * (u**2) + p, u * (e_tot + p)))


# ======================================================================
# BOUNDARY CONDITIONS
# ======================================================================


def bc_creator(u_wo_bc: np.ndarray, reflective_or_open: str) -> np.ndarray:
    """
    Adds transmissive (Open) or reflective ghost cells to conservative
    1D state vector.
    """
    if reflective_or_open == "Reflective":
        bc_left = np.array(
            [[u_wo_bc[0, 0]], [-u_wo_bc[1, 0]], [u_wo_bc[2, 0]]]
        )
        bc_right = np.array(
            [[u_wo_bc[0, -1]], [-u_wo_bc[1, -1]], [u_wo_bc[2, -1]]]
        )
    elif reflective_or_open == "Periodic":
        bc_left = u_wo_bc[:, -1:]
        bc_right = u_wo_bc[:, :1]
    elif reflective_or_open == "Open":
        bc_left = u_wo_bc[:, :1]
        bc_right = u_wo_bc[:, -1:]
    else:
        raise ValueError(
            f"Unknown boundary condition option: {reflective_or_open}"
        )
    return np.hstack((bc_left, u_wo_bc, bc_right))


# ======================================================================
# SLOPE LIMITERS
# ======================================================================


def nan_to_zero_calc(array: np.ndarray) -> np.ndarray:
    """Replaces NaNs and Infinities with 0.0 in numpy arrays."""
    out = np.copy(array)
    out[np.isnan(out) | np.isinf(out)] = 0.0
    return out


def slope_limiter_calc(
    delta_i: np.ndarray,
    c_cfl: float,
    di_uw: np.ndarray,
    di_dw: np.ndarray,
    w_controler: np.ndarray,
    limiter: str,
) -> np.ndarray:
    """Calculates limited slope delta_lim using TVD slope limiters:
    VanLeer, Albada, MINBEE, Superbee, and Unos.
    Guarantees Sweby TVD region bounds for generalized MUSCL parameter w
    in [-1, 1].
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = di_uw / di_dw
    r = nan_to_zero_calc(r)

    w = w_controler
    eps = np.zeros_like(delta_i)

    denom = 1.0 - w + (1.0 + w) * r
    denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps_r = 2.0 / denom_safe
    eps_r = nan_to_zero_calc(eps_r)

    if limiter == "VanLeer":
        pos_mask = r > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            van_leer_val = (2.0 * r) / (1.0 + r)
        van_leer_val = nan_to_zero_calc(van_leer_val)
        eps[pos_mask] = np.minimum(van_leer_val[pos_mask], eps_r[pos_mask])

    elif limiter == "Albada":
        pos_mask = r > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            albada_val = (r * (1.0 + r)) / (1.0 + r**2)
        albada_val = nan_to_zero_calc(albada_val)
        eps[pos_mask] = np.minimum(albada_val[pos_mask], eps_r[pos_mask])

    elif limiter == "MINBEE":
        # Authentic Roe MINBEE: r for 0 < r < 1, 1 for r >= 1
        pos1 = (r > 0) & (r < 1.0)
        pos2 = r >= 1.0
        eps[pos1] = r[pos1]
        eps[pos2] = np.minimum(1.0, eps_r[pos2])

    elif limiter == "Superbee":
        pos_mask = r > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            term1 = np.minimum(2.0 * r, 1.0)
            term2 = np.minimum(r, 2.0)
            superbee_val = np.maximum(0.0, np.maximum(term1, term2))
        superbee_val = nan_to_zero_calc(superbee_val)
        eps[pos_mask] = np.minimum(superbee_val[pos_mask], eps_r[pos_mask])

    elif limiter == "Unos":
        eps = np.minimum(
            np.maximum(
                0.0, np.minimum(np.minimum(2.0, 2.0 * r), (2.0 + r) / 3.0)
            ),
            eps_r,
        )

    else:
        raise ValueError(f"Unknown slope limiter option: {limiter}")

    eps = np.maximum(0.0, nan_to_zero_calc(eps))
    slope = eps * delta_i
    bound = 2.0 * np.minimum(np.abs(di_uw), np.abs(di_dw))
    return np.where(
        di_uw * di_dw > 0,
        np.sign(di_dw) * np.minimum(np.abs(slope), bound),
        0.0,
    )


# ======================================================================
# RIEMANN SOLVER & WAVE SPEEDS
# ======================================================================


def pressure_estimate(
    u_w_bc: np.ndarray, gamma: Union[np.ndarray, float]
) -> np.ndarray:
    """Estimate the intermediate pressure p_* with the PVRS scheme."""
    if np.isscalar(gamma):
        gamma_l = gamma
        gamma_r = gamma
    else:
        gamma_l = gamma[:-1]
        gamma_r = gamma[1:]

    u_prim = uconsv_to_uprim(u_w_bc, gamma)
    u_prim_l = u_prim[:, :-1]
    rho_l = u_prim_l[0, :]
    u_l = u_prim_l[1, :]
    p_l = u_prim_l[2, :]
    a_l = sound_speed_calc(u_prim_l, gamma_l)

    u_prim_r = u_prim[:, 1:]
    rho_r = u_prim_r[0, :]
    u_r = u_prim_r[1, :]
    p_r = u_prim_r[2, :]
    a_r = sound_speed_calc(u_prim_r, gamma_r)

    rho_avg = 0.5 * (rho_r + rho_l)
    a_avg = 0.5 * (a_l + a_r)
    p_pvrs = 0.5 * (p_l + p_r) - 0.5 * (u_r - u_l) * rho_avg * a_avg
    return np.maximum(0.0, p_pvrs)


def wave_speed_estimate(
    u_w_bc: np.ndarray,
    gamma: Union[np.ndarray, float],
    wave_speed_calc_option: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates wave speeds S_L, S_*, S_R at interfaces using Pressure or
    Enthalpy criteria.
    """
    if np.isscalar(gamma):
        gamma_l = gamma
        gamma_r = gamma
    else:
        gamma_l = gamma[:-1]
        gamma_r = gamma[1:]

    with np.errstate(divide="ignore", invalid="ignore"):
        u_prim_l = uconsv_to_uprim(u_w_bc[:, :-1], gamma_l)
        rho_l = np.maximum(u_prim_l[0, :], 1e-12)
        u_l = u_prim_l[1, :]
        p_l = u_prim_l[2, :]
        a_l = sound_speed_calc(u_prim_l, gamma_l)
        e_l = p_l / ((gamma_l - 1.0) * rho_l)
        e_l_tot = rho_l * (0.5 * (u_l**2) + e_l)

        u_prim_r = uconsv_to_uprim(u_w_bc[:, 1:], gamma_r)
        rho_r = np.maximum(u_prim_r[0, :], 1e-12)
        u_r = u_prim_r[1, :]
        p_r = u_prim_r[2, :]
        a_r = sound_speed_calc(u_prim_r, gamma_r)
        e_r = p_r / ((gamma_r - 1.0) * rho_r)
        e_r_tot = rho_r * (0.5 * (u_r**2) + e_r)

    if wave_speed_calc_option == "Pressure Based":
        p_s = pressure_estimate(u_w_bc, gamma)
        q_l = np.ones_like(p_s)
        q_r = np.ones_like(p_s)
        mask_l = p_s > p_l
        mask_r = p_s > p_r

        g_l = gamma_l[mask_l] if isinstance(gamma_l, np.ndarray) else gamma_l
        g_r = gamma_r[mask_r] if isinstance(gamma_r, np.ndarray) else gamma_r

        with np.errstate(divide="ignore", invalid="ignore"):
            q_l[mask_l] = np.sqrt(
                1.0
                + (g_l + 1.0) / (2.0 * g_l) * (p_s[mask_l] / p_l[mask_l] - 1.0)
            )
            q_r[mask_r] = np.sqrt(
                1.0
                + (g_r + 1.0) / (2.0 * g_r) * (p_s[mask_r] / p_r[mask_r] - 1.0)
            )

        s_l = u_l - a_l * q_l
        s_r = u_r + a_r * q_r

    elif wave_speed_calc_option == "Enthalpy Based":
        with np.errstate(divide="ignore", invalid="ignore"):
            sqrt_rho_l = np.sqrt(rho_l)
            sqrt_rho_r = np.sqrt(rho_r)
            denom_roe = np.maximum(sqrt_rho_l + sqrt_rho_r, 1e-12)
            u_avg = (sqrt_rho_l * u_l + sqrt_rho_r * u_r) / denom_roe

            h_l = (e_l_tot + p_l) / rho_l
            h_r = (e_r_tot + p_r) / rho_r
            h_avg = (sqrt_rho_l * h_l + sqrt_rho_r * h_r) / denom_roe

            if isinstance(gamma_l, np.ndarray):
                g_avg = 0.5 * (gamma_l + gamma_r)
            else:
                g_avg = gamma_l

            a_avg_sq = (g_avg - 1.0) * (h_avg - 0.5 * (u_avg**2))
            a_avg = np.sqrt(np.maximum(a_avg_sq, 1e-10))

        # Einfeldt positivity-preserving wave speed bounds
        s_l = np.minimum(u_l - a_l, u_avg - a_avg)
        s_r = np.maximum(u_r + a_r, u_avg + a_avg)
    else:
        raise ValueError(
            f"Unknown wave speed calculation option: {wave_speed_calc_option}"
        )

    denom_s_s = rho_l * (s_l - u_l) - rho_r * (s_r - u_r)
    denom_s_s_safe = np.where(np.abs(denom_s_s) < 1e-14, 1e-14, denom_s_s)
    s_s = (
        p_r - p_l + rho_l * u_l * (s_l - u_l) - rho_r * u_r * (s_r - u_r)
    ) / denom_s_s_safe
    # Physical wave bounds: S_L <= S_* <= S_R (Toro HLLC wave structure)
    s_s = np.clip(s_s, s_l, s_r)
    wave_speed_matrix = np.vstack((s_l, s_s, s_r))
    return s_l, s_s, s_r, wave_speed_matrix


# ======================================================================
# MUSCL RECONSTRUCTION
# ======================================================================


def deltai_calc(
    u_w_bc: np.ndarray, w_controler: str, gamma: Union[np.ndarray, float] = 1.4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates upwind/downwind difference vectors and controller
    parameter w.
    """
    u_inner = u_w_bc[:, 1:-1]
    u_left = u_w_bc[:, :-2]
    u_right = u_w_bc[:, 2:]

    delta_i = 0.5 * (u_right - u_left)
    di_uw = u_inner - u_left
    di_dw = u_right - u_inner

    if w_controler == "Sound Speed Controlled":
        gamma_eff = gamma if np.isscalar(gamma) else gamma[1:-1]
        u_prim = uconsv_to_uprim(u_inner, gamma_eff)
        a = sound_speed_calc(u_prim, gamma_eff)
        u = np.abs(u_prim[1, :])

        with np.errstate(divide="ignore", invalid="ignore"):
            mach = u / np.maximum(a, 1e-12)
        mach[np.isnan(mach) | np.isinf(mach)] = 0.0

        w_array = (1.0 - mach) / (1.0 + mach)
        w_array = np.clip(w_array, -1.0, 1.0)

    elif w_controler == "w=0":
        w_array = np.zeros(u_inner.shape[1])
    else:
        raise ValueError(f"Unknown w_controler option: {w_controler}")

    return delta_i, di_uw, di_dw, w_array


def muscl_wlim_calc(
    u_w_bc: np.ndarray,
    dt: float,
    dx: float,
    gamma: Union[np.ndarray, float],
    c_cfl: float,
    limiter: str,
    w_controler: str,
) -> np.ndarray:
    """Calculates MUSCL reconstruction with slope limiting and 0.5t
    predictor evolution.
    """
    delta_i, di_uw, di_dw, w_array = deltai_calc(u_w_bc, w_controler, gamma)
    delta_lim = slope_limiter_calc(
        delta_i, c_cfl, di_uw, di_dw, w_array, limiter
    )

    u_inner = u_w_bc[:, 1:-1]
    u_l = u_inner - 0.5 * delta_lim
    u_r = u_inner + 0.5 * delta_lim

    gamma_inner = gamma if np.isscalar(gamma) else gamma[1:-1]
    fu_l = u_to_fu_calc(u_l, gamma_inner)
    fu_r = u_to_fu_calc(u_r, gamma_inner)

    d_fu = -(dt / (2.0 * dx)) * (fu_r - fu_l)
    u_l_05t = u_l + d_fu
    u_r_05t = u_r + d_fu

    for state in (u_l_05t, u_r_05t):
        pressure = (gamma_inner - 1.0) * (
            state[2] - 0.5 * state[1] ** 2 / np.maximum(state[0], 1e-12)
        )
        bad = (
            (state[0] <= 1e-12)
            | (pressure <= 1e-12)
            | ~np.isfinite(state).all(axis=0)
        )
        state[:, bad] = u_inner[:, bad]

    num_cells = u_inner.shape[1]
    u_subdiv = np.empty((3, 2 * num_cells))
    u_subdiv[:, 0::2] = u_l_05t
    u_subdiv[:, 1::2] = u_r_05t
    return u_subdiv


def muscl_application(
    u_w_bc: np.ndarray,
    dt: float,
    dx: float,
    gamma: Union[np.ndarray, float],
    c_cfl: float,
    limiter: str,
    w_controler: str,
    reflective_or_open: str,
) -> np.ndarray:
    """Driver function for non-reactive MUSCL reconstruction."""
    u_subdiv = muscl_wlim_calc(
        u_w_bc, dt, dx, gamma, c_cfl, limiter, w_controler
    )
    return bc_creator(u_subdiv, reflective_or_open)


def muscl_application_wqr(
    u_w_bc: np.ndarray,
    dt: float,
    dx: float,
    lambda_w_bc: np.ndarray,
    c_cfl: float,
    limiter: str,
    w_controler: str,
    reflective_or_open: str,
) -> np.ndarray:
    """Driver function for reactive MUSCL reconstruction."""
    gamma_w_bc = gamma_calc(lambda_w_bc)
    u_subdiv = muscl_wlim_calc(
        u_w_bc, dt, dx, gamma_w_bc, c_cfl, limiter, w_controler
    )
    return bc_creator(u_subdiv, reflective_or_open)


# ======================================================================
# HLLC FLUX
# ======================================================================


def hllc_flux_calc_for_muscl(
    u_1_w_bc: np.ndarray,
    wave_speed_matrix: np.ndarray,
    gamma: Union[np.ndarray, float],
    variante: str,
    execution=None,
) -> np.ndarray:
    """Calculates HLLC approximate Riemann flux vector at interfaces."""
    if execution is not None and execution.workers > 1:
        n = wave_speed_matrix.shape[1]
        out = np.empty((3, n))

        def evaluate(bounds):
            start, stop = bounds
            g = gamma if np.isscalar(gamma) else gamma[start: stop + 1]
            value = hllc_flux_calc_for_muscl(
                u_1_w_bc[:, 2 * start: 2 * stop],
                wave_speed_matrix[:, start:stop],
                g,
                variante,
            )
            return start, stop, value

        tasks = ((a, min(a + 8192, n)) for a in range(0, n, 8192))
        for start, stop, value in execution.map(evaluate, tasks):
            out[:, start:stop] = value
        return out
    l1 = wave_speed_matrix.shape[1]
    u_l = u_1_w_bc[:, 0::2]
    u_r = u_1_w_bc[:, 1::2]

    if np.isscalar(gamma):
        gamma_l = gamma
        gamma_r = gamma
    else:
        gamma_l = gamma[:-1]
        gamma_r = gamma[1:]

    u_prim_l = uconsv_to_uprim(u_l, gamma_l)
    rho_l = np.maximum(u_prim_l[0, :], 1e-12)
    u_val_l = u_prim_l[1, :]
    p_l = u_prim_l[2, :]
    e_l = p_l / ((gamma_l - 1.0) * rho_l)
    e_l_tot = rho_l * (0.5 * (u_val_l**2) + e_l)

    u_prim_r = uconsv_to_uprim(u_r, gamma_r)
    rho_r = np.maximum(u_prim_r[0, :], 1e-12)
    u_val_r = u_prim_r[1, :]
    p_r = u_prim_r[2, :]
    e_r = p_r / ((gamma_r - 1.0) * rho_r)
    e_r_tot = rho_r * (0.5 * (u_val_r**2) + e_r)

    f_l = u_to_fu_calc(u_l, gamma_l)
    f_r = u_to_fu_calc(u_r, gamma_r)

    s_l = wave_speed_matrix[0, :]
    s_r = wave_speed_matrix[2, :]
    # The contact wave must satisfy momentum balance for the predicted
    # interface states. Reusing its cell-average value makes smooth
    # acoustic transport only first order, despite MUSCL reconstruction.
    denominator = rho_l * (s_l - u_val_l) - rho_r * (s_r - u_val_r)
    denominator = np.where(abs(denominator) < 1e-14, 1e-14, denominator)
    s_s = (
        p_r
        - p_l
        + rho_l * u_val_l * (s_l - u_val_l)
        - rho_r * u_val_r * (s_r - u_val_r)
    ) / denominator
    s_s = np.clip(s_s, s_l, s_r)

    eps_denom = 1e-12

    if variante == "Variante 1 - Original":
        valid_l = np.abs(s_l - s_s) > eps_denom
        valid_r = np.abs(s_r - s_s) > eps_denom
        denom_l = np.where(valid_l, s_l - s_s, -eps_denom)
        denom_r = np.where(valid_r, s_r - s_s, eps_denom)

        denom_sl_u = np.where(
            np.abs(s_l - u_val_l) < eps_denom, -eps_denom, s_l - u_val_l
        )
        denom_sr_u = np.where(
            np.abs(s_r - u_val_r) < eps_denom, eps_denom, s_r - u_val_r
        )

        term_l_2 = e_l_tot / rho_l + (s_s - u_val_l) * (
            s_s + (p_l / (rho_l * denom_sl_u))
        )
        term_r_2 = e_r_tot / rho_r + (s_s - u_val_r) * (
            s_s + (p_r / (rho_r * denom_sr_u))
        )

        u_ls = (
            rho_l
            * ((s_l - u_val_l) / denom_l)
            * np.vstack((np.ones(l1), s_s, term_l_2))
        )
        u_ls[:, ~valid_l] = u_l[:, ~valid_l]

        u_rs = (
            rho_r
            * ((s_r - u_val_r) / denom_r)
            * np.vstack((np.ones(l1), s_s, term_r_2))
        )
        u_rs[:, ~valid_r] = u_r[:, ~valid_r]

        f_ls = f_l + s_l * (u_ls - u_l)
        f_rs = f_r + s_r * (u_rs - u_r)

    elif variante == "Variante_2":
        d_s = np.vstack((np.zeros(l1), np.ones(l1), s_s))
        denom_l = np.where(
            np.abs(s_l - s_s) < eps_denom,
            np.sign(s_l - s_s + 1e-30) * eps_denom,
            s_l - s_s,
        )
        denom_r = np.where(
            np.abs(s_r - s_s) < eps_denom,
            np.sign(s_r - s_s + 1e-30) * eps_denom,
            s_r - s_s,
        )
        f_ls = (
            s_s * (s_l * u_l - f_l)
            + s_l * (p_l + rho_l * (s_l - u_val_l) * (s_s - u_val_l)) * d_s
        ) / denom_l
        f_rs = (
            s_s * (s_r * u_r - f_r)
            + s_r * (p_r + rho_r * (s_r - u_val_r) * (s_s - u_val_r)) * d_s
        ) / denom_r
    else:
        raise ValueError(f"Unknown HLLC variant: {variante}")

    f_hllc = np.zeros((3, l1))
    mask_left = s_l > 0
    mask_mid_l = (s_l <= 0) & (s_s > 0)
    mask_mid_r = (s_s <= 0) & (s_r >= 0)
    mask_right = s_r < 0

    f_hllc[:, mask_left] = f_l[:, mask_left]
    f_hllc[:, mask_mid_l] = f_ls[:, mask_mid_l]
    f_hllc[:, mask_mid_r] = f_rs[:, mask_mid_r]
    f_hllc[:, mask_right] = f_r[:, mask_right]
    return f_hllc


# ======================================================================
# CFL TIME STEP & STATE ADVANCE
# ======================================================================


def delta_t_estimate(
    wave_speed_matrix: np.ndarray, dx: float, c_cfl: float
) -> float:
    """Calculates adaptive time step dt governed by CFL condition."""
    max_wave_speed = np.max(np.abs(wave_speed_matrix))
    return (c_cfl * dx) / max_wave_speed if max_wave_speed > 0 else 1e-6


def flux_application_05t(
    u_0_w_bc: np.ndarray, flux: np.ndarray, dt: float, dx: float
) -> np.ndarray:
    """Update conservative variables with the interface fluxes."""
    f_plus = flux[:, 1:]
    f_minus = flux[:, :-1]
    d_u = -(dt / dx) * (f_plus - f_minus)
    u_next = u_0_w_bc[:, 1:-1] + d_u
    return u_next


# ======================================================================
# CHEMICAL KINETICS (ARRHENIUS & ZND)
# ======================================================================


def new_lambda_n_e_calc(
    lambda_n: np.ndarray,
    dt_r: float,
    u_0_wo_bc: np.ndarray,
    model: str = "arrhenius",
    beta_n: Union[np.ndarray, None] = None,
    *,
    q_heat: float = 141584000.0,
    activation_energy: float = 17.15 * 4184.0,
    pre_exponential: float = 1.76e5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Applies Arrhenius or 2-stage ZND chemical kinetics to update
    reactant fraction
    lambda, induction parameter beta, and total energy due to heat
    release Q.
    """
    if model not in ("arrhenius", "znd"):
        raise ValueError(f"Unknown kinetic model: {model}")
    if (
        not np.isfinite(
            [dt_r, q_heat, activation_energy, pre_exponential]
        ).all()
        or min(dt_r, q_heat, activation_energy, pre_exponential) < 0
    ):
        raise ValueError(
            "Reaction time and constants must be finite and nonnegative"
        )
    lambda_n = np.asarray(lambda_n, dtype=float)
    if np.any(~np.isfinite(lambda_n)) or np.any(
        (lambda_n < 0) | (lambda_n > 1)
    ):
        raise ValueError("Reactant fraction must lie in [0,1]")
    u_0_wo_bc = np.asarray(u_0_wo_bc, dtype=float)
    if (
        u_0_wo_bc.shape != (3, lambda_n.size)
        or lambda_n.ndim != 1
        or not physical_state(u_0_wo_bc)
    ):
        raise ValueError(
            (
                "Reaction requires a physical (3,n) state and an (n,) "
                "reactant fraction"
            )
        )
    beta_n = (
        np.ones_like(lambda_n)
        if beta_n is None
        else np.asarray(beta_n, dtype=float)
    )
    if (
        beta_n.shape != lambda_n.shape
        or not np.isfinite(beta_n).all()
        or np.any((beta_n < 0) | (beta_n > 1))
    ):
        raise ValueError(
            "Induction fraction must have one value in [0,1] per cell"
        )

    def rate_at(lam, state):
        temperature = np.maximum(temp_calc(gamma_calc(lam), lam, state), 1e-12)
        return pre_exponential * np.exp(
            -activation_energy / (8.314 * temperature)
        )

    def advance(rate, duration):
        if model == "znd":
            active_time = np.maximum(
                0.0, duration - beta_n / np.maximum(rate, 1e-300)
            )
            lam = lambda_n * np.exp(-5.0 * rate * active_time)
            beta = np.clip(beta_n - rate * duration, 0.0, 1.0)
        else:
            lam = lambda_n * np.exp(-rate * duration)
            beta = np.zeros_like(beta_n)
        state = np.copy(u_0_wo_bc)
        state[2] += q_heat * state[0] * (lambda_n - lam)
        return lam, state, beta

    # Exponential midpoint: evaluate temperature/composition at the
    # half-step.
    lam_mid, state_mid, _ = advance(rate_at(lambda_n, u_0_wo_bc), 0.5 * dt_r)
    return advance(rate_at(lam_mid, state_mid), dt_r)


# ======================================================================
# BENCHMARK PRESETS
# ======================================================================


class SimulationPreset(NamedTuple):
    """Container for 1D CFD simulation setup parameters."""

    name: str
    description: str
    mode: str
    n_cells: int
    t_end: float
    cfl: float
    rho_l: float
    u_l: float
    p_l: float
    rho_r: float
    u_r: float
    p_r: float
    limiter: str


PresetConfig = SimulationPreset

PRESETS: Dict[str, SimulationPreset] = {
    "sod_standard": SimulationPreset(
        name="Sod Standard Shock Tube",
        description=(
            "Classic Sod shock tube test case (inert flow, non-reactive)."
        ),
        mode="sod",
        n_cells=100,
        t_end=0.2,
        cfl=0.4,
        rho_l=1.0,
        u_l=0.0,
        p_l=1.0,
        rho_r=0.125,
        u_r=0.0,
        p_r=0.1,
        limiter="VanLeer",
    ),
    "detonation_cj": SimulationPreset(
        name="Reactive Ignition (unvalidated CJ compatibility preset)",
        description=(
            "Reactive H2/O2 mixture under high-pressure ignition conditions."
        ),
        mode="reactive",
        n_cells=100,
        t_end=0.01,
        cfl=0.4,
        rho_l=1.0,
        u_l=0.0,
        p_l=50.0 * 101325.0,
        rho_r=0.125,
        u_r=0.0,
        p_r=0.1 * 101325.0,
        limiter="VanLeer",
    ),
    "rarefaction_wave": SimulationPreset(
        name="Double Expansion Fan",
        description="Symmetric two-rarefaction wave expansion test.",
        mode="sod",
        n_cells=100,
        t_end=0.15,
        cfl=0.4,
        rho_l=1.0,
        u_l=-2.0,
        p_l=0.4,
        rho_r=1.0,
        u_r=2.0,
        p_r=0.4,
        limiter="Albada",
    ),
    "strong_shock": SimulationPreset(
        name="Strong Shock Interaction",
        description=(
            "High pressure-ratio shock wave propagation (Toro Test 3)."
        ),
        mode="sod",
        n_cells=150,
        t_end=0.015,
        cfl=0.4,
        rho_l=1.0,
        u_l=0.0,
        p_l=1000.0,
        rho_r=1.0,
        u_r=0.0,
        p_r=0.01,
        limiter="VanLeer",
    ),
    "toro_test4": SimulationPreset(
        name="Toro Test 4 (Slow Shock & Contact)",
        description=(
            "Toro test 4 from TFG p.87: slow shock and contact discontinuity."
        ),
        mode="sod",
        n_cells=100,
        t_end=0.035,
        cfl=0.4,
        rho_l=1.0,
        u_l=0.0,
        p_l=0.01,
        rho_r=1.0,
        u_r=0.0,
        p_r=100.0,
        limiter="VanLeer",
    ),
}


def get_preset(preset_key: str) -> SimulationPreset:
    """Retrieves simulation preset by key name."""
    if preset_key not in PRESETS:
        raise KeyError(
            f"Preset '{preset_key}' not found. Choices: {list(PRESETS.keys())}"
        )
    return PRESETS[preset_key]


# ======================================================================
# STATE INITIALIZATION & CONSERVATION AUDITING
# ======================================================================


def state_vector_initialization_woqr(
    n: int,
    rho_l: float,
    u_l: float,
    p_l: float,
    rho_r: float,
    u_r: float,
    p_r: float,
    gamma: float,
    divider: float,
) -> np.ndarray:
    """
    Initializes spatial conservative state vector for non-reactive flow.
    """
    u_prim = np.zeros((3, n))
    ratio = divider / 100.0 if divider > 1.0 else divider
    m_i = int(np.round(ratio * n))
    m_i = max(1, min(n - 1, m_i))

    u_prim[0, :m_i] = rho_l
    u_prim[1, :m_i] = u_l
    u_prim[2, :m_i] = p_l

    u_prim[0, m_i:] = rho_r
    u_prim[1, m_i:] = u_r
    u_prim[2, m_i:] = p_r

    return uprim_to_uconsv(u_prim, gamma)


def state_vector_initialization_wqr(
    n: int,
    rho_l: float,
    u_l: float,
    p_l: float,
    rho_r: float,
    u_r: float,
    p_r: float,
    lambda_init: np.ndarray,
    divider: float,
) -> np.ndarray:
    """Initialize the spatial conservative state for reactive flow."""
    u_prim = np.zeros((3, n))
    ratio = divider / 100.0 if divider > 1.0 else divider
    m_i = int(np.round(ratio * n))
    m_i = max(1, min(n - 1, m_i))

    u_prim[0, :m_i] = rho_l
    u_prim[1, :m_i] = u_l
    u_prim[2, :m_i] = p_l

    u_prim[0, m_i:] = rho_r
    u_prim[1, m_i:] = u_r
    u_prim[2, m_i:] = p_r

    gamma = gamma_calc(lambda_init)
    return uprim_to_uconsv(u_prim, gamma)


def compute_conservation_audit(
    u_initial: np.ndarray,
    u_final: np.ndarray,
    dx: float,
    execution_time_sec: float,
    total_steps: int,
    boundary_integral=None,
    source_integral=None,
) -> Dict[str, Any]:
    """Computes changes and balance residuals accounting for integrated
    outward flux and sources.
    """
    initial_mass = float(np.sum(u_initial[0, :]) * dx)
    final_mass = float(np.sum(u_final[0, :]) * dx)
    mass_err = float((final_mass - initial_mass) / (initial_mass + 1e-15))

    initial_mom = float(np.sum(u_initial[1, :]) * dx)
    final_mom = float(np.sum(u_final[1, :]) * dx)

    initial_energy = float(np.sum(u_initial[2, :]) * dx)
    final_energy = float(np.sum(u_final[2, :]) * dx)

    return {
        "execution_time_sec": round(execution_time_sec, 4),
        "total_time_steps": total_steps,
        "initial_mass_kg": initial_mass,
        "final_mass_kg": final_mass,
        "mass_change_rel": mass_err,
        "mass_conservation_error_rel": (
            None
            if boundary_integral is None
            else float(
                (final_mass - initial_mass + boundary_integral[0])
                / max(abs(initial_mass), 1e-15)
            )
        ),
        "balance_residual": (
            None
            if boundary_integral is None
            else (
                np.sum(u_final - u_initial, axis=1) * dx
                + np.asarray(boundary_integral)
                - (
                    np.zeros(3)
                    if source_integral is None
                    else np.asarray(source_integral)
                )
            ).tolist()
        ),
        "initial_momentum": initial_mom,
        "final_momentum": final_mom,
        "initial_energy_j": initial_energy,
        "final_energy_j": final_energy,
    }


# ======================================================================
# MAIN SIMULATION RUNNERS
# ======================================================================


def physical_state(state):
    return (
        np.isfinite(state).all()
        and np.all(state[0] > 1e-12)
        and np.all(
            state[-1] - 0.5 * np.sum(state[1:-1] ** 2, axis=0) / state[0]
            > 1e-12
        )
    )


def hydro_dt_1d(state, gamma, dx, cfl):
    prim = uconsv_to_uprim(state, gamma)
    return cfl * dx / np.max(np.abs(prim[1]) + sound_speed_calc(prim, gamma))


def conservative_update_1d(state_bc, flux, dt, dx, gamma):
    candidate = flux_application_05t(state_bc, flux, dt, dx)
    if not physical_state(candidate):
        # Recompute every interface at first order to retain
        # conservation.
        prim = uconsv_to_uprim(state_bc, gamma)
        speed = np.abs(prim[1]) + sound_speed_calc(prim, gamma)
        alpha = np.maximum(speed[:-1], speed[1:])
        physical_flux = u_to_fu_calc(state_bc, gamma)
        flux = 0.5 * (
            physical_flux[:, :-1]
            + physical_flux[:, 1:]
            - alpha * np.diff(state_bc)
        )
        candidate = flux_application_05t(state_bc, flux, dt, dx)
    if not physical_state(candidate):
        raise RuntimeError("Hydrodynamic update produced a nonphysical state")
    return candidate, flux


def validate_grid(length, cells, cfl, end_time):
    if (
        not isinstance(cells, (int, np.integer))
        or not np.isfinite([length, cfl, end_time]).all()
        or length <= 0
        or cells < 2
        or not 0 < cfl <= 0.5
        or end_time < 0
    ):
        raise ValueError(
            (
                "Require length > 0, cells >= 2, 0 < CFL <= 0.5 and final "
                "time >= 0"
            )
        )


def validate_initial_1d(rho_l, u_l, p_l, rho_r, u_r, p_r, gamma, divider):
    values = [rho_l, u_l, p_l, rho_r, u_r, p_r, gamma, divider]
    if (
        not np.isfinite(values).all()
        or min(rho_l, rho_r, p_l, p_r) <= 0
        or gamma <= 1
        or not 0 < divider < 100
        or divider == 1
    ):
        raise ValueError(
            (
                "Require finite primitives, positive density/pressure, gamma "
                "> 1 and divider in (0,1) or (1,100) percent"
            )
        )


def validate_options_1d(limiter, controller, variant, waves, boundary):
    options = (
        (limiter, ("VanLeer", "Albada", "MINBEE", "Superbee", "Unos")),
        (controller, ("Sound Speed Controlled", "w=0")),
        (variant, ("Variante 1 - Original", "Variante_2")),
        (waves, ("Pressure Based", "Enthalpy Based")),
        (boundary, ("Open", "Reflective", "Periodic")),
    )
    for value, allowed in options:
        if value not in allowed:
            raise ValueError(f"Unknown solver option: {value}")


def scalar_bc(q, bc):
    return np.pad(q, 1, mode="wrap" if bc == "Periodic" else "edge")


def advect_scalar_1d(q, state_old, state_new, mass_flux, dt_dx, bc):
    """Conservatively transport rho*q using the hydrodynamic mass flux
    (upwind).
    """
    padded = scalar_bc(q, bc)
    flux = mass_flux * np.where(mass_flux >= 0, padded[:-1], padded[1:])
    rho_q = state_old[0] * q - dt_dx * np.diff(flux)
    return np.clip(rho_q / np.maximum(state_new[0], 1e-12), 0.0, 1.0)


def run_simulation(
    l: float,
    n: int,
    rho_l: float,
    u_l: float,
    p_l: float,
    rho_r: float,
    u_r: float,
    p_r: float,
    gamma: float = 1.4,
    c_cfl: float = 0.5,
    t_corte: float = 0.2,
    limiter: str = "VanLeer",
    w_controler: str = "Sound Speed Controlled",
    hllc_variante: str = "Variante 1 - Original",
    wave_speed_calc_option: str = "Pressure Based",
    reflective_or_open: str = "Open",
    divider: float = 0.5,
    diagnostics: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs non-reactive 1D CFD simulation (equivalent to
    inputLoopResult.m).
    """
    validate_grid(l, n, c_cfl, t_corte)
    validate_options_1d(
        limiter,
        w_controler,
        hllc_variante,
        wave_speed_calc_option,
        reflective_or_open,
    )
    validate_initial_1d(rho_l, u_l, p_l, rho_r, u_r, p_r, gamma, divider)
    dx = l / float(n)
    u_0_wo_bc = state_vector_initialization_woqr(
        n, rho_l, u_l, p_l, rho_r, u_r, p_r, gamma, divider
    )

    if not physical_state(u_0_wo_bc):
        raise ValueError(
            "Initial state must have positive density and internal energy"
        )
    u_evolution_list = [u_0_wo_bc]
    t_list = [0.0]

    t_curr = 0.0
    boundary_integral = np.zeros(3)
    source_integral = np.zeros(3)

    while t_curr < t_corte:
        u_0_w_bc = bc_creator(u_0_wo_bc, reflective_or_open)

        s_l, s_s, s_r, wave_speed_matrix = wave_speed_estimate(
            u_0_w_bc, gamma, wave_speed_calc_option
        )
        dt = min(
            delta_t_estimate(wave_speed_matrix, dx, c_cfl),
            hydro_dt_1d(u_0_w_bc, gamma, dx, c_cfl),
        )

        if t_curr + dt > t_corte:
            dt = t_corte - t_curr

        if not np.isfinite(dt) or dt <= 1e-15:
            raise RuntimeError(
                "Time integration stalled before the requested final time"
            )

        u_1_w_bc = muscl_application(
            u_0_w_bc,
            dt,
            dx,
            gamma,
            c_cfl,
            limiter,
            w_controler,
            reflective_or_open,
        )
        f_hllc = hllc_flux_calc_for_muscl(
            u_1_w_bc, wave_speed_matrix, gamma, hllc_variante
        )
        u_next, f_hllc = conservative_update_1d(
            u_0_w_bc, f_hllc, dt, dx, gamma
        )

        boundary_integral += dt * (f_hllc[:, -1] - f_hllc[:, 0])
        t_curr += dt
        t_list.append(t_curr)
        u_evolution_list.append(u_next)
        u_0_wo_bc = u_next

    u_evolution = np.stack(u_evolution_list, axis=2)
    t_array = np.array(t_list)
    if diagnostics is not None:
        diagnostics.update(
            boundary_integral=boundary_integral,
            source_integral=source_integral,
        )

    return u_evolution, t_array


def run_simulation_wqr(
    l: float,
    n: int,
    rho_l: float,
    u_l: float,
    p_l: float,
    rho_r: float,
    u_r: float,
    p_r: float,
    lambda_init: Optional[np.ndarray] = None,
    c_cfl: float = 0.5,
    t_corte: float = 0.2,
    limiter: str = "VanLeer",
    w_controler: str = "Sound Speed Controlled",
    hllc_variante: str = "Variante 1 - Original",
    wave_speed_calc_option: str = "Pressure Based",
    reflective_or_open: str = "Open",
    divider: float = 0.5,
    kinetic_model: str = "arrhenius",
    diagnostics: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs reactive 1D CFD simulation supporting Arrhenius or 2-stage ZND
    kinetics.
    """
    validate_grid(l, n, c_cfl, t_corte)
    validate_options_1d(
        limiter,
        w_controler,
        hllc_variante,
        wave_speed_calc_option,
        reflective_or_open,
    )
    validate_initial_1d(rho_l, u_l, p_l, rho_r, u_r, p_r, 1.4, divider)
    if kinetic_model not in ("arrhenius", "znd"):
        raise ValueError(f"Unknown kinetic model: {kinetic_model}")
    dx = l / float(n)
    if lambda_init is None:
        lambda_init = np.ones(n)
    lambda_init = np.asarray(lambda_init, dtype=float)
    if (
        lambda_init.shape != (n,)
        or not np.isfinite(lambda_init).all()
        or np.any((lambda_init < 0) | (lambda_init > 1))
    ):
        raise ValueError("lambda_init must have one value in [0,1] per cell")

    u_0_wo_bc = state_vector_initialization_wqr(
        n, rho_l, u_l, p_l, rho_r, u_r, p_r, lambda_init, divider
    )

    if not physical_state(u_0_wo_bc):
        raise ValueError(
            "Initial state must have positive density and internal energy"
        )
    u_evolution_list = [u_0_wo_bc]
    lambda_evolution_list = [lambda_init]
    t_list = [0.0]

    t_curr = 0.0
    boundary_integral = np.zeros(3)
    source_integral = np.zeros(3)
    lambda_curr = np.copy(lambda_init)
    beta_curr = np.ones(n)

    while t_curr < t_corte:
        u_0_w_bc = bc_creator(u_0_wo_bc, reflective_or_open)
        lambda_w_bc = scalar_bc(lambda_curr, reflective_or_open)
        gamma_w_bc = gamma_calc(lambda_w_bc)

        s_l, s_s, s_r, wave_speed_matrix = wave_speed_estimate(
            u_0_w_bc, gamma_w_bc, wave_speed_calc_option
        )
        dt = min(
            delta_t_estimate(wave_speed_matrix, dx, c_cfl),
            hydro_dt_1d(u_0_w_bc, gamma_w_bc, dx, c_cfl),
        )

        if t_curr + dt > t_corte:
            dt = t_corte - t_curr

        if not np.isfinite(dt) or dt <= 1e-15:
            raise RuntimeError(
                "Time integration stalled before the requested final time"
            )

        # Formal 2nd-order Strang operator splitting: R(dt/2) -> H(dt)
        # -> R(dt/2)
        # Chemistry can raise sound speed; enforce CFL on the heated
        # state too.
        for _ in range(60):
            dt_r_half = 0.5 * dt
            lambda_half, u_chem_1, beta_half = new_lambda_n_e_calc(
                lambda_curr,
                dt_r_half,
                u_0_wo_bc,
                model=kinetic_model,
                beta_n=beta_curr,
            )
            allowed = hydro_dt_1d(u_chem_1, gamma_calc(lambda_half), dx, c_cfl)
            if dt <= allowed * (1.0 + 1e-12):
                break
            dt = min(0.5 * dt, allowed)
        else:
            raise RuntimeError("Could not find a CFL-stable reactive step")

        # Step 2: Hydrodynamics full-step (dt)
        u_chem_1_bc = bc_creator(u_chem_1, reflective_or_open)
        lambda_half_bc = scalar_bc(lambda_half, reflective_or_open)
        gamma_half_bc = gamma_calc(lambda_half_bc)

        s_l, s_s, s_r, wave_speed_matrix = wave_speed_estimate(
            u_chem_1_bc, gamma_half_bc, wave_speed_calc_option
        )
        u_1_w_bc = muscl_application_wqr(
            u_chem_1_bc,
            dt,
            dx,
            lambda_half_bc,
            c_cfl,
            limiter,
            w_controler,
            reflective_or_open,
        )
        f_hllc = hllc_flux_calc_for_muscl(
            u_1_w_bc, wave_speed_matrix, gamma_half_bc, hllc_variante
        )
        u_hydro, f_hllc = conservative_update_1d(
            u_chem_1_bc, f_hllc, dt, dx, gamma_half_bc
        )

        lambda_half = advect_scalar_1d(
            lambda_half,
            u_chem_1,
            u_hydro,
            f_hllc[0],
            dt / dx,
            reflective_or_open,
        )
        beta_half = advect_scalar_1d(
            beta_half,
            u_chem_1,
            u_hydro,
            f_hllc[0],
            dt / dx,
            reflective_or_open,
        )

        # Step 3: Chemistry second half-step (dt/2)
        lambda_np1, u_next, beta_np1 = new_lambda_n_e_calc(
            lambda_half,
            dt_r_half,
            u_hydro,
            model=kinetic_model,
            beta_n=beta_half,
        )

        boundary_integral += dt * (f_hllc[:, -1] - f_hllc[:, 0])
        source_integral += dx * np.sum(
            u_chem_1 - u_0_wo_bc + u_next - u_hydro, axis=1
        )
        t_curr += dt
        t_list.append(t_curr)
        u_evolution_list.append(u_next)
        lambda_evolution_list.append(lambda_np1)

        u_0_wo_bc = u_next
        lambda_curr = lambda_np1
        beta_curr = beta_np1

    u_evolution = np.stack(u_evolution_list, axis=2)
    t_array = np.array(t_list)
    if diagnostics is not None:
        diagnostics.update(
            boundary_integral=boundary_integral,
            source_integral=source_integral,
        )
    lambda_evolution = np.stack(lambda_evolution_list, axis=0)

    return u_evolution, t_array, lambda_evolution


# Backward compatibility alias
run_simulation_wQR = run_simulation_wqr
