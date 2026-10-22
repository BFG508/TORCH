"""TORCH finite-volume solver for 2D Euler and reactive gas dynamics.

Features:
- Unsplit second-order 2D MUSCL-Hancock spatial reconstruction.
- Directional HLLC approximate Riemann solver with Roe-Einfeldt wave
  bounds and tangential momentum advection.
- Sweby-bounded TVD slope limiters: Van Leer, Van Albada, MINBEE,
  Superbee and Unos.
- Periodic (azimuthal RDE), transmissive (open) and reflective
  (slip-wall) boundary conditions.
- Symmetric reaction/hydrodynamic splitting for Arrhenius and reduced
  two-stage kinetics.
- Variable-gamma EOS and explicit SI units.
- Built-in benchmarks: cylindrical explosion, quadrant Riemann problems,
  planar shock tube and unwrapped RDE annular chamber.

Author: BFG508; foundational 1D research by Paul Augustus Aliaga Brown.
"""

from typing import Union, Optional, Dict, Any, Tuple
import numpy as np
from .torchSolver import new_lambda_n_e_calc, validate_grid, physical_state
from .torchFlux import hllc_flux as hllc_flux_1d_slice
from .torchFlux import hllc_flux_grid

# ======================================================================
# THERMODYNAMICS & PHYSICAL CONVERSIONS (2D)
# ======================================================================


def gamma_calc_2d(
    lambda_val: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Return gamma(lambda) = (4.5 - lambda) / (3.5 - lambda)."""
    return (4.5 - lambda_val) / (3.5 - lambda_val)


def uconsv_to_uprim_2d(
    u_consv: np.ndarray, gamma: Union[float, np.ndarray] = 1.4
) -> np.ndarray:
    """
    Converts 2D conservative variables [rho, rho*u, rho*v, E]
    to primitive variables [rho, u, v, p].
    Shape: (4, ny, nx) -> (4, ny, nx).
    """
    rho = np.maximum(u_consv[0], 1e-12)
    u = u_consv[1] / rho
    v = u_consv[2] / rho
    kinetic_e = 0.5 * rho * (u**2 + v**2)
    internal_e = u_consv[3] - kinetic_e
    p = np.maximum((gamma - 1.0) * internal_e, 1e-12)

    u_prim = np.empty_like(u_consv)
    u_prim[0] = rho
    u_prim[1] = u
    u_prim[2] = v
    u_prim[3] = p
    return u_prim


def uprim_to_uconsv_2d(
    u_prim: np.ndarray, gamma: Union[float, np.ndarray] = 1.4
) -> np.ndarray:
    """
    Converts 2D primitive variables [rho, u, v, p]
    to conservative variables [rho, rho*u, rho*v, E].
    Shape: (4, ny, nx) -> (4, ny, nx).
    """
    rho = np.maximum(u_prim[0], 1e-12)
    u = u_prim[1]
    v = u_prim[2]
    p = np.maximum(u_prim[3], 1e-12)
    kinetic_e = 0.5 * rho * (u**2 + v**2)
    internal_e = p / (gamma - 1.0)
    total_e = internal_e + kinetic_e

    u_consv = np.empty_like(u_prim)
    u_consv[0] = rho
    u_consv[1] = rho * u
    u_consv[2] = rho * v
    u_consv[3] = total_e
    return u_consv


def sound_speed_calc_2d(
    u_prim: np.ndarray, gamma: Union[float, np.ndarray] = 1.4
) -> np.ndarray:
    """Calculates 2D speed of sound a = sqrt(gamma * p / rho)."""
    rho = np.maximum(u_prim[0], 1e-12)
    p = np.maximum(u_prim[3], 1e-12)
    return np.sqrt(gamma * p / rho)


def temp_calc_2d(
    gamma: Union[float, np.ndarray],
    lambda_val: np.ndarray,
    u_consv: np.ndarray,
) -> np.ndarray:
    """Calculates temperature T (Kelvin) for 2D reactive mixture."""
    r_univ = 8314.46  # J/(kmol K), with molar masses in kg/kmol
    u_prim = uconsv_to_uprim_2d(u_consv, gamma)
    rho = np.maximum(u_prim[0], 1e-12)
    p = u_prim[3]
    p_pa = p  # SI pressure (Pa)

    m_h2 = 2.016
    m_o2 = 32.0
    m_h2o = 18.016

    x_h2 = lambda_val * (2.0 / 3.0)
    x_o2 = lambda_val * (1.0 / 3.0)
    x_h2o = 1.0 - lambda_val
    m_m = x_h2 * m_h2 + x_o2 * m_o2 + x_h2o * m_h2o

    temp = (p_pa * m_m) / (rho * r_univ)
    return temp


# ======================================================================
# TVD SLOPE LIMITERS (2D)
# ======================================================================


def compute_tvd_slopes_2d(
    arr: np.ndarray, axis: int, limiter: str = "VanLeer"
) -> np.ndarray:
    """
    Computes limited slopes along a specified axis (1 for x, 0 for y)
    for a 2D scalar or vector array with ghost cells.
    """
    lim = limiter.lower()

    if axis == 1:
        # Difference along x
        d_minus = arr[:, :, 1:-1] - arr[:, :, :-2]
        d_plus = arr[:, :, 2:] - arr[:, :, 1:-1]
    elif axis == 0:
        # Difference along y
        d_minus = arr[:, 1:-1, :] - arr[:, :-2, :]
        d_plus = arr[:, 2:, :] - arr[:, 1:-1, :]
    else:
        raise ValueError("Axis must be 0 or 1.")

    safe_d_plus = np.where(np.abs(d_plus) < 1e-14, 1e-14, d_plus)
    r = d_minus / safe_d_plus

    if lim in ("vanleer", "van_leer"):
        r_pos = np.maximum(r, 0.0)
        phi = np.where(r > 0.0, 2.0 * r_pos / (1.0 + r_pos), 0.0)
    elif lim in ("albada", "vanalbada", "van_albada"):
        r_pos = np.maximum(r, 0.0)
        phi = np.where(r > 0.0, (r_pos**2 + r_pos) / (r_pos**2 + 1.0), 0.0)
    elif lim in ("minbee", "minmod"):
        phi = np.maximum(0.0, np.minimum(r, 1.0))
    elif lim == "superbee":
        phi = np.maximum(
            0.0, np.maximum(np.minimum(2.0 * r, 1.0), np.minimum(r, 2.0))
        )
    elif lim == "unos":
        phi = np.maximum(
            0.0, np.minimum(np.minimum(2.0, 2.0 * r), (2.0 + r) / 3.0)
        )
    else:
        r_pos = np.maximum(r, 0.0)
        phi = np.where(r > 0.0, 2.0 * r_pos / (1.0 + r_pos), 0.0)

    # Return limited slope: phi(r) * d_plus
    return phi * d_plus


# ======================================================================
# BOUNDARY CONDITIONS (2D)
# ======================================================================


def apply_boundary_conditions_2d(
    u_inner: np.ndarray,
    bc_x: str = "Periodic",
    bc_y: str = "Open",
    inlet_primitive=None,
) -> np.ndarray:
    """
    Fills 2 ghost cells around 2D conservative domain (4, ny, nx).
    Returns padded array with shape (4, ny + 4, nx + 4).
    """
    n_vars, ny, nx = u_inner.shape
    u_pad = np.zeros((n_vars, ny + 4, nx + 4), dtype=u_inner.dtype)
    u_pad[:, 2: ny + 2, 2: nx + 2] = u_inner

    # X-boundaries (left/right: along axis 2)
    bc_x_lower = bc_x.lower()
    if bc_x_lower in ("periodic", "wrap"):
        u_pad[:, 2: ny + 2, 0] = u_inner[:, :, -2]
        u_pad[:, 2: ny + 2, 1] = u_inner[:, :, -1]
        u_pad[:, 2: ny + 2, -2] = u_inner[:, :, 0]
        u_pad[:, 2: ny + 2, -1] = u_inner[:, :, 1]
    elif bc_x_lower in ("open", "transmissive", "outflow"):
        u_pad[:, 2: ny + 2, 0] = u_inner[:, :, 0]
        u_pad[:, 2: ny + 2, 1] = u_inner[:, :, 0]
        u_pad[:, 2: ny + 2, -2] = u_inner[:, :, -1]
        u_pad[:, 2: ny + 2, -1] = u_inner[:, :, -1]
    elif bc_x_lower in ("reflective", "wall", "slip"):
        # Left wall
        u_pad[:, 2: ny + 2, 1] = u_inner[:, :, 0]
        u_pad[1, 2: ny + 2, 1] = -u_inner[1, :, 0]  # Invert normal u
        u_pad[:, 2: ny + 2, 0] = u_inner[:, :, 1]
        u_pad[1, 2: ny + 2, 0] = -u_inner[1, :, 1]
        # Right wall
        u_pad[:, 2: ny + 2, -2] = u_inner[:, :, -1]
        u_pad[1, 2: ny + 2, -2] = -u_inner[1, :, -1]
        u_pad[:, 2: ny + 2, -1] = u_inner[:, :, -2]
        u_pad[1, 2: ny + 2, -1] = -u_inner[1, :, -2]

    else:
        raise ValueError(f"Unknown x boundary: {bc_x}")

    # Y-boundaries (bottom/top: along axis 1)
    bc_y_lower = bc_y.lower()
    if bc_y_lower in ("periodic", "wrap"):
        u_pad[:, 0, :] = u_pad[:, ny, :]
        u_pad[:, 1, :] = u_pad[:, ny + 1, :]
        u_pad[:, -2, :] = u_pad[:, 2, :]
        u_pad[:, -1, :] = u_pad[:, 3, :]
    elif bc_y_lower == "inflow":
        # Fixed fresh-mixture inlet at y=0 and transmissive outlet at
        # y=Ly.
        inlet = np.asarray(
            (
                [1.0, 0.0, 100.0, 101325.0]
                if inlet_primitive is None
                else inlet_primitive
            ),
            dtype=float,
        )
        if (
            inlet.shape != (4,)
            or not np.isfinite(inlet).all()
            or inlet[0] <= 0
            or inlet[3] <= 0
            or inlet[2] < 0
        ):
            raise ValueError("Inlet requires [rho>0, u, v>=0, p>0] in SI")
        state = uprim_to_uconsv_2d(inlet[:, None, None], 1.4)[:, 0, 0]
        u_pad[:, :2, :] = state[:, None, None]
        u_pad[:, -2:, :] = u_pad[:, ny + 1: ny + 2, :]
    elif bc_y_lower in ("open", "transmissive", "outflow"):
        u_pad[:, 0, :] = u_pad[:, 2, :]
        u_pad[:, 1, :] = u_pad[:, 2, :]
        u_pad[:, -2, :] = u_pad[:, ny + 1, :]
        u_pad[:, -1, :] = u_pad[:, ny + 1, :]
    elif bc_y_lower in ("reflective", "wall", "slip"):
        # Bottom wall (invert normal v = index 2)
        u_pad[:, 1, :] = u_pad[:, 2, :]
        u_pad[2, 1, :] = -u_pad[2, 2, :]
        u_pad[:, 0, :] = u_pad[:, 3, :]
        u_pad[2, 0, :] = -u_pad[2, 3, :]
        # Top wall
        u_pad[:, -2, :] = u_pad[:, ny + 1, :]
        u_pad[2, -2, :] = -u_pad[2, ny + 1, :]
        u_pad[:, -1, :] = u_pad[:, ny, :]
        u_pad[2, -1, :] = -u_pad[2, ny, :]

    else:
        raise ValueError(f"Unknown y boundary: {bc_y}")

    return u_pad


# ======================================================================
# 2D CONVECTIVE TIME STEPPING (UNSPLIT MUSCL-HANCOCK)
# ======================================================================


def muscl_hancock_step_2d(
    u_inner: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    gamma: float = 1.4,
    limiter: str = "VanLeer",
    bc_x: str = "Periodic",
    bc_y: str = "Open",
    return_fluxes: bool = False,
    inlet_primitive=None,
) -> np.ndarray:
    """
    Performs one unsplit 2D MUSCL-Hancock conservative time step.
    u_inner shape: (4, ny, nx).
    Returns updated u_next shape: (4, ny, nx).
    """
    ny, nx = u_inner.shape[1], u_inner.shape[2]

    # 1. Fill ghost cells
    u_pad = apply_boundary_conditions_2d(
        u_inner, bc_x, bc_y, inlet_primitive
    )  # (4, ny+4, nx+4)

    gamma_inner = np.broadcast_to(gamma, (ny, nx))
    gpad = pad_scalar_2d(gamma_inner, bc_x, bc_y, inlet_value=1.4)
    # Reflection changes momentum only, so component zero is a scalar
    # padding.
    gamma = gpad[1: ny + 3, 1: nx + 3]

    # 2. Slopes in active cells (interior + 1 ghost layer)
    # Active range in pad: y in [1, ny+2], x in [1, nx+2]
    u_active = u_pad[:, 1: ny + 3, 1: nx + 3]  # (4, ny+2, nx+2)

    delta_x = compute_tvd_slopes_2d(
        u_pad[:, 1: ny + 3, :], axis=1, limiter=limiter
    )  # (4, ny+2, nx+2)
    delta_y = compute_tvd_slopes_2d(
        u_pad[:, :, 1: nx + 3], axis=0, limiter=limiter
    )  # (4, ny+2, nx+2)

    # 3. Half-step predictor inside each cell
    # Boundary extrapolation in x
    u_xl = u_active - 0.5 * delta_x
    u_xr = u_active + 0.5 * delta_x

    # Boundary extrapolation in y
    u_yb = u_active - 0.5 * delta_y
    u_yt = u_active + 0.5 * delta_y

    # Compute physical Euler fluxes for predictor
    def euler_flux_x(u_state):
        rho = np.maximum(u_state[0], 1e-12)
        u = u_state[1] / rho
        v = u_state[2] / rho
        p = np.maximum(
            (gamma - 1.0) * (u_state[3] - 0.5 * rho * (u**2 + v**2)), 1e-12
        )
        fx = np.empty_like(u_state)
        fx[0] = rho * u
        fx[1] = rho * u**2 + p
        fx[2] = rho * u * v
        fx[3] = u * (u_state[3] + p)
        return fx

    def euler_flux_y(u_state):
        rho = np.maximum(u_state[0], 1e-12)
        u = u_state[1] / rho
        v = u_state[2] / rho
        p = np.maximum(
            (gamma - 1.0) * (u_state[3] - 0.5 * rho * (u**2 + v**2)), 1e-12
        )
        fy = np.empty_like(u_state)
        fy[0] = rho * v
        fy[1] = rho * u * v
        fy[2] = rho * v**2 + p
        fy[3] = v * (u_state[3] + p)
        return fy

    fx_r = euler_flux_x(u_xr)
    fx_l = euler_flux_x(u_xl)
    fy_t = euler_flux_y(u_yt)
    fy_b = euler_flux_y(u_yb)

    # Predictor update to t + dt/2
    u_bar = (
        u_active
        - (0.5 * dt / dx) * (fx_r - fx_l)
        - (0.5 * dt / dy) * (fy_t - fy_b)
    )
    del u_xl, u_xr, u_yb, u_yt, fx_r, fx_l, fy_t, fy_b

    # Ensure physical positivity
    u_bar[0] = np.maximum(u_bar[0], 1e-12)
    p_bar = (gamma - 1.0) * (
        u_bar[3] - 0.5 * (u_bar[1] ** 2 + u_bar[2] ** 2) / u_bar[0]
    )
    u_bar[3] = np.where(
        p_bar < 1e-12,
        1e-12 / (gamma - 1.0)
        + 0.5 * (u_bar[1] ** 2 + u_bar[2] ** 2) / u_bar[0],
        u_bar[3],
    )

    # 4. HLLC Interface Fluxes
    # X-interfaces: interfaces between (j, i) and (j, i+1)
    # Shape: (4, ny, nx + 1)
    # Left state at interface i+1/2 is u_bar_xr at i (which is in
    # 1:ny+1, 0:nx)
    # Right state at interface i+1/2 is u_bar_xl at i+1 (which is in
    # 1:ny+1, 1:nx+1)
    u_l_x = u_bar[:, 1: ny + 1, :-1] + 0.5 * delta_x[:, 1: ny + 1, :-1]
    u_r_x = u_bar[:, 1: ny + 1, 1:] - 0.5 * delta_x[:, 1: ny + 1, 1:]
    flux_x = hllc_flux_grid(
        u_l_x,
        u_r_x,
        gamma[1: ny + 1, :-1],
        gamma[1: ny + 1, 1:],
    )
    del u_l_x, u_r_x

    # Y-interfaces: interfaces between (j, i) and (j+1, i)
    # Shape: (4, ny + 1, nx)
    # For Y interfaces: normal velocity is v (index 2) and tangential is
    # u (index 1)
    u_b_y = u_bar[:, :-1, 1: nx + 1] + 0.5 * delta_y[:, :-1, 1: nx + 1]
    u_t_y = u_bar[:, 1:, 1: nx + 1] - 0.5 * delta_y[:, 1:, 1: nx + 1]
    flux_y = hllc_flux_grid(
        u_b_y,
        u_t_y,
        gamma[:-1, 1: nx + 1],
        gamma[1:, 1: nx + 1],
        direction=1,
    )
    del u_b_y, u_t_y

    # 5. Full-step conservative update
    df_x = flux_x[:, :, 1:] - flux_x[:, :, :-1]
    df_y = flux_y[:, 1:, :] - flux_y[:, :-1, :]

    u_next = u_inner - (dt / dx) * df_x - (dt / dy) * df_y

    if not physical_state(u_next):
        # First-order local Lax-Friedrichs fallback, shared at every
        # interface.
        gp = gpad
        prim = uconsv_to_uprim_2d(u_pad, gp)
        rho, ux, vy, pressure = prim
        sound = sound_speed_calc_2d(prim, gp)
        fx = np.stack(
            (
                u_pad[1],
                u_pad[1] * ux + pressure,
                u_pad[2] * ux,
                ux * (u_pad[3] + pressure),
            )
        )
        fy = np.stack(
            (
                u_pad[2],
                u_pad[1] * vy,
                u_pad[2] * vy + pressure,
                vy * (u_pad[3] + pressure),
            )
        )
        ax = np.maximum(
            (np.abs(ux) + sound)[2:-2, 1:-2], (np.abs(ux) + sound)[2:-2, 2:-1]
        )
        ay = np.maximum(
            (np.abs(vy) + sound)[1:-2, 2:-2], (np.abs(vy) + sound)[2:-1, 2:-2]
        )
        flux_x = 0.5 * (
            fx[:, 2:-2, 1:-2]
            + fx[:, 2:-2, 2:-1]
            - ax * (u_pad[:, 2:-2, 2:-1] - u_pad[:, 2:-2, 1:-2])
        )
        flux_y = 0.5 * (
            fy[:, 1:-2, 2:-2]
            + fy[:, 2:-1, 2:-2]
            - ay * (u_pad[:, 2:-1, 2:-2] - u_pad[:, 1:-2, 2:-2])
        )
        u_next = (
            u_inner
            - dt / dx * np.diff(flux_x, axis=2)
            - dt / dy * np.diff(flux_y, axis=1)
        )
    if not physical_state(u_next):
        raise RuntimeError("Hydrodynamic update produced a nonphysical state")

    return (u_next, flux_x, flux_y) if return_fluxes else u_next


# ======================================================================
# 2D REACTIVE KINETICS & STRANG SPLITTING
# ======================================================================


def reaction_step_2d(
    u_consv: np.ndarray,
    lambda_val: np.ndarray,
    beta_val: Optional[np.ndarray],
    dt_chem: float,
    kinetic_model: str = "arrhenius",
    q_heat: float = 1.41584e8,
    activation_energy: float = 17.15 * 4184.0,
    pre_exponential: float = 1.76e5,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Solves 2D chemical kinetics reaction step R_{dt_chem}.
    Updates reactant fraction lambda, induction parameter beta, and
    total energy E.
    """
    shape = lambda_val.shape
    flat = u_consv.reshape(4, -1)
    normal = np.stack(
        (flat[0], flat[1], flat[3] - 0.5 * flat[2] ** 2 / flat[0])
    )
    lam, reacted, beta = new_lambda_n_e_calc(
        lambda_val.ravel(),
        dt_chem,
        normal,
        kinetic_model,
        None if beta_val is None else beta_val.ravel(),
        q_heat=q_heat,
        activation_energy=activation_energy,
        pre_exponential=pre_exponential,
    )
    result = np.array(u_consv, dtype=float, copy=True)
    result[3] += (reacted[2] - normal[2]).reshape(shape)
    return (
        result,
        lam.reshape(shape),
        beta.reshape(shape) if beta_val is not None else None,
    )


def compute_cfl_dt_2d(
    u_prim: np.ndarray,
    dx: float,
    dy: float,
    cfl: float = 0.45,
    gamma: float = 1.4,
) -> float:
    """Computes stable 2D Courant time step."""
    a = sound_speed_calc_2d(u_prim, gamma)
    u_abs = np.abs(u_prim[1])
    v_abs = np.abs(u_prim[2])

    max_wave_speed_x = np.max((u_abs + a) / dx)
    max_wave_speed_y = np.max((v_abs + a) / dy)
    total_speed = max_wave_speed_x + max_wave_speed_y

    if total_speed <= 1e-12:
        return 1e-3
    return float(cfl / total_speed)


# ======================================================================
# 2D PRESETS & INITIAL CONDITIONS
# ======================================================================


def get_2d_preset(
    preset_name: str, nx: int = 60, ny: int = 60
) -> Dict[str, Any]:
    """Returns initial conditions and domain parameters for standard 2D
    presets:
    - '2d_explosion': Cylindrical blast wave in a box.
    - '2d_implosion': Converging cylindrical shock.
    - '2d_riemann_quadrant': 4-quadrant contact/shock interaction
    (Lax-Liu 3).
    - '2d_shock_tube': 2D planar Sod shock tube.
    - '2d_rde_chamber': Unwrapped RDE combustion channel with spinning
    wave.
    """
    p_name = preset_name.lower()
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    X, Y = np.meshgrid(x, y)

    rho = np.ones((ny, nx))
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.ones((ny, nx))
    gamma = 1.4
    t_end = 0.2
    bc_x = "Reflective"
    bc_y = "Reflective"
    is_reactive = False

    if p_name in ("2d_explosion", "cylindrical_explosion"):
        # High pressure / high density circular core at center (0.5,
        # 0.5)
        radius = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
        rho = np.where(radius <= 0.2, 1.0, 0.125)
        p = np.where(radius <= 0.2, 1.0, 0.1)
        t_end = 0.25
        bc_x = "Transmissive"
        bc_y = "Transmissive"

    elif p_name in ("2d_implosion", "implosion"):
        # High pressure boundary, low pressure center
        radius = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
        rho = np.where(radius <= 0.2, 0.125, 1.0)
        p = np.where(radius <= 0.2, 0.14, 1.0)
        t_end = 0.35
        bc_x = "Reflective"
        bc_y = "Reflective"

    elif p_name in ("2d_riemann_quadrant", "riemann_4quadrant"):
        # Lax-Liu Configuration 3 (contact discontinuities)
        q1 = (X >= 0.5) & (Y >= 0.5)
        q2 = (X < 0.5) & (Y >= 0.5)
        q3 = (X < 0.5) & (Y < 0.5)
        q4 = (X >= 0.5) & (Y < 0.5)

        rho = np.where(q1, 0.5313, np.where(q2, 1.0, np.where(q3, 0.8, 1.0)))
        u = np.where(q1, 0.0, np.where(q2, 0.7276, np.where(q3, 0.0, 0.0)))
        v = np.where(q1, 0.0, np.where(q2, 0.0, np.where(q3, 0.0, 0.7276)))
        p = np.where(q1, 0.4, np.where(q2, 1.0, np.where(q3, 1.0, 1.0)))
        t_end = 0.25
        bc_x = "Transmissive"
        bc_y = "Transmissive"

    elif p_name in ("2d_shock_tube", "planar_shock"):
        # Planar discontinuity at x = 0.5
        rho = np.where(X <= 0.5, 1.0, 0.125)
        p = np.where(X <= 0.5, 1.0, 0.1)
        t_end = 0.2
        bc_x = "Transmissive"
        bc_y = "Reflective"

    elif p_name in ("2d_rde_chamber", "rde_detonation"):
        # Unwrapped 2D rotating detonation combustor: periodic in X,
        # open at Y top
        # High-pressure ignition hotspot at (0.2, 0.2)
        dist = np.sqrt((X - 0.2) ** 2 + (Y - 0.1) ** 2)
        rho = np.where(dist <= 0.1, 2.5, 1.0)
        u = np.where(dist <= 0.1, 500.0, 0.0)
        v = np.zeros_like(X)
        p = np.where(dist <= 0.1, 30.0, 1.0) * 101325.0
        t_end = 0.005
        bc_x = "Periodic"
        bc_y = "Inflow"
        is_reactive = True

    else:
        raise ValueError(f"Unknown 2D preset: {preset_name}")

    return {
        "rho": rho,
        "u": u,
        "v": v,
        "p": p,
        "gamma": gamma,
        "lx": 1.0,
        "ly": 1.0,
        "nx": nx,
        "ny": ny,
        "t_end": t_end,
        "bc_x": bc_x,
        "bc_y": bc_y,
        "is_reactive": is_reactive,
    }


# ======================================================================
# 2D SIMULATION DRIVER LOOP
# ======================================================================


def pad_scalar_2d(q, bc_x, bc_y, inlet_value=1.0):
    ny, nx = q.shape
    padded = apply_boundary_conditions_2d(
        np.broadcast_to(q, (4, ny, nx)).copy(),
        bc_x,
        "Open" if bc_y.lower() == "inflow" else bc_y,
    )[0]
    if bc_y.lower() == "inflow":
        padded[:2, :] = inlet_value
    return padded


def advect_scalar_2d(q, old, new, mass_x, mass_y, dt_dx, dt_dy, bc_x, bc_y):
    ny, nx = q.shape
    padded = pad_scalar_2d(q, bc_x, bc_y)
    fx = mass_x * np.where(mass_x >= 0, padded[2:-2, 1:-2], padded[2:-2, 2:-1])
    fy = mass_y * np.where(mass_y >= 0, padded[1:-2, 2:-2], padded[2:-1, 2:-2])
    rho_q = (
        old[0] * q - dt_dx * np.diff(fx, axis=1) - dt_dy * np.diff(fy, axis=0)
    )
    return np.clip(rho_q / np.maximum(new[0], 1e-12), 0.0, 1.0)


def run_torch_solver_2d(
    preset_name: Optional[str] = "2d_explosion",
    rho_init: Optional[np.ndarray] = None,
    u_init: Optional[np.ndarray] = None,
    v_init: Optional[np.ndarray] = None,
    p_init: Optional[np.ndarray] = None,
    nx: int = 50,
    ny: int = 50,
    lx: float = 1.0,
    ly: float = 1.0,
    t_end: float = 0.2,
    cfl: float = 0.45,
    gamma: float = 1.4,
    limiter: str = "VanLeer",
    bc_x: str = "Transmissive",
    bc_y: str = "Transmissive",
    is_reactive: bool = False,
    kinetic_model: str = "arrhenius",
    max_steps: int = 100000,
    inlet_primitive=None,
    ambient_pressure: float = 101325.0,
    lambda_init=None,
    beta_init=None,
) -> Dict[str, Any]:
    """Main 2D simulation driver executing 2D MUSCL-Hancock with Strang
    splitting.
    Returns dictionary with fields:
    - 'x', 'y': 1D grid coordinate arrays
    - 'X', 'Y': 2D meshgrid arrays
    - 'u_final': final conservative state array (4, ny, nx)
    - 'u_prim_final': final primitive state array (4, ny, nx)
    - 'lambda_final': final reactant fraction (ny, nx) if reactive
    - 'time': final simulation time
    - 'steps': number of elapsed time steps
    - 'mass_conservation_error': relative mass error
    """
    validate_grid(lx, nx, cfl, t_end)
    validate_grid(ly, ny, cfl, t_end)
    if not np.isscalar(gamma) or not np.isfinite(gamma) or gamma <= 1:
        raise ValueError("gamma must be finite and greater than 1")
    if not np.isfinite(ambient_pressure) or ambient_pressure < 0:
        raise ValueError("Ambient pressure must be finite and nonnegative")
    if limiter not in ("VanLeer", "Albada", "MINBEE", "Superbee", "Unos"):
        raise ValueError(f"Unknown slope limiter: {limiter}")
    initial_fields = (rho_init, u_init, v_init, p_init)
    if any(field is not None for field in initial_fields) and any(
        field is None for field in initial_fields
    ):
        raise ValueError(
            (
                "Custom initialization requires rho_init, u_init, v_init and "
                "p_init"
            )
        )
    if preset_name is None and rho_init is None:
        raise ValueError(
            "Provide a preset or all four initial primitive fields"
        )
    if preset_name is not None and rho_init is None:
        cfg = get_2d_preset(preset_name, nx, ny)
        rho_init = cfg["rho"]
        u_init = cfg["u"]
        v_init = cfg["v"]
        p_init = cfg["p"]
        gamma = cfg["gamma"]
        bc_x = cfg["bc_x"]
        bc_y = cfg["bc_y"]
        is_reactive = cfg["is_reactive"]

    dx = lx / nx
    dy = ly / ny
    x_coords = np.linspace(0.5 * dx, lx - 0.5 * dx, nx)
    y_coords = np.linspace(0.5 * dy, ly - 0.5 * dy, ny)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Initialize primitive and conservative states
    u_prim = np.empty((4, ny, nx))
    u_prim[0] = rho_init
    u_prim[1] = u_init
    u_prim[2] = v_init
    u_prim[3] = p_init

    if (
        not np.isfinite(u_prim).all()
        or np.any(u_prim[0] <= 0)
        or np.any(u_prim[3] <= 0)
    ):
        raise ValueError(
            (
                "Initial primitives must be finite with positive density and "
                "pressure"
            )
        )
    if (
        kinetic_model not in ("arrhenius", "znd")
        or not isinstance(max_steps, int)
        or max_steps < 1
    ):
        raise ValueError("Invalid kinetic model or maximum step count")
    lambda_val = None
    beta_val = None
    if is_reactive:
        lambda_val = (
            np.ones((ny, nx))
            if lambda_init is None
            else np.asarray(lambda_init, dtype=float).copy()
        )
        if (
            lambda_val.shape != (ny, nx)
            or not np.isfinite(lambda_val).all()
            or np.any((lambda_val < 0) | (lambda_val > 1))
        ):
            raise ValueError(
                "lambda_init must have one value in [0,1] per cell"
            )
        gamma = gamma_calc_2d(lambda_val)
        if kinetic_model == "znd":
            beta_val = (
                np.ones((ny, nx))
                if beta_init is None
                else np.asarray(beta_init, dtype=float).copy()
            )
            if (
                beta_val.shape != (ny, nx)
                or not np.isfinite(beta_val).all()
                or np.any((beta_val < 0) | (beta_val > 1))
            ):
                raise ValueError(
                    "beta_init must have one value in [0,1] per cell"
                )
    u_consv = uprim_to_uconsv_2d(u_prim, gamma)
    if not physical_state(u_consv):
        raise ValueError(
            (
                "Initial conservative state must have positive density and "
                "internal energy"
            )
        )
    # Validate boundaries and inlet even when the requested final time
    # is zero.
    apply_boundary_conditions_2d(u_consv, bc_x, bc_y, inlet_primitive)
    initial_total_mass = float(np.sum(u_consv[0]) * dx * dy)

    current_time = 0.0
    step_count = 0
    boundary_mass = 0.0

    while current_time < t_end and step_count < max_steps:
        gamma = gamma_calc_2d(lambda_val) if is_reactive else gamma
        state_cfl = apply_boundary_conditions_2d(
            u_consv, bc_x, bc_y, inlet_primitive
        )
        gamma_cfl = pad_scalar_2d(
            np.broadcast_to(gamma, (ny, nx)), bc_x, bc_y, inlet_value=1.4
        )
        u_prim_curr = uconsv_to_uprim_2d(state_cfl, gamma_cfl)
        dt = compute_cfl_dt_2d(u_prim_curr, dx, dy, cfl, gamma_cfl)

        if current_time + dt > t_end:
            dt = t_end - current_time

        if not np.isfinite(dt) or dt <= 1e-15:
            raise RuntimeError(
                "Time integration stalled before the requested final time"
            )

        if is_reactive:
            for _ in range(60):
                heated, lam_half, beta_half = reaction_step_2d(
                    u_consv, lambda_val, beta_val, 0.5 * dt, kinetic_model
                )
                g_half = gamma_calc_2d(lam_half)
                state_cfl = apply_boundary_conditions_2d(
                    heated, bc_x, bc_y, inlet_primitive
                )
                gp_cfl = pad_scalar_2d(g_half, bc_x, bc_y, inlet_value=1.4)
                allowed = compute_cfl_dt_2d(
                    uconsv_to_uprim_2d(state_cfl, gp_cfl), dx, dy, cfl, gp_cfl
                )
                if dt <= allowed * (1.0 + 1e-12):
                    break
                dt = min(0.5 * dt, allowed)
            else:
                raise RuntimeError("Could not find a CFL-stable reactive step")
            u_consv, lambda_val, beta_val = heated, lam_half, beta_half

        # Convective Hydrodynamic Step: Full step dt via 2D
        # MUSCL-Hancock
        gamma = gamma_calc_2d(lambda_val) if is_reactive else gamma
        old_state = u_consv
        u_consv, flux_x, flux_y = muscl_hancock_step_2d(
            u_consv,
            dx,
            dy,
            dt,
            gamma,
            limiter,
            bc_x,
            bc_y,
            return_fluxes=True,
            inlet_primitive=inlet_primitive,
        )
        boundary_mass += dt * (
            dy * np.sum(flux_x[0, :, -1] - flux_x[0, :, 0])
            + dx * np.sum(flux_y[0, -1, :] - flux_y[0, 0, :])
        )
        if is_reactive:
            lambda_val = advect_scalar_2d(
                lambda_val,
                old_state,
                u_consv,
                flux_x[0],
                flux_y[0],
                dt / dx,
                dt / dy,
                bc_x,
                bc_y,
            )
            if beta_val is not None:
                beta_val = advect_scalar_2d(
                    beta_val,
                    old_state,
                    u_consv,
                    flux_x[0],
                    flux_y[0],
                    dt / dx,
                    dt / dy,
                    bc_x,
                    bc_y,
                )

        if is_reactive:
            # Strang Splitting Step 2: Chemistry half-step dt/2
            u_consv, lambda_val, beta_val = reaction_step_2d(
                u_consv, lambda_val, beta_val, 0.5 * dt, kinetic_model
            )

        current_time += dt
        step_count += 1

    if current_time < t_end:
        raise RuntimeError(
            f"Maximum steps reached at t={current_time}; requested {t_end}"
        )
    gamma = gamma_calc_2d(lambda_val) if is_reactive else gamma
    final_total_mass = float(np.sum(u_consv[0]) * dx * dy)
    mass_err = abs(
        final_total_mass - initial_total_mass + boundary_mass
    ) / max(1e-12, initial_total_mass)
    u_prim_final = uconsv_to_uprim_2d(u_consv, gamma)

    return {
        "x": x_coords,
        "y": y_coords,
        "X": X,
        "Y": Y,
        "u_final": u_consv,
        "u_prim_final": u_prim_final,
        "lambda_final": lambda_val,
        "beta_final": beta_val,
        "time": current_time,
        "steps": step_count,
        "mass_conservation_error": mass_err,
        "mass_change_rel": (final_total_mass - initial_total_mass)
        / initial_total_mass,
        "net_boundary_mass": boundary_mass,
        "exit_thrust_per_depth": float(
            dx
            * np.sum(
                u_prim_final[0, -1] * u_prim_final[2, -1] ** 2
                + u_prim_final[3, -1]
                - ambient_pressure
            )
        ),
    }
