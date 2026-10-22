"""TORCH Analysis & Validation Module (Python)
Unifies exact Riemann solver validation (Toro), error norms, AMR
indicators,
spatial convergence study, Navier-Stokes viscous fluxes, and CSV
plotting.

Author: BFG508
"""

import os
import csv
from typing import Union, Tuple, List, Dict, Any
import numpy as np
import matplotlib

from .torchSolver import (
    uconsv_to_uprim,
    sound_speed_calc,
    run_simulation,
    PRESETS,
    get_preset,
)

# ======================================================================
# EXACT RIEMANN SOLVER (TORO SOD SHOCK TUBE)
# ======================================================================


def exact_riemann_sod(
    x_grid: np.ndarray,
    t: float = 0.2,
    x0: float = 0.5,
    gamma: float = 1.4,
    rho_l: float = 1.0,
    u_l: float = 0.0,
    p_l: float = 1.0,
    rho_r: float = 0.125,
    u_r: float = 0.0,
    p_r: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a positive-pressure ideal-gas Riemann problem; vacuum is
    unsupported.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1 or not np.isfinite(x_grid).all():
        raise ValueError("x_grid must be a finite one-dimensional array")
    if (
        not np.isfinite(
            [t, x0, gamma, rho_l, u_l, p_l, rho_r, u_r, p_r, tol]
        ).all()
        or t < 0
        or gamma <= 1
        or min(rho_l, p_l, rho_r, p_r, tol) <= 0
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError(
            (
                "Require physical Riemann states, gamma > 1, t >= 0 and "
                "positive iteration controls"
            )
        )
    if t == 0:
        rho = np.where(x_grid <= x0, rho_l, rho_r)
        u = np.where(x_grid <= x0, u_l, u_r)
        p = np.where(x_grid <= x0, p_l, p_r)
        return rho, u, p

    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)
    if u_r - u_l >= 2.0 * (a_l + a_r) / (gamma - 1.0):
        raise ValueError("Vacuum-generating Riemann states are not supported")

    p_pv = 0.5 * (p_l + p_r) - 0.125 * (u_r - u_l) * (rho_l + rho_r) * (
        a_l + a_r
    )
    p_star = max(tol, p_pv)

    def f_k(p_val, p_k, rho_k, a_k):
        if p_val > p_k:
            a_const = 2.0 / ((gamma + 1.0) * rho_k)
            b_const = (gamma - 1.0) / (gamma + 1.0) * p_k
            return (p_val - p_k) * np.sqrt(a_const / (p_val + b_const))
        else:
            return (2.0 * a_k / (gamma - 1.0)) * (
                (p_val / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0
            )

    def df_k(p_val, p_k, rho_k, a_k):
        if p_val > p_k:
            a_const = 2.0 / ((gamma + 1.0) * rho_k)
            b_const = (gamma - 1.0) / (gamma + 1.0) * p_k
            return np.sqrt(a_const / (p_val + b_const)) * (
                1.0 - 0.5 * (p_val - p_k) / (p_val + b_const)
            )
        else:
            return (1.0 / (rho_k * a_k)) * (p_val / p_k) ** (
                -(gamma + 1.0) / (2.0 * gamma)
            )

    for _ in range(max_iter):
        f_val = (
            f_k(p_star, p_l, rho_l, a_l)
            + f_k(p_star, p_r, rho_r, a_r)
            + (u_r - u_l)
        )
        df_val = df_k(p_star, p_l, rho_l, a_l) + df_k(p_star, p_r, rho_r, a_r)
        p_next = p_star - f_val / df_val
        p_next = max(1e-10, p_next)
        if abs(p_next - p_star) / (0.5 * (p_next + p_star)) < tol:
            p_star = p_next
            break
        p_star = p_next
    else:
        raise RuntimeError("Exact Riemann pressure iteration did not converge")

    u_star = 0.5 * (u_l + u_r) + 0.5 * (
        f_k(p_star, p_r, rho_r, a_r) - f_k(p_star, p_l, rho_l, a_l)
    )

    rho_out = np.zeros_like(x_grid)
    u_out = np.zeros_like(x_grid)
    p_out = np.zeros_like(x_grid)

    left_is_shock = p_star > p_l
    right_is_shock = p_star > p_r
    c_val = (gamma - 1.0) / (gamma + 1.0)

    # Left wave speeds and star density
    if left_is_shock:
        s_shock_l = u_l - a_l * np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * (p_star / p_l)
            + (gamma - 1.0) / (2.0 * gamma)
        )
        rho_star_l = (
            rho_l * (p_star / p_l + c_val) / (1.0 + c_val * p_star / p_l)
        )
        s_head_l = s_shock_l
        s_tail_l = s_shock_l
    else:
        s_head_l = u_l - a_l
        a_star_l = a_l * (p_star / p_l) ** ((gamma - 1.0) / (2.0 * gamma))
        s_tail_l = u_star - a_star_l
        rho_star_l = rho_l * (p_star / p_l) ** (1.0 / gamma)

    # Right wave speeds and star density
    if right_is_shock:
        s_shock_r = u_r + a_r * np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * (p_star / p_r)
            + (gamma - 1.0) / (2.0 * gamma)
        )
        rho_star_r = (
            rho_r * (p_star / p_r + c_val) / (1.0 + c_val * p_star / p_r)
        )
        s_head_r = s_shock_r
        s_tail_r = s_shock_r
    else:
        s_head_r = u_r + a_r
        a_star_r = a_r * (p_star / p_r) ** ((gamma - 1.0) / (2.0 * gamma))
        s_tail_r = u_star + a_star_r
        rho_star_r = rho_r * (p_star / p_r) ** (1.0 / gamma)

    s_contact = u_star
    xi = (x_grid - x0) / t

    for i, s_val in enumerate(xi):
        if s_val < s_contact:
            # Left of contact discontinuity
            if left_is_shock:
                if s_val < s_shock_l:
                    rho_out[i], u_out[i], p_out[i] = rho_l, u_l, p_l
                else:
                    rho_out[i], u_out[i], p_out[i] = rho_star_l, u_star, p_star
            else:
                if s_val < s_head_l:
                    rho_out[i], u_out[i], p_out[i] = rho_l, u_l, p_l
                elif s_val <= s_tail_l:
                    u_fan = (2.0 / (gamma + 1.0)) * (
                        a_l + 0.5 * (gamma - 1.0) * u_l + s_val
                    )
                    a_fan = (2.0 / (gamma + 1.0)) * (
                        a_l + 0.5 * (gamma - 1.0) * (u_l - s_val)
                    )
                    rho_out[i] = rho_l * (a_fan / a_l) ** (2.0 / (gamma - 1.0))
                    u_out[i] = u_fan
                    p_out[i] = p_l * (a_fan / a_l) ** (
                        (2.0 * gamma) / (gamma - 1.0)
                    )
                else:
                    rho_out[i], u_out[i], p_out[i] = rho_star_l, u_star, p_star
        else:
            # Right of contact discontinuity
            if right_is_shock:
                if s_val < s_shock_r:
                    rho_out[i], u_out[i], p_out[i] = rho_star_r, u_star, p_star
                else:
                    rho_out[i], u_out[i], p_out[i] = rho_r, u_r, p_r
            else:
                if s_val <= s_tail_r:
                    rho_out[i], u_out[i], p_out[i] = rho_star_r, u_star, p_star
                elif s_val < s_head_r:
                    u_fan = (2.0 / (gamma + 1.0)) * (
                        -a_r + 0.5 * (gamma - 1.0) * u_r + s_val
                    )
                    a_fan = (2.0 / (gamma + 1.0)) * (
                        a_r - 0.5 * (gamma - 1.0) * (u_r - s_val)
                    )
                    rho_out[i] = rho_r * (a_fan / a_r) ** (2.0 / (gamma - 1.0))
                    u_out[i] = u_fan
                    p_out[i] = p_r * (a_fan / a_r) ** (
                        (2.0 * gamma) / (gamma - 1.0)
                    )
                else:
                    rho_out[i], u_out[i], p_out[i] = rho_r, u_r, p_r

    return rho_out, u_out, p_out


# Backward compatibility alias
exact_riemann_solver = exact_riemann_sod


# ======================================================================
# ERROR NORMS & AMR INDICATORS
# ======================================================================


def compute_error_norms(
    u_num: np.ndarray, u_exact: np.ndarray
) -> tuple[float, float, float]:
    """Calculates L1, L2, and L_inf discrete error norms."""
    diff = np.abs(u_num - u_exact)
    l1 = float(np.mean(diff))
    l2 = float(np.sqrt(np.mean(diff**2)))
    linf = float(np.max(diff))
    return l1, l2, linf


def compute_gradient_indicator(q: np.ndarray) -> np.ndarray:
    """Computes second-derivative gradient indicator for shock/wave
    detection.
    """
    n = len(q)
    indicator = np.zeros(n)
    if n < 3:
        return indicator

    q_padded = np.pad(q, (1, 1), mode="edge")
    num = np.abs(q_padded[2:] - 2.0 * q_padded[1:-1] + q_padded[:-2])
    denom = (
        np.abs(q_padded[2:] - q_padded[1:-1])
        + np.abs(q_padded[1:-1] - q_padded[:-2])
        + 1e-12
    )
    return num / denom


def get_amr_refinement_mask(
    u_prim: np.ndarray, threshold: float = 0.2
) -> np.ndarray:
    """Return a boolean mask marking cells for mesh refinement."""
    rho_ind = compute_gradient_indicator(u_prim[0, :])
    p_ind = compute_gradient_indicator(u_prim[2, :])
    return (rho_ind > threshold) | (p_ind > threshold)


# ======================================================================
# GRID REFINEMENT CONVERGENCE STUDY
# ======================================================================


def run_convergence_study(
    grid_sizes: List[int],
    t_end: float = 0.2,
    cfl: float = 0.5,
    limiter: str = "VanLeer",
    out_dir: str = "out",
) -> Dict[str, Any]:
    """Executes spatial grid refinement study and computes error
    convergence order.
    """
    results = []

    print("=== Spatial Mesh Refinement Convergence Study ===")
    print(f"Time: {t_end}s, CFL: {cfl}, Limiter: {limiter}")
    print(
        (
            f"{'N Cells':>8} | {'L1 Error':>12} | {'L2 Error':>12} | "
            f"{'L_inf Error':>12} | {'Order L1':>10}"
        )
    )
    print("-" * 65)

    prev_l1 = None
    prev_n = None

    for n in grid_sizes:
        u_evo, _ = run_simulation(
            l=1.0,
            n=n,
            rho_l=1.0,
            u_l=0.0,
            p_l=1.0,
            rho_r=0.125,
            u_r=0.0,
            p_r=0.1,
            gamma=1.4,
            c_cfl=cfl,
            t_corte=t_end,
            limiter=limiter,
            wave_speed_calc_option="Pressure Based",
            reflective_or_open="Open",
        )

        u_final = u_evo[:, :, -1]
        u_prim = uconsv_to_uprim(u_final, 1.4)
        rho_num = u_prim[0, :]

        dx = 1.0 / n
        x_grid = np.linspace(0.5 * dx, 1.0 - 0.5 * dx, n)
        rho_ex, _, _ = exact_riemann_sod(x_grid, t=t_end)

        l1, l2, linf = compute_error_norms(rho_num, rho_ex)

        order_l1 = 0.0
        if prev_l1 is not None and prev_n is not None:
            order_l1 = float(np.log(prev_l1 / l1) / np.log(n / prev_n))

        prev_l1 = l1
        prev_n = n

        print(
            (
                f"{n:>8d} | {l1:>12.5e} | {l2:>12.5e} | {linf:>12.5e} | "
                f"{order_l1:>10.2f}"
            )
        )

        results.append(
            {
                "n_cells": n,
                "dx": dx,
                "l1_error": l1,
                "l2_error": l2,
                "linf_error": linf,
                "observed_order_l1": round(order_l1, 3),
            }
        )

    return {"study_results": results}


# ======================================================================
# NAVIER-STOKES VISCOUS & HEAT FLUXES
# ======================================================================


def compute_viscous_heat_fluxes(
    u_prim: np.ndarray,
    dx: float,
    mu: float = 1e-4,
    k_heat: float = 1e-4,
    gamma: float = 1.4,
    r_gas: float = 287.05,
) -> np.ndarray:
    """
    Computes 1D Navier-Stokes viscous dissipation and Fourier heat flux.
    """
    n = u_prim.shape[1]
    flux_visc = np.zeros((3, n - 1))

    u_vel = u_prim[1, :]
    p = u_prim[2, :]
    rho = np.maximum(u_prim[0, :], 1e-12)
    t_gas = p / (rho * r_gas)

    du_dx = (u_vel[1:] - u_vel[:-1]) / dx
    dt_dx = (t_gas[1:] - t_gas[:-1]) / dx
    u_inter = 0.5 * (u_vel[1:] + u_vel[:-1])

    tau_xx = (4.0 / 3.0) * mu * du_dx
    q_x = -k_heat * dt_dx

    flux_visc[1, :] = -tau_xx
    flux_visc[2, :] = -u_inter * tau_xx + q_x
    return flux_visc


# ======================================================================
# CSV PLOTTING UTILITY
# ======================================================================


def plot_csv(
    csv_path: str,
    save_prefix: str = "out/sod_plot",
    overlay_exact: bool = False,
    t_exact: float = 0.2,
    exact_kwargs=None,
    slice_axis: str = "z",
    slice_index=None,
) -> tuple[str, str]:
    """Export 1D/2D profiles at 600 DPI or 3D plane slices at 200 DPI
    (PNG/SVG).
    Overlays Toro's exact Riemann solution if applicable.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as source:
        is_3d = "z" in (csv.DictReader(source).fieldnames or [])
    if is_3d:
        from .torchPlot3D import plot_volume_csv

        return plot_volume_csv(csv_path, save_prefix, slice_axis, slice_index)

    # Use non-interactive backend
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_path = os.fspath(save_prefix)

    x_vals, y_vals, p_vals, rho_vals, u_vals, v_vals, lambda_vals = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    has_lambda = False
    is_2d = False

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fnames = reader.fieldnames or []
        is_2d = "y" in fnames
        has_lambda = "lambda" in fnames
        for row in reader:
            x_vals.append(float(row["x"]))
            p_vals.append(float(row["p"]))
            rho_vals.append(float(row["rho"]))
            u_vals.append(float(row["u"]))
            if is_2d:
                y_vals.append(float(row["y"]))
                v_vals.append(float(row["v"]) if "v" in row else 0.0)
            if has_lambda:
                lambda_vals.append(float(row["lambda"]))

    if not x_vals:
        raise ValueError("CSV contains no simulation data")
    if not np.isfinite(
        x_vals + y_vals + p_vals + rho_vals + u_vals + v_vals + lambda_vals
    ).all():
        raise ValueError("CSV data must be finite")

    out_dir = os.path.dirname(base_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    png_path = f"{base_path}.png"
    svg_path = f"{base_path}.svg"

    if is_2d:
        # 2D Contour Plotting
        x_unique = np.unique(x_vals)
        y_unique = np.unique(y_vals)
        nx = len(x_unique)
        ny = len(y_unique)

        # Recover coordinates explicitly: CSV rows need not be in mesh
        # order.
        pairs = np.column_stack((x_vals, y_vals))
        if len(x_vals) != nx * ny or len(np.unique(pairs, axis=0)) != nx * ny:
            raise ValueError(
                (
                    "2D CSV must contain each cell of a complete rectangular "
                    "grid exactly once"
                )
            )
        order = np.lexsort((x_vals, y_vals))
        rho_vals = np.asarray(rho_vals)[order]
        p_vals = np.asarray(p_vals)[order]
        u_vals = np.asarray(u_vals)[order]
        v_vals = np.asarray(v_vals)[order]
        if has_lambda:
            lambda_vals = np.asarray(lambda_vals)[order]
        rho_grid = np.array(rho_vals).reshape((ny, nx))
        p_grid = np.array(p_vals).reshape((ny, nx))
        u_grid = np.array(u_vals).reshape((ny, nx))
        v_grid = np.array(v_vals).reshape((ny, nx))
        vel_mag = np.sqrt(u_grid**2 + v_grid**2)

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        X_grid, Y_grid = np.meshgrid(x_unique, y_unique)

        c0 = axs[0, 0].pcolormesh(
            X_grid, Y_grid, rho_grid, cmap="viridis", shading="auto"
        )
        axs[0, 0].set_title("Density [kg/m³]")
        axs[0, 0].set_xlabel("x [m]")
        axs[0, 0].set_ylabel("y [m]")
        fig.colorbar(c0, ax=axs[0, 0])

        c1 = axs[0, 1].pcolormesh(
            X_grid, Y_grid, p_grid, cmap="plasma", shading="auto"
        )
        axs[0, 1].set_title("Pressure [Pa]")
        axs[0, 1].set_xlabel("x [m]")
        fig.colorbar(c1, ax=axs[0, 1])

        c2 = axs[1, 0].pcolormesh(
            X_grid, Y_grid, vel_mag, cmap="inferno", shading="auto"
        )
        axs[1, 0].set_title("Velocity Magnitude [m/s]")
        axs[1, 0].set_xlabel("x [m]")
        axs[1, 0].set_ylabel("y [m]")
        fig.colorbar(c2, ax=axs[1, 0])

        if has_lambda:
            lam_grid = np.array(lambda_vals).reshape((ny, nx))
            c3 = axs[1, 1].pcolormesh(
                X_grid, Y_grid, lam_grid, cmap="magma", shading="auto"
            )
            axs[1, 1].set_title("Reactant Fraction Lambda [-]")
        else:
            c3 = axs[1, 1].pcolormesh(
                X_grid, Y_grid, u_grid, cmap="coolwarm", shading="auto"
            )
            axs[1, 1].set_title("u-Velocity [m/s]")
        axs[1, 1].set_xlabel("x [m]")
        fig.colorbar(c3, ax=axs[1, 1])

        fig.suptitle(
            f"2D CFD Simulation Results ({os.path.basename(csv_path)})",
            fontsize=14,
        )
        plt.tight_layout()
        plt.savefig(png_path, dpi=600, format="png")
        plt.savefig(svg_path, dpi=600, format="svg")
        plt.close(fig)

    else:
        # 1D Line Plotting
        num_panels = 4 if has_lambda else 3
        fig, axs = plt.subplots(
            num_panels, 1, figsize=(8, 2.5 * num_panels), sharex=True
        )

        x_min, x_max = min(x_vals), max(x_vals)

        show_exact = overlay_exact and (not has_lambda)
        if show_exact:
            x_dense = np.linspace(x_min, x_max, 1000)
            rho_ex, u_ex, p_ex = exact_riemann_sod(
                x_dense, t=t_exact, **(exact_kwargs or {})
            )

            axs[0].plot(
                x_dense, p_ex, "k--", linewidth=1.5, label="Exact (Toro)"
            )
            axs[1].plot(
                x_dense, rho_ex, "k--", linewidth=1.5, label="Exact (Toro)"
            )
            axs[2].plot(
                x_dense, u_ex, "k--", linewidth=1.5, label="Exact (Toro)"
            )

        axs[0].plot(x_vals, p_vals, "r-o", markersize=3, label="Numerical CFD")
        axs[0].set_ylabel("Pressure [Pa]")
        axs[0].set_title(
            f"1D CFD Simulation Results ({os.path.basename(csv_path)})"
        )
        axs[0].grid(True)
        axs[0].set_xlim(x_min, x_max)

        axs[1].plot(
            x_vals, rho_vals, "b-o", markersize=3, label="Numerical CFD"
        )
        axs[1].set_ylabel("Density [kg/m³]")
        axs[1].grid(True)

        axs[2].plot(x_vals, u_vals, "g-o", markersize=3, label="Numerical CFD")
        axs[2].set_ylabel("Velocity [m/s]")
        if not has_lambda:
            axs[2].set_xlabel("Position x [m]")
        axs[2].grid(True)

        if has_lambda:
            axs[3].plot(
                x_vals, lambda_vals, "m-o", markersize=3, label="Numerical CFD"
            )
            axs[3].set_ylabel("Lambda [-]")
            axs[3].set_xlabel("Position x [m]")
            axs[3].grid(True)

        if show_exact:
            for ax in axs:
                ax.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=600, format="png")
        plt.savefig(svg_path, dpi=600, format="svg")
        plt.close(fig)

    print(f"✔ High-res PNG figure (600 DPI) saved to: {png_path}")
    print(f"✔ Vector SVG figure (600 DPI) saved to: {svg_path}")

    return png_path, svg_path
