"""
TORCH CLI Driver (Python)
Command-line interface for batch CFD simulations, presets, and JSON/CSV/figure export.
Includes convergence study triggers and exact Riemann validation.

Author: BFG508
"""

import os
import time
import json
import argparse
import sys
import csv
import numpy as np

from cfdSolver import (
    run_simulation, run_simulation_wQR, compute_conservation_audit,
    PRESETS, get_preset, exact_riemann_sod, compute_error_norms,
    run_convergence_study, uconsv_to_uprim, gamma_calc, plot_csv,
    run_torch_solver_2d, get_2d_preset, run_torch_solver_3d, get_3d_preset, PRESETS_3D
)


def run_3d_cli(args, supplied):
    """Run a final-state-only 3D case and serialize its volume and balances."""
    name = args.preset or args.preset_3d
    nx = args.nx if "--nx" in supplied else 20
    ny = args.ny if "--ny" in supplied else 20
    cfg = get_3d_preset(name, nx, ny, args.nz)
    start = time.perf_counter()
    result = run_torch_solver_3d(
        preset_name=None,
        primitive_init=cfg["primitive"],
        nx=nx,
        ny=ny,
        nz=args.nz,
        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
        t_end=args.t_corte if "--t_corte" in supplied else cfg["t_end"],
        cfl=args.cfl if "--cfl" in supplied else 0.4,
        limiter=args.limiter,
        is_reactive=(
            (args.mode == "reactive") if "--mode" in supplied else cfg["is_reactive"]
        ),
        kinetic_model=args.kinetic_model,
        bc_x=args.bc_x or args.bc,
        bc_y=args.bc_y or args.bc,
        bc_z=args.bc_z or args.bc,
    )
    fields = ["x", "y", "z", "p", "rho", "u", "v", "w"]
    prim = result["u_prim_final"]
    arrays = [
        result["X"],
        result["Y"],
        result["Z"],
        prim[4],
        prim[0],
        prim[1],
        prim[2],
        prim[3],
    ]
    for label, key in (("lambda", "lambda_final"), ("beta", "beta_final")):
        if result[key] is not None:
            fields.append(label)
            arrays.append(result[key])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(zip(*(a.ravel() for a in arrays)))
    if args.json_out:
        meta = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in result.items()
            if key
            not in (
                "u_final",
                "u_prim_final",
                "lambda_final",
                "beta_final",
                "x",
                "y",
                "z",
                "X",
                "Y",
                "Z",
            )
        }
        meta.update(
            dimension="3d",
            preset=name,
            nx=nx,
            ny=ny,
            nz=args.nz,
            lx=args.lx,
            ly=args.ly,
            lz=args.lz,
            execution_time_sec=time.perf_counter() - start,
        )
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, allow_nan=False)
    if args.plot:
        plot_csv(
            args.out,
            args.plot,
            slice_axis=args.slice_axis,
            slice_index=args.slice_index,
        )
    print(
        f"3D simulation completed: t={result['time']:.6g}, steps={result['steps']}, CSV={args.out}"
    )


def main() -> None:
    """Main entry point for CLI simulation execution."""
    parser = argparse.ArgumentParser(
        description="TORCH CFD Solver 1D/2D/3D (Euler + MUSCL + HLLC + Arrhenius / ZND)"
    )
    parser.add_argument(
        "--dim", choices=["1d", "2d", "3d"], default="1d",
        help="Spatial dimension: 1d, 2d or 3d"
    )
    parser.add_argument(
        "--study", choices=["convergence"], default=None,
        help="Run spatial mesh refinement convergence study"
    )
    parser.add_argument(
        "--preset", type=str, default=None,
        help="Use a preconfigured 1D, 2D or 3D simulation benchmark profile"
    )
    parser.add_argument(
        "--preset_2d", choices=["2d_explosion", "2d_implosion", "2d_riemann_quadrant", "2d_shock_tube", "2d_rde_chamber"],
        default="2d_explosion",
        help="Preconfigured 2D benchmark preset (used when --dim 2d)"
    )
    parser.add_argument(
        "--nx", type=int, default=50,
        help="Number of spatial grid cells in x (for 2D)"
    )
    parser.add_argument(
        "--ny", type=int, default=50,
        help="Number of spatial grid cells in y (for 2D)"
    )
    parser.add_argument(
        "--lx", type=float, default=1.0,
        help="Domain length along x in meters"
    )
    parser.add_argument(
        "--ly", type=float, default=1.0,
        help="Domain length along y in meters (for 2D)"
    )
    parser.add_argument(
        "--mode", choices=["sod", "reactive"], default="sod",
        help="Simulation mode (sod / reactive)"
    )
    parser.add_argument(
        "--kinetic_model", choices=["arrhenius", "znd"], default="arrhenius",
        help="Chemical kinetic mechanism for reactive mode (arrhenius / znd)"
    )
    parser.add_argument(
        "--exact_val", action="store_true",
        help="Compare Sod simulation against exact Toro Riemann solution"
    )
    parser.add_argument(
        "--n", type=int, default=100,
        help="Number of spatial grid cells (for 1D)"
    )
    parser.add_argument(
        "--l", type=float, default=1.0,
        help="Domain length in meters (for 1D)"
    )
    parser.add_argument(
        "--cfl", type=float, default=0.5,
        help="Courant (CFL) number"
    )
    parser.add_argument(
        "--t_corte", type=float, default=0.2,
        help="Final simulation time (s)"
    )
    parser.add_argument(
        "--limiter", choices=["VanLeer", "Albada", "MINBEE", "Superbee", "Unos"], default="VanLeer",
        help="MUSCL slope limiter"
    )
    parser.add_argument(
        "--wavespeed", choices=["Pressure Based", "Enthalpy Based"], default="Pressure Based",
        help="Wave speed estimation option"
    )
    parser.add_argument(
        "--bc", choices=["Open", "Reflective", "Periodic"], default="Open",
        help="Boundary condition option"
    )
    parser.add_argument(
        "--out", type=str, default="out/sod_data.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--json_out", type=str, default=None,
        help="Output JSON metadata file path for conservation metrics"
    )
    parser.add_argument(
        "--plot", type=str, default=None,
        help="Export high-res PNG & SVG plots (prefix path, e.g. out/sod_plot)"
    )

    parser.add_argument('--inlet', nargs=4, type=float, metavar=('RHO','U','V','P'), default=[1.0,0.0,100.0,101325.0], help='2D fresh-mixture inlet primitive state in SI')
    parser.add_argument('--ambient_pressure', type=float, default=101325.0, help='Ambient pressure in Pa for 2D exit thrust')
    parser.add_argument('--preset_3d', choices=PRESETS_3D, default='3d_explosion')
    parser.add_argument('--nz', type=int, default=20, help='Grid cells along z in 3D')
    parser.add_argument('--lz', type=float, default=1.0, help='Domain length along z [m]')
    for axis in ('x','y','z'):
        parser.add_argument(f'--bc_{axis}', choices=['Open','Reflective','Periodic'], default=None,
                            help=f'3D boundary condition along {axis}; defaults to --bc')
    parser.add_argument('--slice_axis', choices=['x','y','z'], default='z', help='3D plot slice normal')
    parser.add_argument('--slice_index', type=int, default=None, help='Zero-based 3D slice index (default: middle)')
    parser.add_argument('--checkpoint',help='Atomic HDF5 checkpoint destination')
    parser.add_argument('--checkpoint_every',type=int,default=0,help='Checkpoint every N completed steps, plus the final state')
    parser.add_argument('--restart',help='Continue a versioned HDF5 checkpoint')
    parser.add_argument('--stop_after_steps',type=int,default=None,help='Pause after N additional complete steps')
    parser.add_argument('--vtk',help='Final VTK .vtr output path')
    parser.add_argument('--xdmf',help='Final XDMF output path (creates matching HDF5)')
    parser.add_argument('--series',help='Temporal output prefix; writes frames and collection indexes')
    parser.add_argument('--output_every',type=int,default=1,help='Temporal output every N steps, plus endpoints')
    parser.add_argument('--output_formats',nargs='+',choices=['vtk','xdmf'],default=['vtk','xdmf'])
    args = parser.parse_args()
    supplied = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}


    if args.checkpoint_every and not args.checkpoint:
        parser.error('--checkpoint_every requires --checkpoint')
    if ('--output_every' in supplied or '--output_formats' in supplied) and not args.series:
        parser.error('Temporal output options require --series')
    if args.checkpoint or args.restart or args.vtk or args.xdmf or args.series or args.stop_after_steps is not None:
        from torchSessionCli import run_session_cli
        try:
            run_session_cli(args,supplied,parser)
        except (ValueError,OSError) as error:
            parser.error(str(error))
        return

    if args.study and args.dim != "1d":
        parser.error("--study convergence is available only in 1D")
    if args.preset:
        valid_presets = PRESETS if args.dim == "1d" else (
            "2d_explosion", "2d_implosion", "2d_riemann_quadrant", "2d_shock_tube", "2d_rde_chamber")
        if args.dim == "3d":
            valid_presets = PRESETS_3D
        if args.preset not in valid_presets:
            parser.error(f"Unknown {args.dim} preset: {args.preset}")

    # Trigger convergence study if requested
    if args.study == "convergence":
        study_result = run_convergence_study(
            grid_sizes=[50, 100, 200, 400, 800],
            t_end=args.t_corte,
            cfl=args.cfl,
            limiter=args.limiter,
            out_dir=os.path.dirname(args.out) or "out"
        )
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as output:
            json.dump(study_result, output, indent=2)
        return

    if args.dim == '3d':
        if args.exact_val or '--inlet' in supplied or '--ambient_pressure' in supplied:
            parser.error('3D uses slices for validation and supports no prescribed inlet/thrust model')
        run_3d_cli(args, supplied)
        return

    # Handle 2D Execution
    if args.dim == "2d":
        p_name = args.preset if (args.preset and args.preset.startswith("2d_")) else args.preset_2d
        print(f"=== Running 2D CFD Simulation (Preset: {p_name}) ===")
        print(f"Grid: {args.nx} x {args.ny}, Domain: [{args.lx}, {args.ly}], Limiter: {args.limiter}")
        cfl_2d = args.cfl if '--cfl' in supplied else 0.45
        time_2d = args.t_corte if '--t_corte' in supplied else get_2d_preset(p_name,args.nx,args.ny)['t_end']
        t_start = time.time()
        res_2d = run_torch_solver_2d(
            preset_name=p_name,
            nx=args.nx,
            ny=args.ny,
            lx=args.lx,
            ly=args.ly,
            t_end=time_2d,
            cfl=cfl_2d,
            limiter=args.limiter,
            kinetic_model=args.kinetic_model,
            inlet_primitive=args.inlet, ambient_pressure=args.ambient_pressure
        )
        t_elapsed = time.time() - t_start

        out_dir = os.path.dirname(args.out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        u_prim = res_2d["u_prim_final"]
        lam = res_2d["lambda_final"]
        X, Y = res_2d["X"], res_2d["Y"]
        ny_grid, nx_grid = X.shape

        with open(args.out, mode='w', newline='') as f:
            writer = csv.writer(f)
            if lam is not None:
                writer.writerow(["x", "y", "p", "rho", "u", "v", "lambda"])
                for j in range(ny_grid):
                    for i in range(nx_grid):
                        writer.writerow([X[j, i], Y[j, i], u_prim[3, j, i], u_prim[0, j, i], u_prim[1, j, i], u_prim[2, j, i], lam[j, i]])
            else:
                writer.writerow(["x", "y", "p", "rho", "u", "v"])
                for j in range(ny_grid):
                    for i in range(nx_grid):
                        writer.writerow([X[j, i], Y[j, i], u_prim[3, j, i], u_prim[0, j, i], u_prim[1, j, i], u_prim[2, j, i]])

        print(f"✔ 2D Simulation completed in {t_elapsed:.3f} s ({res_2d['steps']} time steps).")
        print(f"✔ 2D Data exported to: {args.out}")

        if args.json_out:
            meta_2d = {
                "dimension": "2d",
                "preset": p_name,
                "nx": args.nx,
                "ny": args.ny,
                "t_end": res_2d["time"],
                "steps": res_2d["steps"],
                "mass_conservation_error": res_2d["mass_conservation_error"],
                "mass_change_rel": res_2d["mass_change_rel"],
                "net_boundary_mass": res_2d["net_boundary_mass"],
                "exit_thrust_per_depth": res_2d["exit_thrust_per_depth"],
                "execution_time_sec": t_elapsed
            }
            json_dir = os.path.dirname(args.json_out)
            if json_dir and not os.path.exists(json_dir):
                os.makedirs(json_dir, exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as jf:
                json.dump(meta_2d, jf, indent=2)
            print(f"✔ Metadata saved to: {args.json_out}")

        if args.plot:
            plot_csv(args.out, save_prefix=args.plot)

        return

    # Load preset if specified
    if args.preset:
        preset_cfg = get_preset(args.preset)
        mode = preset_cfg.mode
        n_cells = args.n if "--n" in supplied else preset_cfg.n_cells
        t_corte = args.t_corte if "--t_corte" in supplied else preset_cfg.t_end
        cfl = args.cfl if "--cfl" in supplied else preset_cfg.cfl
        limiter = args.limiter if "--limiter" in supplied else preset_cfg.limiter
        rho_l, u_l, p_l = preset_cfg.rho_l, preset_cfg.u_l, preset_cfg.p_l
        rho_r, u_r, p_r = preset_cfg.rho_r, preset_cfg.u_r, preset_cfg.p_r
        print(f"Loaded Preset: '{preset_cfg.name}' - {preset_cfg.description}")
    else:
        mode = args.mode
        n_cells = args.n
        t_corte = args.t_corte
        cfl = args.cfl
        limiter = args.limiter
        if mode == "sod":
            rho_l, u_l, p_l = 1.0, 0.0, 1.0
            rho_r, u_r, p_r = 0.125, 0.0, 0.1
        else:
            rho_l, u_l, p_l = 1.0, 0.0, 50.0 * 101325.0
            rho_r, u_r, p_r = 0.125, 0.0, 0.1 * 101325.0

    print(f"=== Running 1D CFD Simulation (Mode: {mode.upper()}) ===")
    print(f"Cells: {n_cells}, Length: {args.l} m, CFL: {cfl}, Final time: {t_corte} s")
    print(f"Limiter: {limiter}, Wave Speed: {args.wavespeed}, Boundary: {args.bc}")
    if mode == "reactive":
        print(f"Kinetic Mechanism: {args.kinetic_model.upper()}")

    t_start = time.time()

    diagnostics = {}
    if mode == "sod":
        gamma = 1.4
        u_evo, t_arr = run_simulation(
            l=args.l, n=n_cells,
            rho_l=rho_l, u_l=u_l, p_l=p_l,
            rho_r=rho_r, u_r=u_r, p_r=p_r,
            gamma=gamma, c_cfl=cfl, t_corte=t_corte,
            limiter=limiter, wave_speed_calc_option=args.wavespeed,
            reflective_or_open=args.bc, divider=0.5, diagnostics=diagnostics
        )
        u_final_consv = u_evo[:, :, -1]
        u_final_prim = uconsv_to_uprim(u_final_consv, gamma)
        lambda_evo = None

    else:
        lambda_init = np.ones(n_cells)
        u_evo, t_arr, lambda_evo = run_simulation_wQR(
            l=args.l, n=n_cells,
            rho_l=rho_l, u_l=u_l, p_l=p_l,
            rho_r=rho_r, u_r=u_r, p_r=p_r,
            lambda_init=lambda_init, c_cfl=cfl, t_corte=t_corte,
            limiter=limiter, wave_speed_calc_option=args.wavespeed,
            reflective_or_open=args.bc, divider=0.5,
            kinetic_model=args.kinetic_model, diagnostics=diagnostics
        )
        u_final_consv = u_evo[:, :, -1]
        gamma_final = gamma_calc(lambda_evo[-1, :])
        u_final_prim = uconsv_to_uprim(u_final_consv, gamma_final)

    t_elapsed = time.time() - t_start

    dx = args.l / float(n_cells)
    x_coords = np.linspace(dx / 2.0, args.l - dx / 2.0, n_cells)

    exact_split = max(1,min(n_cells-1,int(np.round(0.5*n_cells)))) * dx
    rho = u_final_prim[0, :]
    u_vel = u_final_prim[1, :]
    p_pres = u_final_prim[2, :]

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, mode='w', newline='') as f:
        writer = csv.writer(f)
        if mode == "reactive":
            writer.writerow(["x", "p", "rho", "u", "lambda"])
            lambda_final = lambda_evo[-1, :]
            for i in range(n_cells):
                writer.writerow([x_coords[i], p_pres[i], rho[i], u_vel[i], lambda_final[i]])
        else:
            writer.writerow(["x", "p", "rho", "u"])
            for i in range(n_cells):
                writer.writerow([x_coords[i], p_pres[i], rho[i], u_vel[i]])

    print(f"✔ Simulation completed in {t_elapsed:.3f} s ({len(t_arr)-1} time steps).")
    print(f"✔ Data exported to: {args.out}")

    # Exact Riemann validation for Sod mode if requested
    exact_metrics = None
    if mode == "sod" and args.exact_val:
        rho_exact, _, _ = exact_riemann_sod(x_coords, t_arr[-1], x0=exact_split, rho_l=rho_l, u_l=u_l, p_l=p_l, rho_r=rho_r, u_r=u_r, p_r=p_r)
        l1_e, l2_e, linf_e = compute_error_norms(rho, rho_exact)
        exact_metrics = {"l1_error": l1_e, "l2_error": l2_e, "linf_error": linf_e}
        print(f"✔ Exact Riemann Validation (Sod): L1 Error = {l1_e:.6e}, L2 Error = {l2_e:.6e}")

    # Export JSON metadata if requested
    if args.json_out:
        audit_metrics = compute_conservation_audit(
            u_evo[:, :, 0], u_evo[:, :, -1], dx, t_elapsed, len(t_arr)-1, **diagnostics
        )
        meta = {
            "mode": mode,
            "cells": n_cells,
            "cfl": cfl,
            "t_end": float(t_arr[-1]),
            "limiter": limiter,
            "audit": audit_metrics,
            "exact_validation": exact_metrics
        }
        json_dir = os.path.dirname(args.json_out)
        if json_dir and not os.path.exists(json_dir):
            os.makedirs(json_dir, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as jf:
            json.dump(meta, jf, indent=2)
        print(f"✔ Metadata & Conservation Audit saved to: {args.json_out}")

    # Generate plot if requested
    if args.plot:
        plot_csv(args.out, save_prefix=args.plot, overlay_exact=(mode == "sod" and args.exact_val), t_exact=float(t_arr[-1]), exact_kwargs=dict(x0=exact_split, rho_l=rho_l, u_l=u_l, p_l=p_l, rho_r=rho_r, u_r=u_r, p_r=p_r))


if __name__ == "__main__":
    main()
