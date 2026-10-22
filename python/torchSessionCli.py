"""CLI adapter for checkpointed sessions; legacy workflows remain
available.
"""

import csv
import json
from pathlib import Path
import numpy as np
from cfdSolver.torchSession import Simulation
from cfdSolver.torchIO import write_vtk, write_xdmf
from cfdSolver.torchGeometry import axes as grid_axes, coordinates
from cfdSolver import get_preset, plot_csv


def run_session_cli(args, supplied, parser):
    if args.study or args.exact_val:
        parser.error(
            (
                "Session IO cannot be combined with --study or --exact_val; "
                "use numerical exports or the legacy validation workflow"
            )
        )
    if args.restart:
        configuration = {
            "--geometry",
            "--origins",
            "--face_boundaries",
            "--primitive",
            "--lambda_value",
            "--beta_value",
            "--dim",
            "--preset",
            "--preset_2d",
            "--preset_3d",
            "--n",
            "--nx",
            "--ny",
            "--nz",
            "--l",
            "--lx",
            "--ly",
            "--lz",
            "--cfl",
            "--limiter",
            "--wavespeed",
            "--bc",
            "--bc_x",
            "--bc_y",
            "--bc_z",
            "--mode",
            "--kinetic_model",
            "--inlet",
            "--ambient_pressure",
        }
        if (
            supplied & configuration
            or "--adaptive" in supplied
            or any(a.startswith("--amr_") for a in supplied)
        ):
            parser.error(
                (
                    "Restart uses saved configuration; only --t_corte may "
                    "change its target time"
                )
            )
        session = Simulation.load(
            args.restart,
            t_end=args.t_corte if "--t_corte" in supplied else None,
        )
    else:
        dim = {"1d": 1, "2d": 2, "3d": 3}[args.dim]
        options = dict(limiter=args.limiter, kinetic_model=args.kinetic_model)
        if "--limiter" not in supplied:
            options.pop("limiter")
        if "--t_corte" in supplied:
            options["t_end"] = args.t_corte
        if "--cfl" in supplied:
            options["cfl"] = args.cfl
        if dim == 1:
            name = args.preset or (
                "detonation_cj" if args.mode == "reactive" else "sod_standard"
            )
            cfg = get_preset(name)
            counts = [
                args.n if "--n" in supplied or not args.preset else cfg.n_cells
            ]
            lengths = [args.l]
            options["boundaries"] = [args.bc]
            options["wavespeed"] = args.wavespeed
            if not args.preset:
                options.setdefault("t_end", args.t_corte)
        else:
            name = args.preset or (
                args.preset_2d if dim == 2 else args.preset_3d
            )
            counts = (
                [args.nx, args.ny]
                if dim == 2
                else [
                    args.nx if "--nx" in supplied else 20,
                    args.ny if "--ny" in supplied else 20,
                    args.nz,
                ]
            )
            lengths = [args.lx, args.ly] + ([args.lz] if dim == 3 else [])
            if "--wavespeed" in supplied:
                parser.error("--wavespeed is a 1D session option")
            if "--bc" in supplied:
                options["boundaries"] = [args.bc] * dim
            axes = [args.bc_x, args.bc_y] + ([args.bc_z] if dim == 3 else [])
            if any(axes):
                from cfdSolver.torchSolver2D import get_2d_preset

                defaults = (
                    ["Open"] * dim
                    if dim == 3
                    else [
                        get_2d_preset(name, *counts)[k]
                        for k in ("bc_x", "bc_y")
                    ]
                )
                options["boundaries"] = [
                    axis or ("Open" if default == "Transmissive" else default)
                    for axis, default in zip(axes, defaults)
                ]
            if dim == 2:
                options["inlet_primitive"] = args.inlet
            if dim == 3 and "--inlet" in supplied and not args.face_boundaries:
                parser.error("3D prescribed inlet remains unsupported")
            if "--mode" in supplied:
                options["reactive"] = args.mode == "reactive"
        if "--mode" in supplied:
            options["reactive"] = args.mode == "reactive"
        options["geometry"] = args.geometry
        if args.origins is not None:
            options["origins"] = args.origins
        if args.face_boundaries:
            options["face_boundaries"] = json.loads(
                Path(args.face_boundaries).read_text()
            )
        for key, value in (
            ("lambda_init", args.lambda_value),
            ("beta_init", args.beta_value),
        ):
            if value is not None:
                options[key] = np.full(tuple(counts[::-1]), value)
        if args.primitive:
            if (
                "--preset" in supplied
                or "--preset_2d" in supplied
                or "--preset_3d" in supplied
            ):
                parser.error(
                    "--primitive and explicit presets are mutually exclusive"
                )
            if len(args.primitive) != dim + 2:
                parser.error(
                    (
                        "--primitive requires density, one local velocity per "
                        "axis, and pressure"
                    )
                )
            primitive = np.broadcast_to(
                np.asarray(args.primitive).reshape((-1,) + (1,) * dim),
                (dim + 2, *counts[::-1]),
            )
            options.setdefault("t_end", args.t_corte)
            options["reactive"] = args.mode == "reactive"
            session = Simulation(primitive, lengths, **options)
        else:
            session = Simulation.preset(
                name, counts, lengths=lengths, **options
            )
    session.set_workers(args.workers)
    if not args.restart:
        if args.adaptive:
            from cfdSolver.torchAdaptiveGeometry import settings

            base = session
            session = base.to_adaptive(
                **{key: getattr(args, "amr_" + key) for key in settings()}
            )
            base.close()
        elif any(a.startswith("--amr_") for a in supplied):
            parser.error("--amr_* options require --adaptive")
    if (
        args.plot
        and not hasattr(session, "mesh")
        and session.config.get("geometry", "cartesian") != "cartesian"
    ):
        parser.error(
            (
                "Curved-grid sessions use VTK/XDMF visualization; --plot "
                "requires Cartesian coordinates"
            )
        )
    session.run(
        step_limit=args.stop_after_steps,
        checkpoint=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        output_prefix=args.series,
        output_every=args.output_every,
        formats=args.output_formats,
    )
    if args.vtk:
        write_vtk(args.vtk, session)
    if args.xdmf:
        write_xdmf(args.xdmf, session)
    dim = session.config["dimension"]
    prim = session.primitive
    centers = [(a[:-1] + a[1:]) / 2 for a in grid_axes(session.config)]
    grids = np.meshgrid(*centers[::-1], indexing="ij")[::-1]
    if hasattr(session, "mesh"):
        grids = list(session.mesh.centers.T)
    labels = (
        coordinates(session.config)
        + ["p", "rho"]
        + (
            list("uvw"[:dim])
            if session.config.get("geometry", "cartesian") == "cartesian"
            else ["u_" + a for a in coordinates(session.config)]
        )
    )
    values = list(grids) + [prim[-1], prim[0]] + list(prim[1:-1])
    if hasattr(session, "mesh"):
        labels += ["width_" + a for a in coordinates(session.config)] + [
            "refinement_level",
            "cell_volume",
        ]
        values += list(session.mesh.widths.T) + [
            session.mesh.levels,
            session.mesh.volumes,
        ]
    for name, field in (("lambda", session.lam), ("beta", session.beta)):
        if field is not None:
            labels.append(name)
            values.append(field)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(labels)
        writer.writerows(zip(*(a.ravel() for a in values)))
    if args.json_out:
        meta = dict(
            config=session.config,
            time=session.time,
            steps=session.steps,
            completed=session.completed,
            initial_integral=session.initial_integral,
            final_integral=session.integral,
            boundary_integral=session.boundary_integral,
            source_integral=session.source_integral,
            balance_residual=session.balance_residual,
        )
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                meta,
                default=lambda x: (
                    x.tolist() if isinstance(x, np.ndarray) else x.item()
                ),
                allow_nan=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.plot and hasattr(session, "mesh"):
        from matplotlib.figure import Figure
        from desktop.view import draw_session

        figure = Figure(figsize=(9, 7))
        draw_session(
            figure,
            session,
            normal="xyz".index(args.slice_axis),
            index=args.slice_index,
        )
        figure.savefig(args.plot, dpi=160)
    elif args.plot:
        plot_csv(
            args.out,
            args.plot,
            slice_axis=args.slice_axis,
            slice_index=args.slice_index,
        )
    session.close()
    print(
        (
            f"Session {('completed' if session.completed else 'paused')}: t="
            f"{session.time:.9g}, step={session.steps}"
        )
    )
