"""Independent AMR advection and warmed CPU/memory benchmark report."""

import argparse
import copy
import json
from pathlib import Path
import platform
import statistics
import time
import tracemalloc

import numpy as np

from cfdSolver import Simulation


def entropy(n, dimension):
    counts = [n] + [4] * (dimension - 1)
    x = (np.arange(n) + 0.5) / n
    p = np.ones((dimension + 2, *counts[::-1]))
    p[0] = 1 + 0.2 * np.sinc(1 / n) * np.sin(2 * np.pi * x)
    p[1:-1] = 0
    p[1] = 0.5
    s = Simulation(
        p, [1.0] * dimension, boundaries=["Periodic"] * dimension, t_end=0.1
    ).to_adaptive(
        max_level=1,
        refine_threshold=1,
        coarsen_threshold=0,
        regrid_every=10000,
    )
    s.regrid(refine=s.mesh.centers[:, 0] < 0.5, coarsen=False)
    s.run()
    x, w = s.mesh.centers[:, 0], s.mesh.widths[:, 0]
    exact = 1 + 0.2 * np.sinc(w) * np.sin(2 * np.pi * (x - 0.5 * s.time))
    error = float(np.sum(abs(s.state[0] - exact) * s.mesh.volumes))
    balance = float(np.max(abs(s.balance_residual)))
    return dict(
        dimension=dimension,
        roots=counts,
        leaves=len(s.mesh.keys),
        density_l1=error,
        balance_linf=balance,
        steps=s.steps,
    )


def make_case(dimension, n, adaptive):
    p = np.ones((dimension + 2, *([n] * dimension)))
    p[1:-1] = 0.1
    x = (np.arange(n) + 0.5) / n
    p[-1] = 1 + 0.3 * np.exp(-(((x - 0.5) / 0.1) ** 2))
    s = Simulation(p, [1.0] * dimension, t_end=10, face_boundaries={})
    if adaptive:
        base = s
        s = base.to_adaptive(
            max_level=1,
            refine_threshold=0.02,
            coarsen_threshold=0.005,
            regrid_every=3,
            max_cells=200000,
        )
        base.close()
    return s


def benchmark(dimension, n, adaptive, workers, steps, repeats):
    template = make_case(dimension, n, adaptive)
    samples = []
    for _ in range(repeats):
        s = copy.deepcopy(template)
        s.set_workers(workers)
        s.run(step_limit=2)
        start = time.perf_counter()
        s.run(step_limit=steps)
        samples.append((time.perf_counter() - start) / steps)
        s.close()
    s = copy.deepcopy(template)
    s.set_workers(workers)
    s.run(step_limit=2)
    tracemalloc.start()
    try:
        s.run(step_limit=steps)
        retained, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    leaves = len(s.mesh.keys) if adaptive else n**dimension
    row = dict(
        dimension=dimension,
        roots=[n] * dimension,
        adaptive=adaptive,
        workers=workers,
        warmup_steps=2,
        measured_steps=steps,
        repeats=repeats,
        seconds_per_step=statistics.median(samples),
        samples=samples,
        leaves=leaves,
        peak_traced_bytes=peak,
        retained_traced_bytes=retained,
        balance_linf=float(np.max(abs(s.balance_residual))),
    )
    if adaptive:
        start = time.perf_counter()
        s.regrid()
        row.update(
            regrid_seconds=time.perf_counter() - start,
            finest_uniform_cells=(2 * n) ** dimension,
            leaf_fraction_of_finest=leaves / (2 * n) ** dimension,
        )
    s.close()
    template.close()
    return row


def study(quick=False, worker_counts=(1, 2, 4)):
    sizes = (12, 24) if quick else (16, 32, 64)
    checks = []
    for d in (1, 2, 3):
        rows = [entropy(n, d) for n in sizes]
        errors = [r["density_l1"] for r in rows]
        orders = np.log2(np.array(errors[:-1]) / errors[1:]).tolist()
        checks.append(
            dict(
                case="periodic nonuniform entropy advection",
                dimension=d,
                results=rows,
                orders=orders,
                passed=min(orders) > 1.7
                and max(r["balance_linf"] for r in rows) < 2e-12,
            )
        )
    benches = []
    for d, n, adaptive in (
        (2, 96 if quick else 192, False),
        (3, 24 if quick else 40, False),
        (2, 16 if quick else 32, True),
        (3, 8 if quick else 12, True),
    ):
        for workers in worker_counts:
            benches.append(
                benchmark(
                    d,
                    n,
                    adaptive,
                    workers,
                    2 if quick else 5,
                    1 if quick else 3,
                )
            )
    return dict(
        schema_version=1,
        suite="TORCH 2.0 AMR and CPU execution",
        runtime=platform.python_version(),
        numpy=np.__version__,
        quick=quick,
        passed=all(c["passed"] for c in checks),
        memory_scope=(
            "Traced allocations during warmed steps; excludes initial "
            "state, grid, native-library memory and process RSS."
        ),
        timing_scope=(
            "Wall time including global CFL, reconstruction, fluxes, "
            "scatter and scheduled regrids; excludes initial pool and "
            "topology construction. Timing is informational."
        ),
        checks=checks,
        benchmarks=benches,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--out", type=Path, default=Path("out/adaptive-python.json")
    )
    args = parser.parse_args()
    report = study(args.quick, tuple(args.workers))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for c in report["checks"]:
        print(c["dimension"], c["orders"], "PASS" if c["passed"] else "FAIL")
    print("Report:", args.out)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
