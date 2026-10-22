"""Reproducible session timing and allocation measurements."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc

import numpy as np


def cases(quick=False):
    """Fixed smooth cases shared with the MATLAB benchmark."""
    return (
        (1, "cartesian", 128 if quick else 512, False),
        (2, "cartesian", 96 if quick else 192, False),
        (3, "cartesian", 24 if quick else 40, False),
        (3, "spherical", 24 if quick else 40, False),
        (3, "annular", 16 if quick else 24, True),
    )


def make_session(dimension, geometry, n, reactive):
    from cfdSolver import Simulation

    p = np.ones((dimension + 2, *([n] * dimension)))
    p[1:-1] = 0.1
    p[-1] = (1e6 if reactive else 1.0) * (
        1 + 0.05 * np.sin(2 * np.pi * (np.arange(n) + 0.5) / n)
    )
    origins = [0.0] * dimension
    lengths = [1.0] * dimension
    if geometry != "cartesian":
        origins, lengths = [0.5, 0.3, 0.0], [0.5, 1.0, 1.0]
    return Simulation(
        p,
        lengths,
        origins=origins,
        geometry=geometry,
        reactive=reactive,
        t_end=10.0,
    )


def measure(case, steps=5, repeats=3, warmup=2):
    """Time untraced runs; measure NumPy/Python allocations separately.

    Peak traced bytes cover steps after warmup, excluding initial state
    and metric creation. This is allocation traffic above that starting
    point, not process RSS, native library memory or a total RAM budget.
    """
    for value in (steps, repeats, warmup):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("Steps, repeats and warmup must be positive")
    samples = []
    for _ in range(repeats):
        session = make_session(*case)
        session.run(step_limit=warmup)
        gc.collect()
        before = session.steps
        start = time.perf_counter()
        session.run(step_limit=steps)
        elapsed = time.perf_counter() - start
        if session.steps - before != steps:
            raise ValueError("End time reached; use fewer benchmark steps")
        samples.append(elapsed / steps)
    session = make_session(*case)
    session.run(step_limit=warmup)
    gc.collect()
    tracemalloc.start()
    try:
        before = session.steps
        session.run(step_limit=steps)
        retained, peak = tracemalloc.get_traced_memory()
        if session.steps - before != steps:
            raise ValueError("End time reached; use fewer benchmark steps")
    finally:
        tracemalloc.stop()
    seconds = statistics.median(samples)
    dimension, geometry, n, reactive = case
    return dict(
        dimension=dimension,
        geometry=geometry,
        counts=[n] * dimension,
        reactive=reactive,
        measured_steps=steps,
        warmup_steps=warmup,
        repeats=repeats,
        seconds_per_step=seconds,
        seconds_per_step_samples=samples,
        cell_updates_per_second=n**dimension / seconds,
        peak_traced_bytes=peak,
        retained_traced_bytes=retained,
        relative_balance_linf=float(
            np.max(
                np.abs(session.balance_residual)
                / np.maximum(np.abs(session.initial_integral), 1.0)
            )
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--label", default="working tree")
    parser.add_argument("--out", type=Path, default=Path("out/benchmark.json"))
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Use solvers from another checkout/archive",
    )
    args = parser.parse_args()
    root = (args.source_root or Path(__file__).resolve().parents[1]).resolve()
    if not (root / "python/cfdSolver/torchSession.py").is_file():
        parser.error(
            "source-root must contain python/cfdSolver/torchSession.py"
        )
    sys.path.insert(0, str(root / "python"))
    try:
        rows = [
            measure(case, args.steps, args.repeats, args.warmup)
            for case in cases(args.quick)
        ]
    except ValueError as exc:
        parser.error(str(exc))
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "python/cfdSolver").glob("*.py"))
    }
    report = dict(
        schema_version=1,
        suite="TORCH session benchmark",
        label=args.label,
        source_root=str(root),
        source_sha256=hashes,
        python=platform.python_version(),
        numpy=np.__version__,
        platform=platform.platform(),
        processor=platform.processor(),
        memory_method="tracemalloc after warmup; not RSS or total memory",
        timing_method="median per-step wall time; tracing disabled",
        results=rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in rows:
        print(
            f"{row['dimension']}D {row['geometry']} "
            f"reactive={row['reactive']}: "
            f"{row['seconds_per_step'] * 1000:.2f} ms/step, "
            f"{row['peak_traced_bytes'] / 2**20:.2f} MiB traced peak"
        )
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
