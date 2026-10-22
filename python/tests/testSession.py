"""Checkpoint/restart contracts, old-driver parity and independent output readers."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfdSolver import Simulation
from cfdSolver.torchIO import (
    read_checkpoint,
    write_checkpoint,
    write_vtk,
    write_xdmf,
    TemporalWriter,
)
from cfdSolver.torchSolver import run_simulation
from cfdSolver.torchSolver2D import run_torch_solver_2d
from cfdSolver.torchSolver3D import run_torch_solver_3d

ROOT = Path(__file__).resolve().parents[2]


def example(dim, reactive=False, model="arrhenius"):
    counts = [24] if dim == 1 else ([8, 6] if dim == 2 else [6, 4, 4])
    p = np.ones((dim + 2, *counts[::-1]))
    p[1:-1] = 0.2
    x = (np.arange(counts[0]) + 0.5) / counts[0]
    p[0] = 1 + 0.1 * np.sin(2 * np.pi * x)
    p[-1] = np.where(x < 0.5, 1e6, 101325) if reactive else 1
    return Simulation(
        p,
        [1.0] * dim,
        t_end=1e-4 if reactive else 0.08,
        boundaries=["Periodic"] * dim,
        reactive=reactive,
        kinetic_model=model,
    )


class SessionTests(unittest.TestCase):
    def test_restart_is_exact_in_all_dimensions_and_models(self):
        with tempfile.TemporaryDirectory() as directory:
            for dim in (1, 2, 3):
                for model in ("inert", "arrhenius", "znd"):
                    with self.subTest(dim=dim, model=model):
                        a = example(
                            dim,
                            model != "inert",
                            "arrhenius" if model == "inert" else model,
                        )
                        continuous = copy.deepcopy(a)
                        continuous.run()
                        a.run(
                            step_limit=2, checkpoint=Path(directory) / "checkpoint.h5"
                        )
                        self.assertFalse(a.completed)
                        restarted = read_checkpoint(Path(directory) / "checkpoint.h5")
                        restarted.run()
                        np.testing.assert_array_equal(restarted.state, continuous.state)
                        if restarted.lam is not None:
                            np.testing.assert_array_equal(restarted.lam, continuous.lam)
                        if restarted.beta is not None:
                            np.testing.assert_array_equal(
                                restarted.beta, continuous.beta
                            )
                        np.testing.assert_array_equal(
                            restarted.boundary_integral, continuous.boundary_integral
                        )
                        np.testing.assert_array_equal(
                            restarted.source_integral, continuous.source_integral
                        )
                        self.assertEqual(restarted.steps, continuous.steps)
                        self.assertEqual(restarted.time, continuous.time)

    def test_inert_sessions_match_legacy_drivers(self):
        s = Simulation.preset("sod_standard", [40], t_end=0.05)
        s.run()
        legacy, t = run_simulation(
            1, 40, 1, 0, 1, 0.125, 0, 0.1, c_cfl=0.4, t_corte=0.05
        )
        np.testing.assert_array_equal(s.state, legacy[:, :, -1])
        for dim in (2, 3):
            counts = [8, 6] if dim == 2 else [8, 6, 4]
            preset = "2d_explosion" if dim == 2 else "3d_explosion"
            s = Simulation.preset(preset, counts, t_end=0.02)
            s.run()
            if dim == 2:
                r = run_torch_solver_2d(nx=8, ny=6, t_end=0.02, cfl=0.4)
            else:
                r = run_torch_solver_3d(nx=8, ny=6, nz=4, t_end=0.02, cfl=0.4)
            np.testing.assert_allclose(s.state, r["u_final"], atol=1e-14)

    def test_periodic_output_does_not_change_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            a = example(2)
            b = copy.deepcopy(a)
            a.run()
            b.run(
                checkpoint=Path(directory) / "c.h5",
                checkpoint_every=2,
                output_prefix=Path(directory) / "history",
                output_every=3,
            )
            np.testing.assert_array_equal(a.state, b.state)
            np.testing.assert_array_equal(a.balance_residual, b.balance_residual)
            manifest = json.loads((Path(directory) / "history.series.json").read_text())
            self.assertEqual(manifest["frames"][0]["step"], 0)
            self.assertEqual(manifest["frames"][-1]["step"], b.steps)
            times = [f["time"] for f in manifest["frames"]]
            self.assertTrue(all(x < y for x, y in zip(times, times[1:])))
            # No retained list of state arrays or time history on the session.
            self.assertFalse(any(isinstance(v, list) for v in vars(b).values()))

    def test_append_series_and_reject_stale_checkpoint_or_other_run(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "series"
            s = example(1)
            s.run(
                step_limit=2, checkpoint=Path(directory) / "c.h5", output_prefix=prefix
            )
            resumed = Simulation.load(Path(directory) / "c.h5")
            resumed.run(output_prefix=prefix)
            manifest = json.loads(Path(str(prefix) + ".series.json").read_text())
            self.assertEqual(
                len({f["time"] for f in manifest["frames"]}), len(manifest["frames"])
            )
            with self.assertRaisesRegex(ValueError, "ahead"):
                TemporalWriter(prefix, s)
            with self.assertRaisesRegex(ValueError, "another"):
                TemporalWriter(prefix, example(1))

    def test_failed_atomic_write_preserves_last_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "c.h5"
            s = example(1)
            s.save(target)
            original = target.read_bytes()
            s.step()
            with patch(
                "cfdSolver.torchIO._hdf_content",
                side_effect=RuntimeError("write failed"),
            ):
                with self.assertRaises(RuntimeError):
                    s.save(target)
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_validation_rejects_corrupt_schema_state_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "c.h5"
            for mutation in (
                "version",
                "time",
                "geometry",
                "nan",
                "missing",
                "integral",
                "precision",
                "missing_metadata",
                "invalid_time",
            ):
                s = example(2)
                s.save(target)
                with h5py.File(target, "r+") as f:
                    if mutation in (
                        "version",
                        "time",
                        "geometry",
                        "integral",
                        "missing_metadata",
                        "invalid_time",
                    ):
                        meta = json.loads(bytes(f["metadata"][:]).decode())
                        if mutation == "version":
                            meta["schema_version"] = 999
                        elif mutation == "time":
                            meta["time"] = -1
                        elif mutation == "geometry":
                            meta["geometry"] = "spherical"
                        elif mutation == "missing_metadata":
                            del meta["time"]
                        elif mutation == "invalid_time":
                            meta["time"] = "invalid"
                        else:
                            meta["current_integral"][0] += 1
                        del f["metadata"]
                        f["metadata"] = np.frombuffer(
                            json.dumps(meta).encode(), dtype=np.uint8
                        )
                    elif mutation == "nan":
                        f["state/rho"][0] = np.nan
                    elif mutation == "precision":
                        v = f["state/rho"][:].astype(np.float32)
                        del f["state/rho"]
                        f["state/rho"] = v
                    else:
                        del f["state/energy"]
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    read_checkpoint(target)

    def test_snapshot_readers_recover_coordinates_fields_and_connectivity(self):
        import meshio
        from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
        from vtkmodules.util.numpy_support import vtk_to_numpy

        with tempfile.TemporaryDirectory() as directory:
            for dim in (1, 2, 3):
                s = example(dim, True)
                s.step()
                p = Path(directory) / str(dim)
                write_vtk(str(p) + ".vtr", s)
                write_xdmf(str(p) + ".xdmf", s)
                reader = vtkXMLRectilinearGridReader()
                reader.SetFileName(str(p) + ".vtr")
                reader.Update()
                grid = reader.GetOutput()
                self.assertEqual(
                    grid.GetFieldData().GetArray("TimeValue").GetValue(0), s.time
                )
                self.assertEqual(
                    grid.GetNumberOfCells(), np.prod(s.config["counts_xyz"])
                )
                np.testing.assert_array_equal(
                    vtk_to_numpy(grid.GetCellData().GetArray("rho")), s.state[0].ravel()
                )
                np.testing.assert_array_equal(
                    vtk_to_numpy(grid.GetCellData().GetArray("pressure")),
                    s.primitive[-1].ravel(),
                )
                mesh = meshio.read(str(p) + ".xdmf")
                self.assertEqual(
                    mesh.cells[0].type, {1: "line", 2: "quad", 3: "hexahedron"}[dim]
                )
                np.testing.assert_array_equal(
                    mesh.cell_data["rho"][0], s.state[0].ravel()
                )
                np.testing.assert_array_equal(
                    mesh.cell_data["lambda"][0], s.lam.ravel()
                )
                points = mesh.points[mesh.cells[0].data].mean(axis=1)
                axes = [
                    (np.arange(n) + 0.5) * h
                    for n, h in zip(s.config["counts_xyz"], s.spacing)
                ]
                centers = np.meshgrid(*axes[::-1], indexing="ij")[::-1]
                for k in range(dim):
                    np.testing.assert_allclose(points[:, k], centers[k].ravel())
                # Visualization HDF5 is itself a valid restart snapshot.
                r = Simulation.load(str(p) + ".h5")
                np.testing.assert_array_equal(r.state, s.state)

    def test_xdmf_temporal_reader_and_pvd_references(self):
        import meshio

        with tempfile.TemporaryDirectory() as directory:
            s = example(3)
            prefix = Path(directory) / "series"
            s.run(output_prefix=prefix, output_every=2)
            with meshio.xdmf.TimeSeriesReader(str(prefix) + ".xdmf") as reader:
                points, cells = reader.read_points_cells()
                times = []
                for i in range(reader.num_steps):
                    time, _, fields = reader.read_data(i)
                    times.append(time)
                np.testing.assert_array_equal(fields["energy"][0], s.state[-1].ravel())
                self.assertEqual(times[-1], s.time)
            datasets = ET.parse(str(prefix) + ".pvd").findall(".//DataSet")
            self.assertEqual(len(datasets), len(times))
            self.assertTrue(
                all((Path(directory) / f.attrib["file"]).exists() for f in datasets)
            )

    def test_cli_pause_restart_and_override_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "c.h5"
            output = directory / "out.csv"
            cli = [sys.executable, str(ROOT / "python/torchCli.py")]
            first = subprocess.run(
                cli
                + [
                    "--preset",
                    "sod_standard",
                    "--n",
                    "24",
                    "--t_corte",
                    ".08",
                    "--stop_after_steps",
                    "2",
                    "--checkpoint",
                    str(checkpoint),
                    "--out",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("paused", first.stdout)
            final = subprocess.run(
                cli
                + [
                    "--restart",
                    str(checkpoint),
                    "--checkpoint",
                    str(checkpoint),
                    "--vtk",
                    str(directory / "state.vtr"),
                    "--xdmf",
                    str(directory / "state.xdmf"),
                    "--out",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(final.returncode, 0, final.stderr)
            self.assertTrue(Simulation.load(checkpoint).completed)
            invalid = subprocess.run(
                cli + ["--restart", str(checkpoint), "--cfl", ".1"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(invalid.returncode, 2)


if __name__ == "__main__":
    unittest.main()
