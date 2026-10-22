"""AMR conservation, analytic advection, restart and CPU execution.
"""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Barrier, Lock, get_ident
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

import h5py
import numpy as np
from matplotlib.figure import Figure

if __package__:
    from . import _bootstrap
    from .testGeometry import example
else:
    import _bootstrap
    from testGeometry import example
from cfdSolver import Simulation, exact_riemann_sod
from cfdSolver.torchAdaptiveGeometry import LeafMesh, children, settings
from cfdSolver.torchExecution import Execution
from cfdSolver.torchIO import TemporalWriter, write_vtk, write_xdmf
from desktop.export import export_session
from desktop.view import draw_session, fields, session_plane


def uniform(dim=1, geometry="cartesian", reactive=False, **options):
    counts = [8, 4, 4][:dim]
    p = np.ones((dim + 2, *counts[::-1]))
    p[1:-1] = 0
    origins = [0.4, 0.3, 0.2][:dim] if geometry != "cartesian" else [0.0] * dim
    s = Simulation(
        p,
        [1.0] * dim,
        origins=origins,
        geometry=geometry,
        t_end=0.02,
        reactive=reactive,
    )
    return s.to_adaptive(max_level=2, **options)


class AdaptiveTests(unittest.TestCase):
    def test_rest_and_exact_metric_partition_in_all_geometries(self):
        for d in (1, 2, 3):
            for geometry in (
                "cartesian",
                "cylindrical",
                "annular",
                "spherical",
            ):
                with self.subTest(d=d, geometry=geometry):
                    s = uniform(d, geometry)
                    total = s.integral.copy()
                    volume = s.mesh.volumes.sum()
                    s.regrid(refine=s.mesh.centers[:, 0] < 0.8, coarsen=False)
                    self.assertGreater(s.mesh.levels.max(), 0)
                    self.assertAlmostEqual(s.mesh.volumes.sum(), volume)
                    np.testing.assert_allclose(s.integral, total, atol=2e-14)
                    s.run(step_limit=2)
                    np.testing.assert_allclose(
                        s.primitive[1:-1], 0, atol=1e-14
                    )
                    np.testing.assert_allclose(
                        s.balance_residual, 0, atol=3e-14
                    )

    def test_refined_rest_at_radial_origin_and_spherical_poles(self):
        for d in (1, 2):
            for geometry in ("cylindrical", "spherical"):
                p = np.ones((d + 2, *([6] * d)))
                p[1:-1] = 0
                lengths = [1.] * d
                if geometry == "spherical" and d == 2:
                    lengths[1] = np.pi
                s = Simulation(p, lengths, geometry=geometry,
                               t_end=.01).to_adaptive(max_level=1)
                s.regrid(refine=s.mesh.low[:, 0] == 0, coarsen=False)
                s.run()
                np.testing.assert_allclose(s.primitive[1:-1], 0, atol=2e-14)
                np.testing.assert_allclose(s.balance_residual, 0, atol=3e-14)

    def test_adaptive_sod_along_all_axes_against_exact_riemann(self):
        for d in (1, 2, 3):
            for axis in range(d):
                counts = [2] * d
                counts[axis] = 32
                p = np.ones((d + 2, *counts[::-1]))
                p[1:-1] = 0
                shape = [1] * d
                shape[d - 1 - axis] = 32
                right = np.broadcast_to((np.arange(32) >= 16).reshape(shape),
                                        counts[::-1])
                p[0, right] = .125
                p[-1, right] = .1
                s = Simulation(p, [1.] * d, t_end=.08).to_adaptive(
                    max_level=1, regrid_every=2)
                s.run()
                exact = exact_riemann_sod(s.mesh.centers[:, axis], t=s.time)[0]
                error = np.sum(abs(s.state[0] - exact) * s.mesh.volumes)
                self.assertLess(error, .025)
                self.assertGreater(s.mesh.levels.max(), 0)
                np.testing.assert_allclose(s.balance_residual, 0, atol=1e-13)

    def test_refine_restrict_species_and_energy(self):
        for d in (1, 2, 3):
            s = example(d, "spherical", True).to_adaptive(
                max_level=2, refine_threshold=1
            )
            before = s.integral.copy()
            species = np.sum(s.state[0] * s.lam * s.mesh.volumes)
            s.regrid(
                refine=np.ones(len(s.mesh.keys), dtype=bool), coarsen=False
            )
            np.testing.assert_allclose(s.integral, before, rtol=5e-15)
            self.assertAlmostEqual(
                np.sum(s.state[0] * s.lam * s.mesh.volumes), species
            )
            # Constant state/composition makes every
            # complete sibling group eligible.
            s.state[:] = s.state[:, :1]
            s.lam[:] = 0.37
            before = s.integral.copy()
            species = np.sum(s.state[0] * s.lam * s.mesh.volumes)
            count = len(s.mesh.keys)
            s.regrid(refine=np.zeros(len(s.mesh.keys), dtype=bool))
            self.assertLess(len(s.mesh.keys), count)
            np.testing.assert_allclose(s.integral, before, rtol=5e-15)
            self.assertAlmostEqual(
                np.sum(s.state[0] * s.lam * s.mesh.volumes), species
            )

    def test_two_to_one_closure_including_periodic_seam(self):
        for d in (1, 2, 3):
            s = uniform(d)
            s.config["face_boundaries"] = [{"type": "periodic"}] * (2 * d)
            s.mesh = LeafMesh(s.config, s.mesh.keys)
            for _ in range(2):
                s.regrid(refine=s.mesh.low[:, 0] == 0, coarsen=False)
            m = s.mesh
            np.testing.assert_array_less(
                abs(
                    m.levels[m.left[m.interior]]
                    - m.levels[m.right[m.interior]]
                ),
                2,
            )
            self.assertGreater(m.levels[m.high[:, 0] == 1].max(), 0)
            s.run()
            np.testing.assert_allclose(s.balance_residual, 0, atol=2e-14)

    def test_capacity_and_minimum_level(self):
        s = uniform(max_cells=8)
        old = s.mesh.keys.copy()
        s.regrid(refine=np.ones(8, dtype=bool), coarsen=False)
        self.assertTrue(s.budget_limited)
        np.testing.assert_array_equal(s.mesh.keys, old)
        s = uniform(min_level=1, max_cells=16)
        self.assertEqual(len(s.mesh.keys), 16)
        self.assertEqual(s.mesh.levels.min(), 1)
        with self.assertRaises(ValueError):
            uniform(min_level=2, max_cells=16)
        for options in (
            {"max_level": 9},
            {"max_cells": True},
            {"refine_threshold": 0.01, "coarsen_threshold": 0.02},
        ):
            with self.assertRaises(ValueError):
                settings(**options)

    def test_reject_invalid_trees(self):
        s = uniform()
        root = s.mesh.keys
        invalid = [
            root[:-1],
            np.vstack((root, root[0])),
            np.vstack((root, children(tuple(root[0])))),
            root.astype(float),
        ]
        for keys in invalid:
            with self.assertRaises(ValueError):
                LeafMesh(s.config, keys)
        keys = np.vstack((root[1:], children(tuple(root[0]))))
        # Refine the interface child again against an
        # unrefined neighboring root.
        fine = keys[(keys[:, 0] == 1) & (keys[:, 1] == 1)][0]
        keys = np.vstack(
            (keys[~np.all(keys == fine, axis=1)], children(tuple(fine)))
        )
        with self.assertRaises(ValueError):
            LeafMesh(s.config, keys)

    def test_second_order_nonuniform_entropy_advection(self):
        errors = []
        for n in (16, 32, 64):
            x = (np.arange(n) + 0.5) / n
            p = np.array(
                [
                    1 + 0.2 * np.sinc(1 / n) * np.sin(2 * np.pi * x),
                    np.full(n, 0.5),
                    np.ones(n),
                ]
            )
            s = Simulation(
                p, [1], boundaries=["Periodic"], t_end=0.1
            ).to_adaptive(
                max_level=1,
                refine_threshold=1,
                coarsen_threshold=0,
                regrid_every=10000,
            )
            s.regrid(refine=s.mesh.centers[:, 0] < 0.5, coarsen=False)
            s.run()
            x, width = s.mesh.centers[:, 0], s.mesh.widths[:, 0]
            exact = 1 + 0.2 * np.sinc(width) * np.sin(
                2 * np.pi * (x - 0.5 * s.time)
            )
            errors.append(np.sum(abs(s.state[0] - exact) * width))
            np.testing.assert_allclose(s.balance_residual, 0, atol=4e-14)
        self.assertGreater(
            min(np.log2(np.array(errors[:-1]) / errors[1:])), 1.8
        )

    def test_exact_restart_changing_mesh_all_models_and_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "restart.h5"
            for d in (1, 2, 3):
                for model in ("inert", "arrhenius", "znd"):
                    p = np.ones((d + 2, *([4] * d)))
                    p[1:-1] = 0.1
                    p[0, ..., 2:] = 0.7
                    reactive = model != "inert"
                    s = Simulation(
                        p,
                        [1] * d,
                        t_end=0.08,
                        reactive=reactive,
                        kinetic_model=model if reactive else "arrhenius",
                    ).to_adaptive(max_level=1, regrid_every=1)
                    s.step()
                    s.save(path)
                    loaded = Simulation.load(path)
                    loaded.set_workers(2)
                    s.run()
                    loaded.run()
                    for attr in (
                        "state",
                        "lam",
                        "beta",
                        "boundary_integral",
                        "source_integral",
                    ):
                        np.testing.assert_array_equal(
                            getattr(loaded, attr), getattr(s, attr)
                        )
                    np.testing.assert_array_equal(
                        loaded.mesh.keys, s.mesh.keys
                    )
                    self.assertEqual(
                        (loaded.regrids, loaded.mesh_revision),
                        (s.regrids, s.mesh_revision),
                    )
                    loaded.close()
                    s.close()

    def test_transactional_step_and_failed_checkpoint(self):
        s = uniform()
        state, keys, time = s.state.copy(), s.mesh.keys.copy(), s.time
        with patch(
            "cfdSolver.torchAdaptive.remesh", side_effect=ValueError("failure")
        ):
            s.config["amr"]["regrid_every"] = 1
            with self.assertRaises(ValueError):
                s.step()
        np.testing.assert_array_equal(s.state, state)
        np.testing.assert_array_equal(s.mesh.keys, keys)
        self.assertEqual(s.time, time)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.h5"
            s.save(path)
            old = path.read_bytes()
            s.state[0, 0] = -1
            with self.assertRaises(ValueError):
                s.save(path)
            self.assertEqual(path.read_bytes(), old)

    def test_reject_corrupt_leaf_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.h5"
            for mode in (
                "layout",
                "composition",
                "counter",
                "integral",
                "indices",
            ):
                s = uniform(reactive=mode == "composition")
                s.save(path)
                with h5py.File(path, "r+") as h:
                    meta = json.loads(bytes(h["metadata"][:]))
                    if mode == "layout":
                        meta["layout"] = "x-fastest"
                    if mode == "composition":
                        del h["state/lambda"]
                    if mode == "counter":
                        meta["mesh_revision"] = meta["regrids"] + 1
                    if mode == "integral":
                        meta["current_integral"] = [1.0]
                    if mode == "indices":
                        h["mesh/indices"][0] = 1
                    del h["metadata"]
                    h["metadata"] = np.frombuffer(
                        json.dumps(meta).encode(), dtype=np.uint8
                    )
                with self.assertRaises(ValueError):
                    Simulation.load(path)

    def test_vtk_xdmf_changing_topology_independent_readers(self):
        import meshio
        from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
        from vtkmodules.util.numpy_support import vtk_to_numpy

        with tempfile.TemporaryDirectory() as folder:
            for d in (1, 2, 3):
                s = uniform(d, "spherical")
                prefix = Path(folder) / f"case{d}"
                writer = TemporalWriter(prefix, s)
                writer.append(s)
                s.regrid(refine=s.mesh.centers[:, 0] < 0.9, coarsen=False)
                with self.assertRaises(ValueError):
                    writer.append(s)
                s.step()
                writer.append(s)
                write_vtk(prefix.with_suffix(".vtu"), s)
                reader = vtkXMLUnstructuredGridReader()
                reader.SetFileName(str(prefix.with_suffix(".vtu")))
                reader.Update()
                grid = reader.GetOutput()
                self.assertEqual(grid.GetNumberOfCells(), len(s.mesh.keys))
                np.testing.assert_array_equal(
                    vtk_to_numpy(grid.GetCellData().GetArray("rho")),
                    s.state[0],
                )
                write_xdmf(prefix.with_name("static.xdmf"), s)
                mesh = meshio.read(prefix.with_name("static.xdmf"))
                np.testing.assert_array_equal(
                    mesh.cell_data["refinement_level"][0], s.mesh.levels
                )
                grids = ET.parse(prefix.with_suffix(".xdmf")).findall(
                    "./Domain/Grid/Grid"
                )
                counts = [
                    int(g.find("Topology").get("NumberOfElements"))
                    for g in grids
                ]
                self.assertEqual(
                    counts, [np.prod(s.config["counts_xyz"]), len(s.mesh.keys)]
                )
                for g, count in zip(grids, counts):
                    filename = g.find("Topology/DataItem").text.split(":")[0]
                    with h5py.File(prefix.parent / filename) as h:
                        self.assertEqual(h["mesh/cells"].shape[0], count)

    def test_leaf_views_and_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            for d in (1, 2, 3):
                s = uniform(d, "spherical")
                s.regrid(refine=s.mesh.centers[:, 0] < 0.9, coarsen=False)
                for name in fields(s):
                    plane = session_plane(s, name)
                    self.assertTrue(np.isfinite(plane.values).all())
                for mode in ("Map", "3D plane"):
                    figure = Figure(figsize=(8, 7))
                    draw_session(figure, s, "Refinement level", mode=mode)
                    figure.savefig(Path(folder) / "view.png")
                export_session(Path(folder) / "cells.csv", s)
                lines = (Path(folder) / "cells.csv").read_text().splitlines()
                self.assertEqual(len(lines), len(s.mesh.keys) + 1)
                self.assertIn("refinement_level", lines[0])
                self.assertIn("width_r", lines[0])

    def test_cli_amr_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            command = [
                sys.executable,
                str(Path(_bootstrap.SOURCE_ROOT) / "torchCli.py"),
            ]
            subprocess.run(
                command
                + [
                    "--adaptive",
                    "--dim",
                    "1d",
                    "--n",
                    "16",
                    "--t_corte",
                    ".04",
                    "--amr_max_level",
                    "1",
                    "--stop_after_steps",
                    "1",
                    "--checkpoint",
                    str(path / "s.h5"),
                    "--out",
                    str(path / "a.csv"),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                command
                + [
                    "--restart",
                    str(path / "s.h5"),
                    "--workers",
                    "2",
                    "--checkpoint",
                    str(path / "s.h5"),
                    "--vtk",
                    str(path / "s.vtu"),
                    "--out",
                    str(path / "b.csv"),
                ],
                check=True,
                capture_output=True,
            )
            self.assertTrue(Simulation.load(path / "s.h5").completed)


class ParallelTests(unittest.TestCase):
    def test_real_bounded_concurrency_and_none_items(self):
        lock, barrier, ids, live, maximum = Lock(), Barrier(2), set(), [0], [0]

        def task(value):
            with lock:
                ids.add(get_ident())
                live[0] += 1
                maximum[0] = max(maximum[0], live[0])
            barrier.wait(timeout=5)
            with lock:
                live[0] -= 1
            return value

        context = Execution(2)
        self.assertEqual(
            list(context.map(task, [None, 1, 2, 3])), [None, 1, 2, 3]
        )
        self.assertEqual(len(ids), 2)
        self.assertEqual(maximum[0], 2)
        context.close()
        self.assertIsNone(context.pool)

    def test_uniform_serial_parallel_bitwise_all_kernels(self):
        cases = [
            Simulation.preset("sod_standard", [17000], t_end=0.001),
            Simulation.preset("2d_explosion", [96, 96], t_end=0.01),
            Simulation.preset("3d_sod_x", [24] * 3, t_end=0.01),
            example(3, "spherical", True),
        ]
        for s in cases:
            other = copy.deepcopy(s)
            other.set_workers(2)
            s.run(step_limit=2)
            other.run(step_limit=2)
            np.testing.assert_array_equal(s.state, other.state)
            np.testing.assert_array_equal(s.lam, other.lam)
            np.testing.assert_array_equal(
                s.balance_residual, other.balance_residual
            )
            other.close()
        for value in (0, 65, True, 1.5):
            with self.assertRaises(ValueError):
                Execution(value)


if __name__ == "__main__":
    unittest.main()
