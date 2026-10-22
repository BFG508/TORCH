"""Independent metric/characteristic checks and v1.2 integration regressions."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfdSolver import Simulation, exact_riemann_sod
from cfdSolver.torchBoundaries import ghost
from cfdSolver.torchFiniteVolume import rhs
from cfdSolver.torchGeometry import Metrics, axes, map_points
from cfdSolver.torchIO import write_vtk, write_xdmf, TemporalWriter


def example(dim, geometry, reactive=False):
    counts = [12, 6, 4][:dim]
    origin = [0.4, 0.3, 0.2][:dim] if geometry != "cartesian" else [0.1] * dim
    lengths = [0.8, 0.6, 0.4][:dim]
    centers = [
        o + (np.arange(n) + 0.5) * l / n for o, l, n in zip(origin, lengths, counts)
    ]
    grids = np.meshgrid(*centers[::-1], indexing="ij")[::-1]
    p = np.ones((dim + 2, *counts[::-1]))
    p[0] = 1 + 0.05 * np.sin(3 * grids[0])
    if dim > 1:
        p[0] += 0.03 * np.cos(2 * grids[1])
    if dim == 3:
        p[0] += 0.01 * grids[2]
    p[1:-1] = 0.15
    p[-1] = np.where(grids[0] < 0.8, 1e6, 101325.0) if reactive else 1.0
    return Simulation(
        p,
        lengths,
        geometry=geometry,
        origins=origin,
        t_end=2e-5 if reactive else 0.025,
        reactive=reactive,
        face_boundaries={
            "x_min": {
                "type": "inflow",
                "primitive": [1.0, 0.15]
                + [0.15] * (dim - 1)
                + [101325.0 if reactive else 1.0],
            },
            "x_max": {
                "type": "pressure_outlet",
                "pressure": 101325.0 if reactive else 1.0,
                "backflow": [1.0, -0.1]
                + [0.0] * (dim - 1)
                + [101325.0 if reactive else 1.0],
            },
        },
    )


class GeometryTests(unittest.TestCase):
    def test_exact_physical_volumes_all_geometries_dimensions(self):
        for dim in (1, 2, 3):
            for geometry in ("cartesian", "cylindrical", "annular", "spherical"):
                s = example(dim, geometry)
                r0 = 0.4 if geometry != "cartesian" else 0.1
                r1 = r0 + 0.8
                if geometry == "cartesian":
                    expected = np.prod([0.8, 0.6, 0.4][:dim])
                elif geometry == "spherical":
                    expected = (
                        (r1**3 - r0**3)
                        / 3
                        * (
                            4 * np.pi
                            if dim == 1
                            else (np.cos(0.3) - np.cos(0.9))
                            * (2 * np.pi if dim == 2 else 0.4)
                        )
                    )
                else:
                    expected = (
                        (r1 * r1 - r0 * r0)
                        / 2
                        * (
                            2 * np.pi
                            if dim == 1
                            else (
                                2 * np.pi * 0.6
                                if geometry == "cylindrical" and dim == 2
                                else 0.6 * (0.4 if dim == 3 else 1)
                            )
                        )
                    )
                self.assertAlmostEqual(s.metrics.volumes.sum(), expected, places=13)

    def test_uniform_rest_preserved_at_radial_origin_and_poles(self):
        for dim in (1, 2, 3):
            for geometry in ("cartesian", "cylindrical", "annular", "spherical"):
                origin = (
                    [0.3] * dim if dim == 3 or geometry == "annular" else [0.0] * dim
                )
                lengths = (
                    [1.0, np.pi, 0.5][:dim]
                    if geometry == "spherical" and dim < 3
                    else [0.5] * dim
                )
                p = np.ones((dim + 2, *([6] * dim)))
                p[1:-1] = 0
                s = Simulation(
                    p, lengths, geometry=geometry, origins=origin, t_end=0.02
                )
                old = s.state.copy()
                s.run()
                np.testing.assert_allclose(s.state, old, rtol=0, atol=3e-14)
                np.testing.assert_allclose(s.balance_residual, 0, atol=3e-14)

    def test_radial_homologous_expansion_against_analytic_solution(self):
        # u=a0*r/(1+a0*t), rho=rho0/f^D, p=p0/f^(gamma*D).
        for geometry, D in (("cylindrical", 2), ("spherical", 3)):
            errors = []
            for n in (24, 48, 96):
                r = 0.5 + (np.arange(n) + 0.5) / n
                p = np.array([np.ones(n), 0.2 * r, np.ones(n)])
                s = Simulation(p, [1.0], origins=[0.5], geometry=geometry, t_end=0.02)
                s.run()
                mask = (r > 0.7) & (r < 1.3)
                f = 1 + 0.2 * s.time
                exact = np.array(
                    [np.full(n, f ** (-D)), 0.2 * r / f, np.full(n, f ** (-1.4 * D))]
                )
                errors.append(np.mean(abs(s.primitive[:, mask] - exact[:, mask])))
            self.assertGreater(errors[0] / errors[1], 2.5, (geometry, errors))
            self.assertGreater(errors[1] / errors[2], 2.5, (geometry, errors))

    def test_rotating_equilibrium_has_convergent_metric_residual(self):
        for geometry, dim in (("annular", 2), ("cylindrical", 3), ("spherical", 3)):
            errors = []
            for n in (16, 32):
                counts = [n, n] + ([4] if dim == 3 else [])
                lengths = [1.0, 0.6] + ([0.5] if dim == 3 else [])
                origin = [0.5, 0.3] + ([0.2] if dim == 3 else [])
                grids = np.meshgrid(
                    *[
                        o + (np.arange(k) + 0.5) * l / k
                        for o, l, k in zip(origin, lengths, counts)
                    ][::-1],
                    indexing="ij",
                )[::-1]
                r, t = grids[:2]
                p = np.ones((dim + 2, *counts[::-1]))
                p[1:-1] = 0
                speed = 0.2 * r * (np.sin(t) if geometry == "spherical" else 1.0)
                p[3 if geometry == "spherical" else 2] = speed
                p[-1] = 1 + 0.5 * speed**2
                s = Simulation(
                    p, lengths, geometry=geometry, origins=origin, t_end=0.01
                )
                derivative = rhs(s.state, [], s.config, s.metrics)[0]
                active = [slice(None)] * (dim + 1)
                active[-1] = slice(2, -2)
                active[-2] = slice(2, -2)
                errors.append(np.max(abs(derivative[tuple(active)])))
            self.assertGreater(errors[0] / errors[1], 2.5, (geometry, errors))

    def test_characteristic_boundaries_satisfy_physical_invariants(self):
        gamma = np.array([1.4])
        p = np.array([[1.2], [40.0], [101325.0]])
        a = np.sqrt(gamma * p[-1] / p[0])
        outlet, _, _ = ghost(
            p, gamma, [], dict(type="pressure_outlet", pressure=90000.0), 0, 1, False
        )
        anew = np.sqrt(gamma * outlet[-1] / outlet[0])
        np.testing.assert_allclose(outlet[-1], 90000.0)
        np.testing.assert_allclose(
            outlet[-1] / outlet[0] ** gamma, p[-1] / p[0] ** gamma
        )
        np.testing.assert_allclose(
            outlet[1] + 2 * anew / (gamma - 1), p[1] + 2 * a / (gamma - 1)
        )
        sup = p.copy()
        sup[1] = 600
        np.testing.assert_array_equal(
            ghost(
                sup, gamma, [], dict(type="pressure_outlet", pressure=1.0), 0, 1, False
            )[0],
            sup,
        )
        face = dict(
            type="stagnation_inlet",
            total_pressure=110000.0,
            total_temperature=300.0,
            gas_constant=287.0,
        )
        inside = np.array([[1.2], [20.0], [101325.0]])
        inlet, g, _ = ghost(inside, gamma, [], face, 0, -1, False)
        speed = inlet[1]
        sound = np.sqrt(g * inlet[-1] / inlet[0])
        mach = speed / sound
        np.testing.assert_allclose(
            inlet[-1] * (1 + 0.5 * (g - 1) * mach**2) ** (g / (g - 1)),
            110000.0,
            rtol=1e-13,
        )
        np.testing.assert_allclose(
            inlet[-1] / (287 * inlet[0]) + speed**2 / (2 * 1.4 * 287 / 0.4),
            300.0,
            rtol=1e-13,
        )
        np.testing.assert_allclose(
            speed - 2 * sound / (g - 1),
            inside[1] - 2 * np.sqrt(g * inside[-1] / inside[0]) / (g - 1),
        )
        # Choked reservoir limits the inflow Mach number to one.
        sonic, _, _ = ghost(
            np.array([[1.0], [0.0], [1.0]]), gamma, [], face, 0, -1, False
        )
        self.assertAlmostEqual(
            float(sonic[1, 0] / np.sqrt(1.4 * sonic[-1, 0] / sonic[0, 0])), 1.0
        )
        with self.assertRaises(ValueError):
            ghost(
                np.array([[1.0], [-20.0], [101325.0]]),
                gamma,
                [],
                dict(type="pressure_outlet", pressure=101325.0),
                0,
                1,
                False,
            )
        back, _, q = ghost(
            np.array([[1.0], [-20.0], [101325.0]]),
            gamma,
            [np.array([0.5])],
            dict(
                type="pressure_outlet",
                pressure=101325.0,
                backflow=[2.0, -30.0, 101325.0],
                **{"lambda": 0.8},
            ),
            0,
            1,
            True,
        )
        np.testing.assert_array_equal(back[:, 0], [2.0, -30.0, 101325.0])
        self.assertEqual(q[0][0], 0.8)

    def test_closed_metrics_preserve_mass_energy_and_cold_species(self):
        for dim in (1, 2, 3):
            for geometry in ("cartesian", "cylindrical", "annular", "spherical"):
                s = example(dim, geometry)
                p = s.primitive
                p[1:-1] = 0.05
                fractions = np.linspace(0.2, 0.8, p[0].size).reshape(p.shape[1:])
                faces = {
                    a + side: "wall" for a in "xyz"[:dim] for side in ("_min", "_max")
                }
                s = Simulation(
                    p,
                    s.config["lengths_xyz"],
                    geometry=geometry,
                    origins=s.config["origins_xyz"],
                    t_end=0.01,
                    reactive=True,
                    lambda_init=fractions,
                    face_boundaries=faces,
                )
                initial = s.integral.copy()
                species = np.sum(s.state[0] * s.lam * s.metrics.volumes)
                s.run()
                np.testing.assert_allclose(
                    s.integral[[0, -1]], initial[[0, -1]], rtol=2e-13, atol=2e-13
                )
                self.assertAlmostEqual(
                    np.sum(s.state[0] * s.lam * s.metrics.volumes), species, places=12
                )
                np.testing.assert_allclose(s.balance_residual, 0, atol=2e-13)

    def test_inflow_and_pressure_outlet_on_every_face(self):
        for dim in (1, 2, 3):
            for direction in range(dim):
                for side in (-1, 1):
                    p = np.ones((dim + 2, *([5] * dim)))
                    p[1:-1] = 0
                    p[direction + 1] = -side * 0.2
                    values = [1.0] + [0.0] * dim + [1.0]
                    values[direction + 1] = -side * 0.2
                    faces = {
                        a + edge: "periodic"
                        for a in "xyz"[:dim]
                        for edge in ("_min", "_max")
                    }
                    axis = "xyz"[direction]
                    faces[axis + ("_min" if side == -1 else "_max")] = dict(
                        type="inflow", primitive=values, **{"lambda": 0.6}
                    )
                    faces[axis + ("_max" if side == -1 else "_min")] = dict(
                        type="pressure_outlet", pressure=1.0
                    )
                    for reactive in (False, True):
                        s = Simulation(
                            p,
                            [1.0] * dim,
                            t_end=0.02,
                            face_boundaries=faces,
                            reactive=reactive,
                            lambda_init=np.full(p.shape[1:], 0.6) if reactive else None,
                        )
                        initial = s.state.copy()
                        s.run()
                        np.testing.assert_allclose(
                            s.state, initial, atol=1e-14, rtol=1e-14
                        )
                        np.testing.assert_allclose(s.balance_residual, 0, atol=1e-13)

    def test_restart_all_metric_models_and_schema1_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "restart.h5"
            for dim in (1, 2, 3):
                for geometry in ("cartesian", "cylindrical", "annular", "spherical"):
                    for reactive in (False, True):
                        s = example(dim, geometry, reactive)
                        full = copy.deepcopy(s)
                        full.run()
                        s.run(step_limit=1)
                        s.save(file)
                        continued = Simulation.load(file)
                        continued.run()
                        np.testing.assert_array_equal(full.state, continued.state)
                        np.testing.assert_array_equal(
                            full.boundary_integral, continued.boundary_integral
                        )
                        np.testing.assert_array_equal(
                            full.source_integral, continued.source_integral
                        )
                        if reactive:
                            np.testing.assert_array_equal(full.lam, continued.lam)
            p = np.ones((3, 12))
            p[1] = 0
            s = Simulation(p, [1.0], t_end=0.1)
            s.save(file)
            with h5py.File(file, "r+") as f:
                meta = json.loads(bytes(f["metadata"][:]).decode())
                meta["schema_version"] = 1
                meta["solver_version"] = "1.1"
                del meta["config"]["geometry"]
                del meta["config"]["origins_xyz"]
                del f["metadata"]
                f["metadata"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
            old = Simulation.load(file)
            old.run()
            old.save(file)
            with h5py.File(file) as f:
                self.assertEqual(
                    json.loads(bytes(f["metadata"][:]).decode())["schema_version"], 2
                )
            # Python-only reactive ZND preserves both composition fields in geometry sessions.
            s = example(2, "annular", True)
            p = s.primitive
            s = Simulation(
                p,
                s.config["lengths_xyz"],
                t_end=2e-6,
                geometry="annular",
                origins=s.config["origins_xyz"],
                reactive=True,
                kinetic_model="znd",
            )
            full = copy.deepcopy(s)
            full.run()
            s.run(step_limit=1)
            s.save(file)
            s = Simulation.load(file)
            s.run()
            np.testing.assert_array_equal(s.state, full.state)
            np.testing.assert_array_equal(s.beta, full.beta)

    def test_physical_mesh_vtk_xdmf_and_temporal_restart(self):
        import meshio
        from vtkmodules.vtkIOXML import vtkXMLStructuredGridReader
        from vtkmodules.util.numpy_support import vtk_to_numpy

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for geometry in ("cylindrical", "annular", "spherical"):
                for dim in (1, 2, 3):
                    s = example(dim, geometry)
                    s.run(output_prefix=folder / "history")
                    write_vtk(folder / "grid.vts", s)
                    write_xdmf(folder / "grid.xdmf", s)
                    m = meshio.read(folder / "grid.xdmf")
                    reader = vtkXMLStructuredGridReader()
                    reader.SetFileName(str(folder / "grid.vts"))
                    reader.Update()
                    v = reader.GetOutput()
                    np.testing.assert_allclose(
                        vtk_to_numpy(v.GetPoints().GetData()), m.points, atol=1e-14
                    )
                    aa = axes(s.config) + [np.array([0.0])] * (3 - dim)
                    z, y, x = np.meshgrid(aa[2], aa[1], aa[0], indexing="ij")
                    np.testing.assert_allclose(
                        m.points,
                        map_points(
                            s.config, np.column_stack([x.ravel(), y.ravel(), z.ravel()])
                        ),
                    )
                    self.assertEqual(v.GetNumberOfCells(), s.state[0].size)
                    np.testing.assert_array_equal(
                        m.cell_data["rho"][0], s.state[0].ravel()
                    )
                    TemporalWriter(folder / "history", s).append(s)
                    np.testing.assert_array_equal(
                        Simulation.load(folder / "grid.h5").state, s.state
                    )
                    for path in folder.iterdir():
                        path.unlink()

    def test_sod_against_exact_solution_with_separate_face_conditions(self):
        n = 120
        r = (np.arange(n) + 0.5) / n
        p = np.ones((3, n))
        p[1] = 0
        p[0, r > 0.5] = 0.125
        p[-1, r > 0.5] = 0.1
        s = Simulation(
            p,
            [1.0],
            t_end=0.08,
            face_boundaries={"x_min": "outflow", "x_max": "outflow"},
        )
        s.run()
        exact = exact_riemann_sod(r, t=0.08)[0]
        self.assertLess(np.mean(abs(s.primitive[0] - exact)), 0.012)
        np.testing.assert_allclose(s.balance_residual, 0, atol=2e-13)

    def test_invalid_geometries_and_incomplete_boundaries_rejected(self):
        p = np.ones((4, 4, 4))
        p[1:-1] = 0
        for opts in (
            dict(geometry="unknown"),
            dict(geometry="annular"),
            dict(geometry="spherical", lengths=[1.0, 4.0]),
            dict(face_boundaries={"x_min": "periodic"}),
            dict(face_boundaries={"bad": "wall"}),
            dict(
                face_boundaries={
                    "x_min": {"type": "inflow", "primitive": [1.0, -1.0, 0.0, 1.0]}
                }
            ),
            dict(
                geometry="annular",
                origins=[0.5, 0.0],
                lengths=[1.0, 2 * np.pi],
                face_boundaries={"x_min": "periodic", "x_max": "periodic"},
            ),
        ):
            kwargs = opts.copy()
            lengths = kwargs.pop("lengths", [1.0, 1.0])
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                Simulation(p, lengths, t_end=0.01, **kwargs)
        p3 = np.ones((5, 4, 4, 4))
        p3[1:-1] = 0
        for geo in ("spherical", "cylindrical"):
            with self.assertRaises(ValueError):
                Simulation(p3, [1.0, 1.0, 1.0], geometry=geo, t_end=0.01)

    def test_cli_metric_faces_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            face = d / "faces.json"
            face.write_text(
                json.dumps(
                    {
                        "x_min": "wall",
                        "x_max": "wall",
                        "y_min": "periodic",
                        "y_max": "periodic",
                    }
                )
            )
            base = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "torchCli.py"),
            ]
            command = base + [
                "--dim",
                "2d",
                "--geometry",
                "annular",
                "--origins",
                ".5",
                "0",
                "--nx",
                "8",
                "--ny",
                "8",
                "--lx",
                ".5",
                "--ly",
                str(2 * np.pi),
                "--primitive",
                "1",
                "0",
                "0",
                "1",
                "--face_boundaries",
                str(face),
                "--t_corte",
                ".2",
                "--stop_after_steps",
                "1",
                "--checkpoint",
                str(d / "c.h5"),
                "--out",
                str(d / "a.csv"),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(
                base
                + [
                    "--restart",
                    str(d / "c.h5"),
                    "--vtk",
                    str(d / "a.vts"),
                    "--out",
                    str(d / "a.csv"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((d / "a.vts").exists())
            self.assertEqual(
                (d / "a.csv").read_text().splitlines()[0], "r,theta,p,rho,u_r,u_theta"
            )


if __name__ == "__main__":
    unittest.main()
