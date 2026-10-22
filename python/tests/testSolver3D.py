"""3D physical invariants, dimensional reduction, convergence and language parity."""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfdSolver.torchSolver3D import (
    run_torch_solver_3d,
    get_3d_preset,
    uprim_to_uconsv_3d,
    uconsv_to_uprim_3d,
    apply_boundary_conditions_3d,
    muscl_hancock_step_3d,
    reaction_step_3d,
)
from cfdSolver.torchSolver import gamma_calc, new_lambda_n_e_calc, physical_state
from cfdSolver.torchSolver2D import muscl_hancock_step_2d, uprim_to_uconsv_2d
from cfdSolver.torchAnalysis import exact_riemann_sod, plot_csv
from cfdSolver.torchFlux import hllc_flux

ROOT = Path(__file__).resolve().parents[2]


def uniform(shape=(4, 4, 4), velocity=(0.3, -0.2, 0.1), pressure=1.0):
    p = np.ones((5, *shape))
    for m, v in enumerate(velocity, 1):
        p[m] = v
    p[4] = pressure
    return p


class Solver3DTests(unittest.TestCase):
    def test_roundtrip_and_flux_with_both_tangential_velocities(self):
        primitive = uniform(velocity=(2, -3, 4), pressure=100)
        state = uprim_to_uconsv_3d(primitive)
        np.testing.assert_allclose(uconsv_to_uprim_3d(state), primitive)
        flat = state.reshape(5, -1)
        flux = hllc_flux(flat, flat)
        expected = flat * 2
        expected[1] += 100
        expected[4] = 2 * (flat[4] + 100)
        np.testing.assert_allclose(flux, expected)
        # A material contact carries both tangential momenta at the mass speed.
        left = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
        right = np.array([[2.0], [2.0], [-3.0], [-4.0], [100.0]])
        ul = uprim_to_uconsv_3d(left)
        ur = uprim_to_uconsv_3d(right)
        contact = hllc_flux(ul, ur)
        self.assertAlmostEqual(contact[2, 0], 6)
        self.assertAlmostEqual(contact[3, 0], 8)

    def test_reflective_ghosts_flip_only_normal_momentum(self):
        state = uprim_to_uconsv_3d(uniform())
        for axis, momentum, keyword in ((3, 1, "bc_x"), (2, 2, "bc_y"), (1, 3, "bc_z")):
            padded = apply_boundary_conditions_3d(state, **{keyword: "Reflective"})
            index = [slice(None), 2, 2, 2]
            index[axis] = 1
            ghost = padded[tuple(index)]
            expected = state[:, 0, 0, 0].copy()
            expected[momentum] *= -1
            np.testing.assert_allclose(ghost, expected)

    def test_uniform_state_and_full_periodic_balance(self):
        p = uniform()
        r = run_torch_solver_3d(
            primitive_init=p,
            nx=4,
            ny=4,
            nz=4,
            t_end=0.05,
            bc_x="Periodic",
            bc_y="Periodic",
            bc_z="Periodic",
        )
        np.testing.assert_allclose(r["u_prim_final"], p, atol=1e-14)
        np.testing.assert_allclose(r["balance_residual"], 0, atol=1e-14)
        self.assertEqual(r["time"], 0.05)

    def test_sod_along_each_axis_and_rotation_invariance(self):
        profiles = []
        for axis in ("x", "y", "z"):
            grid = dict(nx=4, ny=4, nz=4)
            grid["n" + axis] = 32
            r = run_torch_solver_3d(preset_name="3d_sod_" + axis, t_end=0.08, **grid)
            dimension = ("z", "y", "x").index(axis)
            rho = np.moveaxis(r["u_prim_final"][0], dimension, -1)[0, 0]
            exact = exact_riemann_sod(r[axis], t=0.08)[0]
            self.assertLess(np.mean(abs(rho - exact)), 0.04)
            self.assertTrue(physical_state(r["u_final"]))
            profiles.append(rho)
            np.testing.assert_allclose(r["balance_residual"], 0, atol=1e-13)
        for rho in profiles[1:]:
            np.testing.assert_allclose(rho, profiles[0], atol=1e-13)

    def test_extruded_2d_step_matches_3d(self):
        ny, nx, nz = 8, 10, 3
        y, x = np.meshgrid(
            (np.arange(ny) + 0.5) / ny, (np.arange(nx) + 0.5) / nx, indexing="ij"
        )
        p = np.stack(
            (
                1 + 0.1 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),
                0.2 + 0 * x,
                -0.1 + 0 * x,
                1 + 0 * x,
            )
        )
        state2 = uprim_to_uconsv_2d(p)
        state3 = np.repeat(
            np.stack((state2[0], state2[1], state2[2], np.zeros_like(x), state2[3]))[
                :, None
            ],
            nz,
            axis=1,
        )
        out2 = muscl_hancock_step_2d(
            state2, 1 / nx, 1 / ny, 0.002, bc_x="Periodic", bc_y="Periodic"
        )
        out3 = muscl_hancock_step_3d(
            state3,
            1 / nx,
            1 / ny,
            1 / nz,
            0.002,
            bc_x="Periodic",
            bc_y="Periodic",
            bc_z="Periodic",
        )
        np.testing.assert_allclose(out3[[0, 1, 2, 4], 0], out2, rtol=1e-13, atol=1e-14)

    def test_smooth_entropy_wave_mesh_convergence(self):
        errors = []
        for n in (16, 32, 64):
            p = uniform((4, 4, n), velocity=(1, 0, 0))
            x = (np.arange(n) + 0.5) / n
            p[0] = 1 + 0.2 * np.sin(2 * np.pi * x)
            r = run_torch_solver_3d(
                primitive_init=p,
                nx=n,
                ny=4,
                nz=4,
                t_end=0.1,
                bc_x="Periodic",
                bc_y="Periodic",
                bc_z="Periodic",
            )
            errors.append(
                np.mean(
                    abs(
                        r["u_prim_final"][0] - (1 + 0.2 * np.sin(2 * np.pi * (x - 0.1)))
                    )
                )
            )
        self.assertGreater(errors[0] / errors[1], 3)
        self.assertGreater(errors[1] / errors[2], 3)

    def test_spherical_explosion_symmetry_and_all_limiters(self):
        for limiter in ("VanLeer", "Albada", "MINBEE", "Superbee", "Unos"):
            r = run_torch_solver_3d(nx=8, ny=8, nz=8, t_end=0.02, limiter=limiter)
            rho = r["u_prim_final"][0]
            np.testing.assert_allclose(rho, rho.transpose(2, 1, 0), atol=1e-13)
            np.testing.assert_allclose(rho, rho[::-1], atol=1e-13)
            self.assertTrue(physical_state(r["u_final"]))
            self.assertLess(np.max(abs(r["balance_residual"])), 1e-12)

    def test_reflective_wall_mass_energy_and_open_boundary_balance(self):
        p = uniform(velocity=(1, 0.5, -0.3))
        for bc in ("Reflective", "Open"):
            r = run_torch_solver_3d(
                primitive_init=p,
                nx=4,
                ny=4,
                nz=4,
                t_end=0.04,
                bc_x=bc,
                bc_y=bc,
                bc_z=bc,
            )
            np.testing.assert_allclose(r["balance_residual"], 0, atol=1e-13)
            if bc == "Reflective":
                np.testing.assert_allclose(
                    r["boundary_integral"][[0, 4]], 0, atol=1e-13
                )

    def test_uniform_chemistry_parity_and_energy_source(self):
        for model in ("arrhenius", "znd"):
            p = uniform(velocity=(2, 3, 4), pressure=1e6)
            state = uprim_to_uconsv_3d(p)
            lam = np.ones(p.shape[1:])
            beta = lam.copy()
            out, l, b = reaction_step_3d(state, lam, beta, 1e-6, model)
            normal = np.array([[1.0], [2.0], [1e6 / 0.4 + 2]])
            expected, reacted, expected_beta = new_lambda_n_e_calc(
                np.ones(1), 1e-6, normal, model, np.ones(1)
            )
            np.testing.assert_allclose(l, expected[0])
            np.testing.assert_allclose(b, expected_beta[0])
            np.testing.assert_allclose(
                out[4] - state[4], reacted[2, 0] - normal[2, 0], atol=1e-8
            )
        r = run_torch_solver_3d(preset_name="3d_ignition", nx=4, ny=4, nz=4, t_end=1e-5)
        self.assertGreater(r["source_integral"][4], 0)
        self.assertLess(abs(r["balance_residual"][4]) / r["final_integral"][4], 1e-12)

    def test_periodic_species_and_induction_transport_in_three_directions(self):
        n = 8
        p = uniform((n, n, n), velocity=(1, 0.5, 0.3), pressure=100)
        lam = np.zeros((n, n, n))
        lam[2:4, 2:4, 2:4] = 1
        r = run_torch_solver_3d(
            primitive_init=p,
            nx=n,
            ny=n,
            nz=n,
            t_end=0.02,
            is_reactive=True,
            kinetic_model="znd",
            lambda_init=lam,
            beta_init=1 - lam,
            bc_x="Periodic",
            bc_y="Periodic",
            bc_z="Periodic",
        )
        species = r["u_final"][0] * r["lambda_final"]
        self.assertAlmostEqual(species.sum(), lam.sum(), places=10)
        self.assertAlmostEqual(
            (r["u_final"][0] * r["beta_final"]).sum(), (1 - lam).sum(), places=10
        )
        for key in ("X", "Y", "Z"):
            self.assertGreater((species * r[key]).sum(), (lam * r[key]).sum())

    def test_strong_shock_and_expansion_remain_physical(self):
        for limiter in ("VanLeer", "Albada", "MINBEE", "Superbee", "Unos"):
            for case in ("shock", "expansion"):
                p = uniform((4, 4, 24), velocity=(0, 0, 0))
                end = 0.002 if case == "shock" else 0.02
                if case == "shock":
                    p[4, :, :, :12] = 1000
                    p[4, :, :, 12:] = 0.01
                else:
                    p[4] = 0.4
                    p[1, :, :, :12] = -2
                    p[1, :, :, 12:] = 2
                r = run_torch_solver_3d(
                    primitive_init=p,
                    nx=24,
                    ny=4,
                    nz=4,
                    t_end=end,
                    cfl=0.5,
                    limiter=limiter,
                )
                self.assertTrue(physical_state(r["u_final"]))
                self.assertEqual(r["time"], end)
                np.testing.assert_allclose(r["balance_residual"], 0, atol=1e-11)

    def test_conservative_fallback_replaces_invalid_interface_fluxes(self):
        p = uniform((4, 4, 4))
        p[0, :, :, 1:3] = 2
        state = uprim_to_uconsv_3d(p)

        def invalid(left, right, *args, **kwargs):
            return np.full_like(left, np.nan)

        with patch("cfdSolver.torchSolver3D.hllc_flux", side_effect=invalid):
            final, fx, fy, fz = muscl_hancock_step_3d(
                state,
                0.25,
                0.25,
                0.25,
                0.005,
                bc_x="Periodic",
                bc_y="Periodic",
                bc_z="Periodic",
                return_fluxes=True,
            )
        self.assertTrue(physical_state(final))
        expected = state - 0.005 / 0.25 * (
            np.diff(fx, axis=3) + np.diff(fy, axis=2) + np.diff(fz, axis=1)
        )
        np.testing.assert_allclose(final, expected)
        np.testing.assert_allclose(
            final.sum(axis=(1, 2, 3)), state.sum(axis=(1, 2, 3)), atol=1e-12
        )

    def test_input_validation_and_step_limit(self):
        for options in (
            dict(nx=1),
            dict(ny=2.5),
            dict(cfl=0.6),
            dict(lz=0),
            dict(gamma=1),
            dict(bc_z="bad"),
            dict(preset_name="bad"),
            dict(limiter="bad"),
            dict(primitive_init=np.zeros((5, 4, 4, 4))),
            dict(beta_init=np.ones((4, 4, 4))),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_torch_solver_3d(
                    nx=options.pop("nx", 4),
                    ny=options.pop("ny", 4),
                    nz=4,
                    t_end=0,
                    **options,
                )
        default = run_torch_solver_3d(preset_name="3d_ignition", nx=4, ny=4, nz=4)
        self.assertEqual(default["time"], 1e-5)
        with self.assertRaisesRegex(RuntimeError, "Maximum steps"):
            run_torch_solver_3d(nx=4, ny=4, nz=4, t_end=1, max_steps=1)

    def test_cli_volume_export_metadata_and_shuffled_slice_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "volume.csv"
            metadata = Path(directory) / "volume.json"
            command = [
                sys.executable,
                str(ROOT / "python/torchCli.py"),
                "--dim",
                "3d",
                "--preset",
                "3d_ignition",
                "--nx",
                "6",
                "--ny",
                "4",
                "--nz",
                "4",
                "--t_corte",
                "1e-6",
                "--out",
                str(source),
                "--json_out",
                str(metadata),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            meta = json.loads(metadata.read_text())
            self.assertEqual(meta["dimension"], "3d")
            self.assertEqual(meta["time"], 1e-6)
            self.assertEqual(meta["nz"], 4)
            with source.open() as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows), 1 + 6 * 4 * 4)
            self.assertIn("w", rows[0])
            self.assertIn("lambda", rows[0])
            with source.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(rows[0])
                writer.writerows(rows[:0:-1])
            # New Matplotlib releases inspect Figure.savefig while importing pyplot.
            # Complete that import before replacing the method in this export test.
            import matplotlib.pyplot as plt

            self.addCleanup(plt.close, "all")
            with patch("matplotlib.figure.Figure.savefig") as save:
                for axis in ("x", "y", "z"):
                    paths = plot_csv(
                        str(source),
                        str(Path(directory) / ("slice." + axis)),
                        slice_axis=axis,
                    )
                    self.assertTrue(paths[0].endswith("slice." + axis + ".png"))
                self.assertEqual(save.call_count, 6)
            with self.assertRaises(ValueError):
                plot_csv(str(source), slice_index=100)
            with source.open("a") as handle:
                handle.write(",".join(rows[1]) + "\n")
            with self.assertRaisesRegex(ValueError, "rectangular"):
                plot_csv(str(source))

    @unittest.skipUnless(shutil.which("octave"), "Octave unavailable")
    def test_matlab_python_inert_and_reactive_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            for preset, end in (
                ("3d_sod_y", 0.01),
                ("3d_explosion", 0.02),
                ("3d_ignition", 2e-6),
            ):
                target = Path(directory) / "state.csv"
                script = Path(directory) / "check.m"
                script.write_text(
                    f"addpath('{ROOT}/matlab'); r=torchSolver3D('preset_name','{preset}','nx',6,'ny',4,'nz',4,'t_end',{end}); dlmwrite('{target}',reshape(r.u_final,5,[]),',','precision',17);"
                )
                done = subprocess.run(
                    ["octave", "--quiet", "--no-gui", str(script)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(done.returncode, 0, done.stderr)
                r = run_torch_solver_3d(preset_name=preset, nx=6, ny=4, nz=4, t_end=end)
                np.testing.assert_allclose(
                    r["u_final"].reshape(5, -1, order="F"),
                    np.loadtxt(target, delimiter=","),
                    rtol=2e-12,
                    atol=1e-8,
                )


if __name__ == "__main__":
    unittest.main()
