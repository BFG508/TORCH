"""Regression tests for physical units, transport, CLI and integration
contracts.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

if __package__:
    from . import _bootstrap
else:
    import _bootstrap
from cfdSolver.torchSolver import (
    advect_scalar_1d,
    compute_conservation_audit,
    gamma_calc,
    new_lambda_n_e_calc,
    run_simulation,
    run_simulation_wqr,
    temp_calc,
    uprim_to_uconsv,
)
from cfdSolver.torchSolver2D import (
    apply_boundary_conditions_2d,
    gamma_calc_2d,
    reaction_step_2d,
    run_torch_solver_2d,
    uconsv_to_uprim_2d,
)

ROOT = Path(__file__).resolve().parents[2]


class RegressionTests(unittest.TestCase):
    def test_chemistry_si_temperature_and_dimension_parity(self):
        for temperature in (300.0, 1500.0):
            rho = 101325.0 * 12.0106666666667 / (8314.46 * temperature)
            state = uprim_to_uconsv(np.array([[rho], [40.0], [101325.0]]), 1.4)
            self.assertAlmostEqual(
                temp_calc(1.4, np.ones(1), state)[0], temperature
            )
            state2d = np.array(
                [
                    state[0],
                    state[1],
                    rho * np.ones(1) * 25,
                    state[2] + 0.5 * rho * 25**2,
                ]
            ).reshape(4, 1, 1)
            for model in ("arrhenius", "znd"):
                lam, reacted, beta = new_lambda_n_e_calc(
                    np.ones(1), 1e-6, state, model, np.ones(1)
                )
                out, lam2, beta2 = reaction_step_2d(
                    state2d, np.ones((1, 1)), np.ones((1, 1)), 1e-6, model
                )
                np.testing.assert_allclose(lam2.ravel(), lam, rtol=1e-13)
                np.testing.assert_allclose(beta2.ravel(), beta, rtol=1e-13)
                np.testing.assert_allclose(
                    (out - state2d)[3].ravel(), (reacted - state)[2], atol=1e-8
                )
                if temperature == 300:
                    self.assertGreater(lam[0], 1 - 1e-10)
        # Low SI pressure must never be interpreted as atmospheres.
        state = uprim_to_uconsv(np.array([[1.0], [0.0], [100.0]]), 1.4)
        self.assertLess(temp_calc(1.4, np.ones(1), state)[0], 1.0)

    def test_reaction_temporal_convergence(self):
        # Homogeneous smooth chemistry: halving dt should reduce
        # midpoint error ~4x.
        rho = 101325 * 12.0106666666667 / (8314.46 * 1500)
        initial = uprim_to_uconsv(np.array([[rho], [0.0], [101325.0]]), 1.4)

        def solve(steps):
            state, lam = initial.copy(), np.ones(1)
            for _ in range(steps):
                lam, state, _ = new_lambda_n_e_calc(lam, 1e-5 / steps, state)
            return lam[0]

        reference = solve(8192)
        errors = [abs(solve(n) - reference) for n in (64, 128, 256)]
        self.assertGreater(errors[0] / errors[1], 3.5)
        self.assertGreater(errors[1] / errors[2], 3.5)

    def test_periodic_reactive_transport_conserves_species(self):
        n = 60
        lam0 = np.zeros(n)
        lam0[10:20] = 1.0
        states, times, lam = run_simulation_wqr(
            1.0,
            n,
            1.0,
            2.0,
            100.0,
            1.0,
            2.0,
            100.0,
            lambda_init=lam0,
            t_corte=0.03,
            reflective_or_open="Periodic",
        )
        self.assertAlmostEqual(times[-1], 0.03)
        np.testing.assert_allclose(
            np.sum(states[0, :, -1] * lam[-1]),
            np.sum(states[0, :, 0] * lam0),
            rtol=1e-10,
        )
        self.assertGreater(
            np.sum(np.arange(n) * states[0, :, -1] * lam[-1]),
            np.sum(np.arange(n) * lam0),
        )
        # Induction uses the same conservative transport contract.
        old = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [10.0, 10.0, 10.0]])
        beta = advect_scalar_1d(
            np.array([1.0, 0.0, 0.0]),
            old,
            old,
            np.full(4, 2.0),
            0.25,
            "Periodic",
        )
        np.testing.assert_allclose(beta, [0.5, 0.5, 0.0])

    def test_all_limiters_reach_end_with_positive_internal_energy(self):
        for limiter in ("VanLeer", "Albada", "MINBEE", "Superbee", "Unos"):
            for waves in ("Pressure Based", "Enthalpy Based"):
                states, times = run_simulation(
                    1.0,
                    40,
                    1.0,
                    0.0,
                    1.0,
                    0.125,
                    0.0,
                    0.1,
                    t_corte=0.05,
                    limiter=limiter,
                    wave_speed_calc_option=waves,
                )
                self.assertAlmostEqual(times[-1], 0.05)
                self.assertTrue(
                    np.all(states[2] - 0.5 * states[1] ** 2 / states[0] > 0)
                )

    def test_smooth_hydrodynamic_mesh_convergence(self):
        errors = []
        for nx in (20, 40, 80):
            x = (np.arange(nx) + 0.5) / nx
            rho = np.tile(1 + 0.2 * np.sin(2 * np.pi * x), (4, 1))
            result = run_torch_solver_2d(
                preset_name=None,
                rho_init=rho,
                u_init=np.ones_like(rho),
                v_init=np.zeros_like(rho),
                p_init=np.ones_like(rho),
                nx=nx,
                ny=4,
                t_end=0.1,
                bc_x="Periodic",
                bc_y="Periodic",
            )
            errors.append(
                np.mean(
                    abs(
                        result["u_prim_final"][0]
                        - (1 + 0.2 * np.sin(2 * np.pi * (x - 0.1)))
                    )
                )
            )
        self.assertGreater(errors[0] / errors[1], 3.0)
        self.assertGreater(errors[1] / errors[2], 3.0)

    def test_2d_periodic_species_and_induction_transport(self):
        nx, ny = 20, 10
        rho = np.ones((ny, nx))
        lam = np.zeros((ny, nx))
        lam[2:5, 4:8] = 1
        beta = 1 - lam
        result = run_torch_solver_2d(
            preset_name=None,
            rho_init=rho,
            u_init=rho,
            v_init=0.5 * rho,
            p_init=100 * rho,
            nx=nx,
            ny=ny,
            t_end=0.02,
            bc_x="Periodic",
            bc_y="Periodic",
            is_reactive=True,
            kinetic_model="znd",
            lambda_init=lam,
            beta_init=beta,
        )
        species = result["u_final"][0] * result["lambda_final"]
        np.testing.assert_allclose(np.sum(species), np.sum(lam), atol=1e-10)
        self.assertGreater(
            np.sum(species * result["X"]), np.sum(lam * result["X"])
        )
        self.assertGreater(
            np.sum(species * result["Y"]), np.sum(lam * result["Y"])
        )
        induction = result["u_final"][0] * result["beta_final"]
        np.testing.assert_allclose(induction.sum(), beta.sum(), atol=1e-10)

    def test_2d_parameters_and_completion(self):
        result = run_torch_solver_2d(nx=12, ny=10, lx=2.0, ly=3.0, t_end=0.001)
        self.assertAlmostEqual(result["time"], 0.001)
        self.assertAlmostEqual(result["x"][-1], 2.0 - 1.0 / 12)
        self.assertAlmostEqual(result["y"][-1], 3.0 - 1.5 / 10)
        with self.assertRaisesRegex(RuntimeError, "Maximum steps"):
            run_torch_solver_2d(nx=12, ny=10, t_end=0.1, max_steps=1)
        with self.assertRaises(ValueError):
            run_torch_solver_2d(preset_name="misspelled")

    def test_reactive_eos_and_open_boundary_balance(self):
        result = run_torch_solver_2d(
            preset_name="2d_rde_chamber", nx=10, ny=10, t_end=2e-6
        )
        expected = uconsv_to_uprim_2d(
            result["u_final"], gamma_calc_2d(result["lambda_final"])
        )
        np.testing.assert_allclose(result["u_prim_final"], expected)
        self.assertLess(result["mass_conservation_error"], 1e-12)
        self.assertTrue(np.isfinite(result["exit_thrust_per_depth"]))
        padded = apply_boundary_conditions_2d(
            result["u_final"], "Periodic", "Inflow"
        )
        self.assertTrue(np.all(padded[2, :2, :] == 100.0))
        self.assertTrue(np.all(padded[0, :2, :] == 1.0))

    def test_audit_accounts_for_boundary_and_chemical_source(self):
        diagnostic = {}
        states, times = run_simulation(
            1.0,
            40,
            1.0,
            1.0,
            1.0,
            0.125,
            0.0,
            0.1,
            t_corte=0.1,
            diagnostics=diagnostic,
        )
        audit = compute_conservation_audit(
            states[:, :, 0],
            states[:, :, -1],
            1 / 40,
            0.0,
            len(times) - 1,
            **diagnostic,
        )
        self.assertLess(abs(audit["mass_conservation_error_rel"]), 1e-12)
        np.testing.assert_allclose(audit["balance_residual"], 0.0, atol=1e-12)
        self.assertGreater(abs(audit["mass_change_rel"]), 1e-5)
        diagnostic = {}
        states, times, _ = run_simulation_wqr(
            1.0,
            8,
            1.0,
            0.0,
            1e6,
            1.0,
            0.0,
            1e6,
            t_corte=1e-6,
            reflective_or_open="Periodic",
            diagnostics=diagnostic,
        )
        audit = compute_conservation_audit(
            states[:, :, 0],
            states[:, :, -1],
            1 / 8,
            0.0,
            len(times) - 1,
            **diagnostic,
        )
        np.testing.assert_allclose(audit["balance_residual"], 0.0, atol=1e-7)

    def test_cli_exact_periodic_and_explicit_preset_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "state.csv"
            meta = Path(directory) / "audit.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "python/torchCli.py"),
                    "--preset",
                    "sod_standard",
                    "--n",
                    "20",
                    "--t_corte",
                    "0.01",
                    "--cfl",
                    "0.3",
                    "--limiter",
                    "MINBEE",
                    "--exact_val",
                    "--bc",
                    "Periodic",
                    "--out",
                    str(data),
                    "--json_out",
                    str(meta),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(meta.read_text())
            self.assertEqual(result["cells"], 20)
            self.assertEqual(result["t_end"], 0.01)
            self.assertEqual(result["cfl"], 0.3)
            self.assertEqual(result["limiter"], "MINBEE")
            self.assertIn("linf_error", result["exact_validation"])
            self.assertEqual(len(data.read_text().splitlines()), 21)

    def test_cli_convergence_json_and_2d_export(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "study.json"
            command = [sys.executable, str(ROOT / "python/torchCli.py")]
            completed = subprocess.run(
                command
                + [
                    "--study",
                    "convergence",
                    "--t_corte",
                    "0.01",
                    "--out",
                    str(target),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                len(json.loads(target.read_text())["study_results"]), 5
            )
            completed = subprocess.run(
                command
                + [
                    "--dim",
                    "2d",
                    "--nx",
                    "8",
                    "--ny",
                    "6",
                    "--lx",
                    "2",
                    "--ly",
                    "3",
                    "--t_corte",
                    "0.001",
                    "--out",
                    str(Path(directory) / "data.csv"),
                    "--json_out",
                    str(target),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(target.read_text())
            self.assertEqual(result["t_end"], 0.001)
            self.assertEqual(result["nx"], 8)
            self.assertEqual(result["ny"], 6)
            self.assertLess(result["mass_conservation_error"], 1e-12)

    @unittest.skipUnless(shutil.which("octave"), "Octave unavailable")
    def test_matlab_python_reactive_driver_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            target1 = Path(directory) / "one.csv"
            target2 = Path(directory) / "two.csv"
            script = Path(directory) / "check.m"
            script.write_text(
                f"addpath('{ROOT}/matlab'); setupTorch;\n"
                "[U,t,lam]=torchSolver('reactive',1,12,1,0,1e6,.5,0,101325,"
                "ones(1,12),.4,1e-5,'VanLeer','Sound Speed Controlled',"
                "'Variante 1 - Original','Pressure Based','Open',50);\n"
                f"dlmwrite('{target1}',[U(:,:,end);lam(end,:)],',',"
                "'precision',17);\n"
                "[U,t,lam]=torchSolver2D('preset','2d_rde_chamber',12,10,2e-6,"
                ".4,'VanLeer');\n"
                f"dlmwrite('{target2}',[reshape(U,4,[]);reshape(lam,1,[])],"
                "',','precision',17);\n"
            )
            completed = subprocess.run(
                ["octave", "--quiet", "--no-gui", str(script)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            state, _, lam = run_simulation_wqr(
                1, 12, 1, 0, 1e6, 0.5, 0, 101325, c_cfl=0.4, t_corte=1e-5
            )
            np.testing.assert_allclose(
                np.vstack((state[:, :, -1], lam[-1])),
                np.loadtxt(target1, delimiter=","),
                rtol=1e-12,
                atol=1e-8,
            )
            result = run_torch_solver_2d(
                preset_name="2d_rde_chamber", nx=12, ny=10, t_end=2e-6, cfl=0.4
            )
            expected = np.vstack(
                (
                    result["u_final"].reshape(4, -1, order="F"),
                    result["lambda_final"].reshape(1, -1, order="F"),
                )
            )
            np.testing.assert_allclose(
                expected,
                np.loadtxt(target2, delimiter=","),
                rtol=1e-12,
                atol=1e-8,
            )

    @unittest.skipUnless(shutil.which("octave"), "Octave unavailable")
    def test_matlab_python_chemistry_parity(self):
        state = uprim_to_uconsv(
            np.array([[1.0, 1.0], [0.0, 0.0], [101325.0, 1e6]]),
            gamma_calc(np.array([1.0, 0.5])),
        )
        lam, expected, _ = new_lambda_n_e_calc(
            np.array([1.0, 0.5]), 1e-6, state
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.csv"
            script = Path(directory) / "check.m"
            literal = ";".join(
                " ".join(format(v, ".17g") for v in row) for row in state
            )
            script.write_text(
                (
                    f"addpath('{ROOT}/matlab'); setupTorch; "
                    f"[lam,U]=torchReaction([{literal}],[1,0.5],1e-6); "
                    f"dlmwrite('{target}',[U;lam],',','precision',17);"
                )
            )
            completed = subprocess.run(
                ["octave", "--quiet", "--no-gui", str(script)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            np.testing.assert_allclose(
                np.loadtxt(target, delimiter=","),
                np.vstack((expected, lam)),
                rtol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
