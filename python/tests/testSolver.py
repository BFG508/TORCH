"""
Automated Test Suite for the 1D CFD Solver Package.
Checks physical conversions and basic integration behavior.
"""

import sys
from pathlib import Path

# Allow importing cfdSolver from package root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
import unittest
import numpy as np

from cfdSolver import (
    gamma_calc, uconsv_to_uprim, uprim_to_uconsv,
    bc_creator, slope_limiter_calc,
    run_simulation, run_simulation_wQR,
)


class TestCFDSolver(unittest.TestCase):
    """ Unit and integration tests for physical correctness and stability. """

    def setUp(self) -> None:
        """ Treats all warnings as errors during test execution. """
        warnings.simplefilter("error")

    def test_physics_conversions(self) -> None:
        """ Tests roundtrip consistency between primitive and conservative variables. """
        gamma = 1.4
        rho = np.array([1.0, 0.5, 0.125])
        u = np.array([0.0, 100.0, -50.0])
        p = np.array([100000.0, 50000.0, 10000.0])

        u_prim_orig = np.vstack((rho, u, p))
        u_consv = uprim_to_uconsv(u_prim_orig, gamma)
        u_prim_rec = uconsv_to_uprim(u_consv, gamma)

        np.testing.assert_allclose(u_prim_rec, u_prim_orig, rtol=1e-10)

    def test_gamma_calc(self) -> None:
        """ Tests gamma calculations for pure reactants (lambda=1) and pure products (lambda=0). """
        gamma_unburned = gamma_calc(1.0)
        gamma_burned = gamma_calc(0.0)

        self.assertAlmostEqual(gamma_unburned, 1.4, places=5)
        self.assertAlmostEqual(gamma_burned, 9.0 / 7.0, places=5)

    def test_slope_limiters(self) -> None:
        """ Verifies all slope limiters (VanLeer, Albada, MINBEE, Superbee, Unos) execute correctly. """
        delta_i = np.array([[1.0, -0.5, 0.0], [2.0, 1.0, 0.5], [0.1, -0.1, 0.0]])
        di_uw = np.array([[0.8, -0.4, 0.1], [1.9, 0.9, 0.4], [0.05, -0.05, 0.0]])
        di_dw = np.array([[1.2, -0.6, -0.1], [2.1, 1.1, 0.6], [0.15, -0.15, 0.0]])
        w = np.array([0.0, 0.0, 0.0])
        c_cfl = 0.5

        for limiter in ["VanLeer", "Albada", "MINBEE", "Superbee", "Unos"]:
            delta_lim = slope_limiter_calc(delta_i, c_cfl, di_uw, di_dw, w, limiter)
            self.assertFalse(np.isnan(delta_lim).any(), f"NaN detected in {limiter}")
            self.assertFalse(np.isinf(delta_lim).any(), f"Inf detected in {limiter}")

        # Mathematical verification: At r = 1, w = 0, eps_r = 0.5, so slope limiters evaluate to 0.5
        d_linear = np.array([1.0])
        d_uw_1 = np.array([1.0])
        d_dw_1 = np.array([1.0])
        w_zero = np.array([0.0])
        for lim in ["VanLeer", "Albada", "MINBEE", "Superbee"]:
            lim_val = slope_limiter_calc(d_linear, c_cfl, d_uw_1, d_dw_1, w_zero, lim)
            self.assertAlmostEqual(float(lim_val[0]), 0.5, places=6, msg=f"{lim} failed at r=1")

        # Minbee at r = 2, w = 0: eps_r = 1/3, so limited slope is 1/3
        d_uw_2 = np.array([2.0])
        minbee_val = slope_limiter_calc(d_linear, c_cfl, d_uw_2, d_dw_1, w_zero, "MINBEE")
        self.assertAlmostEqual(float(minbee_val[0]), 1.0 / 3.0, places=5)

    def test_boundary_conditions(self) -> None:
        """ Verifies ghost cells for open and reflective boundary conditions. """
        u_inner = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]])

        # Open Boundary
        u_open = bc_creator(u_inner, "Open")
        self.assertEqual(u_open.shape, (3, 5))
        np.testing.assert_array_equal(u_open[:, 0], u_inner[:, 0])
        np.testing.assert_array_equal(u_open[:, -1], u_inner[:, -1])

        # Reflective Boundary
        u_refl = bc_creator(u_inner, "Reflective")
        self.assertEqual(u_refl.shape, (3, 5))
        self.assertEqual(u_refl[1, 0], -u_inner[1, 0])
        self.assertEqual(u_refl[1, -1], -u_inner[1, -1])

    def test_sod_shock_tube_simulation(self) -> None:
        """ Runs non-reactive Sod Shock Tube simulation and verifies numerical sanity. """
        u_evo, t_arr = run_simulation(
            l=1.0, n=50,
            rho_l=1.0, u_l=0.0, p_l=1.0,
            rho_r=0.125, u_r=0.0, p_r=0.1,
            gamma=1.4, c_cfl=0.5, t_corte=0.15,
            limiter="VanLeer", wave_speed_calc_option="Pressure Based",
            reflective_or_open="Open"
        )

        self.assertGreater(len(t_arr), 1)
        self.assertFalse(np.isnan(u_evo).any(), "NaN detected in Sod simulation")
        self.assertFalse(np.isinf(u_evo).any(), "Inf detected in Sod simulation")

    def test_reactive_detonation_simulation(self) -> None:
        """ Runs reactive H2-O2 detonation simulation and verifies physical bounds. """
        lambda_init = np.ones(50)
        u_evo, t_arr, lambda_evo = run_simulation_wQR(
            l=1.0, n=50,
            rho_l=1.0, u_l=0.0, p_l=1e6,
            rho_r=0.5, u_r=0.0, p_r=101325.0,
            lambda_init=lambda_init, c_cfl=0.5, t_corte=1e-5,
            limiter="Albada", wave_speed_calc_option="Enthalpy Based",
            reflective_or_open="Reflective"
        )

        self.assertGreater(len(t_arr), 1)
        self.assertAlmostEqual(t_arr[-1],1e-5)
        self.assertLess(float(lambda_evo[-1].min()),0.99)
        self.assertFalse(np.isnan(u_evo).any(), "NaN in reactive state")
        self.assertFalse(np.isnan(lambda_evo).any(), "NaN in reactant fraction")
    def test_exact_riemann_and_errors(self) -> None:
        """ Verifies Toro exact Riemann solver across shock and double-rarefaction wave patterns. """
        from cfdSolver import exact_riemann_sod, compute_error_norms
        x = np.linspace(0.0, 1.0, 100)
        # Toro Test 1: Sod shock tube (expansion left, shock right)
        rho_ex, u_ex, p_ex = exact_riemann_sod(x, t=0.2)
        self.assertEqual(len(rho_ex), 100)
        self.assertTrue(np.all(rho_ex > 0))
        self.assertTrue(np.all(p_ex > 0))
        l1, l2, linf = compute_error_norms(rho_ex, rho_ex)
        self.assertAlmostEqual(l1, 0.0)
        self.assertAlmostEqual(l2, 0.0)
        self.assertAlmostEqual(linf, 0.0)

        # Toro Test 2: Double Rarefaction wave (expansion left, expansion right)
        rho_2, u_2, p_2 = exact_riemann_sod(
            x, t=0.15, rho_l=1.0, u_l=-2.0, p_l=0.4,
            rho_r=1.0, u_r=2.0, p_r=0.4
        )
        self.assertTrue(np.all(rho_2 > 0))
        self.assertTrue(np.all(p_2 > 0))
        self.assertAlmostEqual(float(np.mean(u_2)), 0.0, places=2)  # Symmetric expansion

    def test_presets_and_audit(self) -> None:
        """ Verifies preset retrieval and conservation law auditing metrics. """
        from cfdSolver import get_preset, compute_conservation_audit, PRESETS
        self.assertIn("sod_standard", PRESETS)
        preset = get_preset("sod_standard")
        self.assertEqual(preset.mode, "sod")
        self.assertEqual(preset.n_cells, 100)

        u0 = np.ones((3, 50))
        u1 = np.ones((3, 50))
        audit = compute_conservation_audit(u0, u1, dx=0.02, execution_time_sec=0.1, total_steps=10, boundary_integral=np.zeros(3))
        self.assertIn("mass_conservation_error_rel", audit)
        self.assertAlmostEqual(audit["mass_conservation_error_rel"], 0.0)

    def test_2d_physics_conversions(self) -> None:
        """ Verifies 2D conservative to primitive conversions and speed of sound. """
        from cfdSolver import uconsv_to_uprim_2d, uprim_to_uconsv_2d, sound_speed_calc_2d
        rho = np.full((10, 10), 1.2)
        u = np.full((10, 10), 50.0)
        v = np.full((10, 10), -25.0)
        p = np.full((10, 10), 101325.0)
        u_prim_orig = np.stack((rho, u, v, p), axis=0)

        u_consv = uprim_to_uconsv_2d(u_prim_orig, gamma=1.4)
        u_prim_rec = uconsv_to_uprim_2d(u_consv, gamma=1.4)
        np.testing.assert_allclose(u_prim_rec, u_prim_orig, rtol=1e-10)

        a = sound_speed_calc_2d(u_prim_orig, gamma=1.4)
        self.assertTrue(np.all(a > 300.0))

    def test_2d_explosion_simulation(self) -> None:
        """ Runs 2D cylindrical blast wave simulation and verifies physical sanity. """
        from cfdSolver import run_torch_solver_2d
        res = run_torch_solver_2d(
            preset_name="2d_explosion",
            nx=25, ny=25,
            t_end=0.02,
            cfl=0.4,
            limiter="VanLeer"
        )
        self.assertGreater(res["steps"], 1)
        self.assertFalse(np.isnan(res["u_final"]).any(), "NaN detected in 2D explosion")
        self.assertFalse(np.isinf(res["u_final"]).any(), "Inf detected in 2D explosion")
        self.assertTrue(np.all(res["u_prim_final"][0] > 0.0), "Negative density in 2D explosion")
        self.assertTrue(np.all(res["u_prim_final"][3] > 0.0), "Negative pressure in 2D explosion")

    def test_2d_reactive_rde_simulation(self) -> None:
        """ Runs 2D reactive Strang-splitting RDE simulation and verifies combustion. """
        from cfdSolver import run_torch_solver_2d
        res = run_torch_solver_2d(
            preset_name="2d_rde_chamber",
            nx=20, ny=20,
            t_end=0.001,
            cfl=0.35,
            is_reactive=True,
            limiter="Superbee"
        )
        self.assertGreater(res["steps"], 1)
        self.assertFalse(np.isnan(res["u_final"]).any(), "NaN detected in 2D reactive RDE")
        self.assertTrue(np.all(res["lambda_final"] >= 0.0))
        self.assertTrue(np.all(res["lambda_final"] <= 1.0))


if __name__ == "__main__":
    unittest.main()
