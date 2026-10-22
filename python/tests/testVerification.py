"""Independent references and finite-volume averaging checks."""

import unittest

import numpy as np

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from cfdSolver.torchVerification import (
    acoustic_average,
    normal_shock_states,
    vortex_average,
    measure_case,
)


class ReferenceTests(unittest.TestCase):
    def test_1d_acoustics_have_second_order_mesh_convergence(self):
        errors = [
            measure_case("acoustic", n, 1, "hllc")["density_l1"]
            for n in (24, 48, 96)
        ]
        orders = np.log2(np.array(errors[:-1]) / errors[1:])
        self.assertTrue(np.all(orders > 1.7), (errors, orders))

    def test_normal_shock_conserves_flux_and_increases_entropy(self):
        for mach in (1.1, 2.0, 5.0):
            left, right = normal_shock_states(mach)
            fluxes = []
            for rho, u, p in (left, right):
                enthalpy = 1.4 / 0.4 * p / rho + 0.5 * u * u
                fluxes.append([rho * u, rho * u * u + p, rho * u * enthalpy])
            np.testing.assert_allclose(fluxes[0], fluxes[1], rtol=2e-15)
            self.assertGreater(right[-1] / right[0] ** 1.4, 1.0)
            self.assertLess(right[1] / np.sqrt(1.4 * right[-1] / right[0]), 1)
        _, right = normal_shock_states()
        self.assertAlmostEqual(right[0], 8 / 3)
        self.assertAlmostEqual(right[-1], 4.5)
        for mach in (0.5, 1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                normal_shock_states(mach)

    def test_vortex_advects_by_a_cell_and_extrudes_with_axial_momentum(self):
        initial = vortex_average(20)
        translated = vortex_average(20, 0.5)
        np.testing.assert_allclose(
            translated, np.roll(initial, (1, 1), axis=(1, 2)), atol=3e-15
        )
        volume = vortex_average(20, dimension=3)
        np.testing.assert_allclose(volume[0, 2], initial[0])
        np.testing.assert_allclose(volume[1:3, 2], initial[1:3])
        np.testing.assert_allclose(volume[3, 2], 0.25 * initial[0])
        np.testing.assert_allclose(
            volume[-1, 2], initial[-1] + 0.5 * 0.25**2 * initial[0]
        )
        self.assertLess(initial[0].min(), 0.7)
        self.assertTrue(np.all(initial[0] > 0))

    def test_acoustic_period_speed_and_cell_integral_in_all_dimensions(self):
        for dim in (1, 2, 3):
            period = 1 / (0.2 * dim + np.sqrt(1.4 * dim))
            initial = acoustic_average(12, dimension=dim)
            final = acoustic_average(12, period, dim)
            np.testing.assert_array_equal(initial, final)
            means = initial.reshape(dim + 2, -1).mean(axis=1)
            np.testing.assert_allclose(
                means, [1] + [0.2] * dim + [2.5 + 0.02 * dim], atol=2e-15
            )
            self.assertGreater(np.ptp(initial[0]), 1e-6)


if __name__ == "__main__":
    unittest.main()
