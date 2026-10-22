"""Numerical equivalence and bounded allocation contracts."""

import copy
import gc
import tracemalloc
import unittest
from unittest.mock import patch

import numpy as np

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from cfdSolver import Simulation
from cfdSolver.torchFlux import hllc_flux, hllc_flux_grid
from cfdSolver.torchFiniteVolume import cfl_dt
from cfdSolver.torchVerification import conservative_average


class PerformanceContracts(unittest.TestCase):
    def test_batched_flux_matches_variable_gamma_flux_on_strided_grids(self):
        rng = np.random.default_rng(143)
        for dim in (2, 3):
            shape = (dim + 2, *([6] * dim))
            p = 1 + rng.random(shape)
            g = 1.2 + 0.3 * rng.random(shape[1:])
            left = conservative_average(p, g)[..., ::2]
            gr = g[..., ::2] + 0.05
            right = conservative_average(p * 1.05, g + 0.05)[..., ::2]
            gl = g[..., ::2]
            for direction in range(dim):
                m = direction + 1
                order = [0, m] + [k for k in range(1, dim + 1) if k != m]
                order += [dim + 1]
                reference = hllc_flux(
                    left[order].reshape(dim + 2, -1),
                    right[order].reshape(dim + 2, -1),
                    gl.ravel(),
                    gamma_right=gr.ravel(),
                ).reshape(left.shape)[np.argsort(order)]
                for batch in (1, 17, 8192):
                    actual = hllc_flux_grid(
                        left,
                        right,
                        gl,
                        gr,
                        direction=direction,
                        batch_size=batch,
                    )
                    np.testing.assert_array_equal(actual, reference)

    def test_batches_preserve_physical_flux_and_bound_kernel_work(self):
        p = np.ones((5, 9, 11, 17))
        p[1], p[2], p[3], p[-1] = 0.3, -0.2, 0.1, 2.0
        state = conservative_average(p)
        original = state.copy()
        sizes = []

        def flux(left, right, *args, **kwargs):
            sizes.append(left.shape[1])
            return hllc_flux(left, right, *args, **kwargs)

        with patch("cfdSolver.torchFlux.hllc_flux", side_effect=flux):
            for direction in range(3):
                result = hllc_flux_grid(
                    state, state, 1.4, direction=direction, batch_size=113
                )
                expected = state * p[direction + 1]
                expected[direction + 1] += p[-1]
                expected[-1] += p[direction + 1] * p[-1]
                np.testing.assert_allclose(result, expected, atol=1e-14)
        self.assertLessEqual(max(sizes), 113)
        np.testing.assert_array_equal(state, original)

    def test_hllc_peak_allocation_is_lower_on_large_faces(self):
        p = np.ones((5, 65536))
        p[1:-1] = 0.2
        state = conservative_average(p)
        peaks = []
        for callback in (hllc_flux, hllc_flux_grid):
            gc.collect()
            tracemalloc.start()
            try:
                output = callback(state, state, 1.4)
                peaks.append(tracemalloc.get_traced_memory()[1])
                del output
            finally:
                tracemalloc.stop()
        self.assertLess(peaks[1], 0.6 * peaks[0], peaks)

    def test_session_memory_does_not_grow_with_step_history(self):
        p = np.ones((4, 12, 16))
        p[1:-1] = 0.2
        s = Simulation(
            p,
            [1, 1],
            t_end=10,
            face_boundaries={"x_min": "wall", "x_max": "wall"},
        )
        s.run(step_limit=2)
        peaks = []
        for steps in (2, 30):
            session = copy.deepcopy(s)
            gc.collect()
            tracemalloc.start()
            try:
                session.run(step_limit=steps)
                peaks.append(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
        self.assertLess(peaks[1], 1.4 * peaks[0], peaks)

    def test_cfl_includes_prescribed_hot_reactive_inlet(self):
        p = np.ones((4, 8, 12))
        p[1:-1] = 0
        s = Simulation(
            p,
            [1, 1],
            t_end=0.1,
            reactive=True,
            face_boundaries={
                "x_min": dict(
                    type="inflow",
                    primitive=[1, 100, 0, 1e6],
                    **{"lambda": 0.0},
                )
            },
        )
        gamma_inlet = 4.5 / 3.5
        expected_rate = (100 + np.sqrt(gamma_inlet * 1e6)) * 12 + np.sqrt(
            1.4
        ) * 8
        self.assertAlmostEqual(
            cfl_dt(s.state, s.lam, s.config, s.metrics),
            0.4 / expected_rate,
            places=16,
        )


if __name__ == "__main__":
    unittest.main()
