"""Regression coverage for rejected inputs and trustworthy reference/export data."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfdSolver.torchAnalysis import exact_riemann_sod, plot_csv
from cfdSolver.torchSolver import new_lambda_n_e_calc, run_simulation, run_simulation_wqr
from cfdSolver.torchSolver2D import run_torch_solver_2d


class ValidationTests(unittest.TestCase):
    def test_invalid_1d_inputs_are_rejected_before_zero_time_return(self):
        initial = dict(l=1, n=8, rho_l=1, u_l=0, p_l=1,
                       rho_r=0.125, u_r=0, p_r=0.1, t_corte=0)
        for runner in (run_simulation, run_simulation_wqr):
            for override in (dict(p_l=-1), dict(rho_r=-1), dict(divider=np.nan),
                             dict(divider=150), dict(limiter='typo'),
                             dict(reflective_or_open='typo')):
                with self.subTest(runner=runner.__name__, override=override):
                    with self.assertRaises(ValueError):
                        runner(**(initial | override))
        for gamma in (0.5, 1, np.inf):
            with self.assertRaises(ValueError):
                run_simulation(**initial, gamma=gamma)
        with self.assertRaises(ValueError):
            run_simulation_wqr(**initial, kinetic_model='typo')

    def test_invalid_2d_inputs_are_rejected_before_initialization(self):
        for options in (dict(nx=2.5), dict(ambient_pressure=np.nan),
                        dict(gamma=0.5), dict(rho_init=1), dict(preset_name=None),
                        dict(limiter='typo')):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_torch_solver_2d(t_end=0, **options)
        with self.assertRaises(ValueError):
            run_torch_solver_2d(preset_name=None, rho_init=1, u_init=0, v_init=0,
                                p_init=1, nx=4, ny=4, bc_x='typo', t_end=0)

    def test_invalid_induction_fraction_and_state_shape(self):
        state = np.array([[1., 1.], [0., 0.], [2.5, 2.5]])
        for beta in ([-1, 1], [np.nan, 1], [1], [2, 1]):
            with self.assertRaises(ValueError):
                new_lambda_n_e_calc(np.ones(2), 0, state, 'znd', beta)
        with self.assertRaises(ValueError):
            new_lambda_n_e_calc(np.ones(3), 0, state)

    def test_exact_reference_preserves_float_results_for_integer_coordinates(self):
        actual = exact_riemann_sod(np.array([0, 1, 2]), t=0.2, x0=1)
        expected = exact_riemann_sod(np.array([0., 1., 2.]), t=0.2, x0=1)
        for values, reference in zip(actual, expected):
            self.assertEqual(values.dtype.kind, 'f')
            np.testing.assert_allclose(values, reference)
        self.assertGreater(actual[0][1], 0)
        with self.assertRaisesRegex(RuntimeError, 'converge'):
            exact_riemann_sod(np.array([0.5]), max_iter=1)
        with self.assertRaisesRegex(ValueError, 'Vacuum'):
            exact_riemann_sod(np.array([0.5]), u_l=-10, u_r=10)

    def test_plot_uses_coordinates_and_preserves_dotted_prefix(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'grid.csv'
            source.write_text('x,y,p,rho,u,v\n1,1,1,4,0,0\n0,0,1,1,0,0\n0,1,1,3,0,0\n1,0,1,2,0,0\n')
            with patch('matplotlib.pyplot.savefig') as save, patch('matplotlib.pyplot.close'):
                paths = plot_csv(str(source), str(Path(directory) / 'result.v1'))
                figure = plt.gcf()
                density = np.asarray(figure.axes[0].collections[0].get_array())
                np.testing.assert_array_equal(density.ravel(), [1, 2, 3, 4])
                self.assertTrue(paths[0].endswith('result.v1.png'))
                self.assertEqual(save.call_count, 2)
            plt.close(figure)
            source.write_text('x,y,p,rho,u,v\n0,0,1,1,0,0\n1,1,1,2,0,0\n')
            with self.assertRaisesRegex(ValueError, 'rectangular'):
                plot_csv(str(source), str(Path(directory) / "invalid"))

    def test_cli_rejects_wrong_dimension_preset_without_exporting(self):
        cli = Path(__file__).resolve().parents[1] / 'torchCli.py'
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'result.csv'
            for options in (['--dim', '2d', '--preset', 'sod_standard'],
                            ['--preset', '2d_explosion'],
                            ['--dim', '2d', '--study', 'convergence']):
                result = subprocess.run([sys.executable, str(cli), *options,
                                         '--out', str(target)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
