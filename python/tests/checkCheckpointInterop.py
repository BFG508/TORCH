"""Two-stage CI fixture exchange with MATLAB, including independent readers.

prepare DIR -> MATLAB testCheckpointInterop(DIR) -> verify DIR
"""

from pathlib import Path
import sys
import copy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfdSolver import Simulation


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for dim in (1, 2, 3):
        for reactive in (False, True):
            counts = [12, 6, 4][:dim]
            p = np.ones((dim + 2, *counts[::-1]))
            p[1:-1] = 0.2
            x = (np.arange(counts[0]) + 0.5) / counts[0]
            p[0] = 1 + 0.1 * np.sin(2 * np.pi * x)
            p[-1] = np.where(x < 0.5, 1e6, 101325) if reactive else 1.0
            session = Simulation(
                p,
                [1.0] * dim,
                t_end=1e-4 if reactive else 0.08,
                reactive=reactive,
                boundaries=["Periodic"] * dim,
            )
            key = f"{dim}_{int(reactive)}"
            full = copy.deepcopy(session)
            full.run()
            full.save(folder / f"expected_{key}.h5")
            session.run(step_limit=2)
            assert not session.completed
            session.save(folder / f"python_{key}.h5")
    s = Simulation.preset("3d_ignition", [4, 4, 4], kinetic_model="znd")
    s.save(folder / "znd.h5")


def verify(folder):
    from cfdSolver.torchIO import TemporalWriter
    import meshio
    from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy

    for dim in (1, 2, 3):
        for reactive in (False, True):
            key = f"{dim}_{int(reactive)}"
            expected = Simulation.load(folder / f"expected_{key}.h5")
            mid = Simulation.load(folder / f"matlab_{key}.h5")
            mid.run()
            np.testing.assert_allclose(mid.state, expected.state, rtol=2e-11, atol=1e-8)
            np.testing.assert_allclose(
                mid.boundary_integral, expected.boundary_integral, rtol=2e-10, atol=1e-8
            )
            np.testing.assert_allclose(
                mid.source_integral, expected.source_integral, rtol=2e-11, atol=1e-8
            )
            final = Simulation.load(folder / f"matlab_final_{key}.h5")
            np.testing.assert_allclose(
                final.state, expected.state, rtol=2e-11, atol=1e-8
            )
            mesh = meshio.read(folder / f"matlab_{key}.xdmf")
            np.testing.assert_array_equal(
                mesh.cell_data["rho"][0], final.state[0].ravel()
            )
            reader = vtkXMLRectilinearGridReader()
            reader.SetFileName(str(folder / f"matlab_{key}.vtr"))
            reader.Update()
            assert (
                reader.GetOutput().GetFieldData().GetArray("TimeValue").GetValue(0)
                == final.time
            )
            np.testing.assert_array_equal(
                vtk_to_numpy(reader.GetOutput().GetCellData().GetArray("energy")),
                final.state[-1].ravel(),
            )
            prefix = folder / f"history_{key}"
            with meshio.xdmf.TimeSeriesReader(str(prefix) + ".xdmf") as temporal:
                temporal.read_points_cells()
                time, _, fields = temporal.read_data(temporal.num_steps - 1)
                np.testing.assert_array_equal(fields["rho"][0], final.state[0].ravel())
                assert time == final.time
            # Python can append/reopen a MATLAB-produced series without losing metadata.
            TemporalWriter(prefix, final).append(final)
    print(
        "Python/MATLAB checkpoint exchange, restart and independent VTK/XDMF readers: PASS"
    )


if __name__ == "__main__":
    operation = sys.argv[1]
    folder = Path(sys.argv[2])
    {"prepare": prepare, "verify": verify}[operation](folder)
