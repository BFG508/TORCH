"""v1.2 Python/MATLAB metric, boundary, checkpoint and export interchange."""

from pathlib import Path
import copy
import json
import sys
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cfdSolver import Simulation
from cfdSolver.torchIO import TemporalWriter, _mesh
from testGeometry import example


def cases():
    for dim in (1, 2, 3):
        for geometry in ("cartesian", "cylindrical", "annular", "spherical"):
            for reactive in (False, True):
                yield f"{dim}_{geometry}_{int(reactive)}", example(
                    dim, geometry, reactive
                )
        shape = tuple([6] * dim)
        p = np.ones((dim + 2, *shape))
        p[1:-1] = 0
        p[1] = 0.1
        faces = {
            "x_min": dict(
                type="stagnation_inlet",
                total_pressure=1.1,
                total_temperature=1.0,
                gas_constant=1.0,
            ),
            "x_max": dict(
                type="pressure_outlet",
                pressure=1.0,
                backflow=[1.0, -0.1] + [0.0] * (dim - 1) + [1.0],
            ),
        }
        yield f"{dim}_reservoir", Simulation(
            p, [1.0] * dim, t_end=0.15, face_boundaries=faces
        )


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    keys = []
    for key, s in cases():
        keys.append(key)
        full = copy.deepcopy(s)
        full.run()
        full.save(folder / f"expected_{key}.h5")
        s.run(step_limit=1)
        assert not s.completed, key
        s.save(folder / f"python_{key}.h5")
    (folder / "cases.txt").write_text("\n".join(keys))
    legacy = Simulation.preset("3d_sod_x", [6, 4, 4], t_end=0.04)
    full = copy.deepcopy(legacy)
    full.run()
    full.save(folder / "expected_legacy.h5")
    legacy.run(step_limit=1)
    legacy.save(folder / "legacy_schema1.h5")
    with h5py.File(folder / "legacy_schema1.h5", "r+") as handle:
        meta = json.loads(bytes(handle["metadata"][:]).decode())
        meta["schema_version"], meta["solver_version"] = 1, "1.1"
        del meta["config"]["geometry"]
        del meta["config"]["origins_xyz"]
        del handle["metadata"]
        handle["metadata"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)


def verify(folder):
    import meshio
    from vtkmodules.vtkIOXML import (
        vtkXMLStructuredGridReader,
        vtkXMLRectilinearGridReader,
    )
    from vtkmodules.util.numpy_support import vtk_to_numpy

    for key in (folder / "cases.txt").read_text().splitlines():
        expected = Simulation.load(folder / f"expected_{key}.h5")
        final = Simulation.load(folder / f"matlab_final_{key}.h5")
        middle = Simulation.load(folder / f"matlab_{key}.h5")
        middle.run()
        for actual in (final, middle):
            for attr in ("state", "boundary_integral", "source_integral"):
                np.testing.assert_allclose(
                    getattr(actual, attr),
                    getattr(expected, attr),
                    rtol=3e-10,
                    atol=2e-8,
                    err_msg=key + " " + attr,
                )
            if actual.lam is not None:
                np.testing.assert_allclose(
                    actual.lam, expected.lam, rtol=3e-10, atol=2e-12
                )
        np.testing.assert_allclose(final.balance_residual, 0, atol=2e-7, err_msg=key)
        curved = final.config.get("geometry", "cartesian") != "cartesian"
        reader = (
            vtkXMLStructuredGridReader() if curved else vtkXMLRectilinearGridReader()
        )
        reader.SetFileName(str(folder / f'matlab_{key}{".vts" if curved else ".vtr"}'))
        reader.Update()
        mesh = meshio.read(folder / f"matlab_{key}.xdmf")
        np.testing.assert_allclose(
            mesh.points, _mesh(final)[1], atol=2e-15, err_msg=key
        )
        np.testing.assert_array_equal(mesh.cells[0].data, _mesh(final)[2])
        np.testing.assert_array_equal(
            vtk_to_numpy(reader.GetOutput().GetCellData().GetArray("rho")),
            final.state[0].ravel(),
        )
        with meshio.xdmf.TimeSeriesReader(
            str(folder / f"history_{key}.xdmf")
        ) as series:
            series.read_points_cells()
            time, _, data = series.read_data(series.num_steps - 1)
            assert time == final.time
            np.testing.assert_array_equal(data["rho"][0], final.state[0].ravel())
        TemporalWriter(folder / f"history_{key}", final).append(final)
    legacy = Simulation.load(folder / "matlab_legacy.h5")
    expected = Simulation.load(folder / "expected_legacy.h5")
    np.testing.assert_allclose(legacy.state, expected.state, rtol=2e-11, atol=1e-12)
    with h5py.File(folder / "matlab_legacy.h5") as handle:
        assert json.loads(bytes(handle["metadata"][:]).decode())["schema_version"] == 2
    print(
        "PASS 27 geometry/boundary Python-MATLAB continuations and independent exports"
    )


if __name__ == "__main__":
    {"prepare": prepare, "verify": verify}[sys.argv[1]](Path(sys.argv[2]))
