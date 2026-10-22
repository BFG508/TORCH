"""Exchange AMR trees and scientific exports with native MATLAB.
"""

import copy
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import h5py

import numpy as np

if __package__:
    from . import _bootstrap
    from .testGeometry import example
else:
    import _bootstrap
    from testGeometry import example
from cfdSolver import Simulation


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    keys = []
    for d in (1, 2, 3):
        for geometry in ("cartesian", "cylindrical", "annular", "spherical"):
            for reactive in (False, True):
                key = f"{d}_{geometry}_{int(reactive)}"
                keys.append(key)
                s = example(d, geometry, reactive).to_adaptive(
                    max_level=1, regrid_every=1
                )
                full = copy.deepcopy(s)
                full.run()
                full.save(folder / f"expected_{key}.h5")
                s.step()
                assert not s.completed, key
                s.save(folder / f"python_{key}.h5")
    (folder / "cases.txt").write_text("\n".join(keys))


def verify(folder):
    import meshio
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy

    for key in (folder / "cases.txt").read_text().splitlines():
        expected = Simulation.load(folder / f"expected_{key}.h5")
        final = Simulation.load(folder / f"matlab_final_{key}.h5")
        partial = Simulation.load(folder / f"matlab_{key}.h5")
        partial.set_workers(2)
        partial.run()
        for actual in (final, partial):
            np.testing.assert_array_equal(
                actual.mesh.keys, expected.mesh.keys, err_msg=key
            )
            for attr in ("state", "boundary_integral", "source_integral"):
                np.testing.assert_allclose(
                    getattr(actual, attr),
                    getattr(expected, attr),
                    rtol=5e-10,
                    atol=3e-8,
                    err_msg=key + " " + attr,
                )
            if actual.lam is not None:
                np.testing.assert_allclose(
                    actual.lam, expected.lam, rtol=5e-10, atol=2e-12
                )
            actual.close()
        collection = ET.parse(folder / f"history_{key}.xdmf")
        for grid in collection.findall("./Domain/Grid/Grid"):
            topology = grid.find("Topology")
            count = int(topology.get("NumberOfElements"))
            filename = topology.find("DataItem").text.split(":")[0]
            with h5py.File(folder / filename) as handle:
                assert handle["mesh/cells"].shape[0] == count
                assert handle["state/rho"].shape == (count,)
        mesh = meshio.read(folder / f"matlab_{key}.xdmf")
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(folder / f"matlab_{key}.vtu"))
        reader.Update()
        vtk = reader.GetOutput()
        assert vtk.GetNumberOfCells() == len(final.mesh.keys)
        for name in ("rho", "pressure", "refinement_level", "cell_volume"):
            np.testing.assert_allclose(
                vtk_to_numpy(vtk.GetCellData().GetArray(name)),
                mesh.cell_data[name][0],
                rtol=1e-14,
            )
    print("PASS 24 adaptive Python/MATLAB continuations and independent IO")


if __name__ == "__main__":
    globals()[sys.argv[1]](Path(sys.argv[2]))
