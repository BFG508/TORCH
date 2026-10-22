"""Verify Python-to-MATLAB-to-Python desktop continuations."""

import copy
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from cfdSolver import Simulation
from desktop.model import create_session, defaults


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for dim, geometry in ((2, "annular"), (3, "spherical")):
        spec = defaults(dim, geometry)
        spec.update(
            counts=[12, 8, 6][:dim],
            preset="Uniform",
            t_end=2e-5,
            reactive=True,
            pressure=1e6,
            velocities=[0.15] * dim,
        )
        session = create_session(spec)
        full = copy.deepcopy(session)
        full.run()
        full.save(folder / f"expected_{dim}.h5")
        session.run(step_limit=1)
        assert not session.completed
        session.save(folder / f"python_{dim}.h5")


def verify(folder):
    from desktop.app import SimulationApp

    for dim in (2, 3):
        expected = Simulation.load(folder / f"expected_{dim}.h5")
        matlab = Simulation.load(folder / f"matlab_{dim}.h5")
        np.testing.assert_allclose(
            matlab.state,
            expected.state,
            rtol=3e-10,
            atol=2e-8,
        )
        np.testing.assert_allclose(matlab.lam, expected.lam, atol=1e-12)
        viewed = Simulation.load(folder / f"matlab_displayed_{dim}.h5")
        assert viewed.time < matlab.time
        errors = []
        with patch(
            "desktop.app.messagebox.showerror",
            side_effect=lambda *a, **k: errors.append(str(a)),
        ):
            app = SimulationApp(dim)
            try:
                prefix = (folder / f"series_{dim}.txt").read_text()
                app.load(prefix + ".series.json", series=True)
                target = expected.time * 1.25
                direct = Simulation.load(
                    folder / f"matlab_{dim}.h5", t_end=target
                )
                direct.run()
                app.vars["t_end"].set(str(target))
                app.run_button.invoke()
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    app.update()
                    if not app.runner.running and app.runner.results.empty():
                        if str(app.run_button["state"]) != "disabled":
                            break
                    time.sleep(0.01)
                else:
                    raise AssertionError("Python desktop did not finish")
                assert not errors, errors
                np.testing.assert_array_equal(app.session.state, direct.state)
                np.testing.assert_array_equal(app.session.lam, direct.lam)
            finally:
                app.close()
                while not app.closed:
                    app.update()
                    time.sleep(0.01)
    print("PASS Python/MATLAB desktop restart, displayed export and resume")


if __name__ == "__main__":
    {"prepare": prepare, "verify": verify}[sys.argv[1]](Path(sys.argv[2]))
