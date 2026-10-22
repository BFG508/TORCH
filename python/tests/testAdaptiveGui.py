"""Adaptive desktop controls, worker lifecycle and disk playback."""

import copy
import os
from pathlib import Path
import tempfile
import time
import unittest

if __package__:
    from . import _bootstrap
else:
    import _bootstrap
from desktop.model import defaults, create_session


@unittest.skipUnless(os.environ.get("DISPLAY"), "Requires a graphical display")
class AdaptiveGuiTests(unittest.TestCase):
    def test_adaptive_controls_run_playback_and_restart(self):
        from desktop.app import SimulationApp

        with tempfile.TemporaryDirectory() as folder:
            app = SimulationApp()
            try:
                for d in (2, 3):
                    app.new()
                    spec = defaults(d)
                    spec.update(
                        counts=[8, 6, 4][:d],
                        adaptive=True,
                        workers=2,
                        amr=dict(max_level=1, regrid_every=1),
                        t_end=0.04,
                    )
                    app._apply_spec(spec)
                    app.vars["folder"].set(folder)
                    app.vars["interval"].set("1")
                    expected = create_session(copy.deepcopy(spec))
                    expected.run()
                    app.run_button.invoke()
                    deadline = time.monotonic() + 30
                    while app.runner.running and time.monotonic() < deadline:
                        app.update()
                        time.sleep(0.01)
                    for _ in range(8):
                        app.update()
                        time.sleep(0.02)
                    self.assertFalse(app.runner.running)
                    self.assertEqual(app.session.steps, expected.steps)
                    import numpy as np

                    np.testing.assert_array_equal(
                        app.session.state, expected.state
                    )
                    self.assertEqual(
                        str(app.entries["amr_max_level"]["state"]), "disabled"
                    )
                    self.assertEqual(
                        str(app.entries["workers"]["state"]), "normal"
                    )
                    app.field_var.set("Refinement level")
                    app.redraw()
                    app.frame_var.set(0)
                    app._history_index()
                    self.assertEqual(app.displayed.time, 0)
                    path = Path(folder) / "restart.h5"
                    app.session.save(path)
                    app.load(path)
                    app.vars["workers"].set("1")
                    app.vars["t_end"].set(".08")
                    app.run()
                    app.pause()
                    while app.runner.running:
                        app.update()
                        time.sleep(0.01)
                    expected.close()
            finally:
                app.close()
                while not app.closed:
                    app.update()
                    time.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
