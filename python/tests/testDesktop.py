"""Desktop models, worker ownership, real widgets and disk playback."""

import copy
from pathlib import Path
from queue import Empty
from threading import Event
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from cfdSolver import Simulation
from cfdSolver.torchIO import TemporalWriter
from desktop.controller import SessionRunner
from desktop.export import export_session
from desktop.model import (
    config_spec,
    create_session,
    defaults,
    face_keys,
    history_frames,
    presets,
)
from desktop.view import draw_session, session_plane


class DesktopModelTests(unittest.TestCase):
    def test_all_geometry_presets_have_valid_local_states(self):
        for dim in (2, 3):
            for geometry in (
                "cartesian",
                "cylindrical",
                "annular",
                "spherical",
            ):
                for name in presets(dim, geometry):
                    spec = defaults(dim, geometry)
                    spec.update(counts=[8, 6, 4][:dim], preset=name)
                    s = create_session(spec)
                    self.assertEqual(
                        s.state.shape, (dim + 2, *spec["counts"][::-1])
                    )
                    self.assertTrue(np.isfinite(s.state).all())
                    self.assertTrue(np.all(s.primitive[[0, -1]] > 0))
                    if geometry != "cartesian":
                        np.testing.assert_allclose(
                            s.primitive[0], spec["density"]
                        )
                    if name == "Rotating equilibrium":
                        p = s.primitive
                        speed = np.sqrt(np.sum(p[1:-1] ** 2, axis=0))
                        np.testing.assert_allclose(
                            p[-1],
                            spec["pressure"] + 0.5 * p[0] * speed**2,
                        )

    def test_slice_orientation_and_cell_centres_on_every_axis(self):
        x = 1 + (np.arange(7) + 0.5) * 2 / 7
        y = 2 + (np.arange(5) + 0.5) * 3 / 5
        z = 3 + (np.arange(3) + 0.5) * 4 / 3
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        p = np.ones((5, 3, 5, 7))
        p[1:-1] = 0
        p[-1] = 10 + xx + 10 * yy + 100 * zz
        s = Simulation(p, [2.0, 3.0, 4.0], origins=[1.0, 2.0, 3.0], t_end=0.1)
        for normal in range(3):
            index = 1
            plane = session_plane(s, normal=normal, index=index)
            expected = np.take(p[-1], index, axis=2 - normal)
            np.testing.assert_allclose(plane.values, expected)
            np.testing.assert_allclose(
                plane.profile,
                expected[expected.shape[0] // 2],
            )
            np.testing.assert_allclose(
                plane.vertices[..., normal],
                (x, y, z)[normal][index],
            )
        with self.assertRaises(ValueError):
            session_plane(s, normal=2, index=100)
        spec = defaults(2, "annular")
        spec.update(counts=[8, 12])
        annulus = session_plane(create_session(spec))
        radius = np.sqrt(
            annulus.vertices[..., 0] ** 2 + annulus.vertices[..., 1] ** 2
        )
        np.testing.assert_allclose(radius[:, 0], 0.5)
        np.testing.assert_allclose(radius[:, -1], 1.0)

    def test_worker_pause_restart_is_exact_and_progress_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = defaults(2)
            spec.update(counts=[12, 8], t_end=0.2)
            s = create_session(spec)
            expected = copy.deepcopy(s)
            expected.run()
            entered, release = Event(), Event()
            original = Simulation.step

            def controlled(session):
                original(session)
                if session.steps == 2:
                    entered.set()
                    release.wait(10)

            runner = SessionRunner()
            prefix = Path(directory) / "history"
            with patch.object(Simulation, "step", controlled):
                runner.start(s, prefix, 1)
                try:
                    self.assertTrue(entered.wait(10))
                    runner.request_pause()
                finally:
                    release.set()
                result = runner.results.get(timeout=10)
                runner.thread.join(timeout=10)
            self.assertFalse(result.error)
            self.assertEqual(result.session.steps, 2)
            self.assertFalse(result.session.completed)
            self.assertLessEqual(runner.progress.qsize(), 1)
            data = history_frames(prefix)
            self.assertEqual(len(data["frames"]), 3)
            resumed = Simulation.load(
                prefix.parent / (data["frames"][-1]["stem"] + ".h5")
            )
            runner.start(resumed, prefix, 2)
            result = runner.results.get(timeout=10)
            runner.thread.join(timeout=10)
            self.assertFalse(result.error)
            np.testing.assert_array_equal(result.session.state, expected.state)
            np.testing.assert_array_equal(
                result.session.boundary_integral,
                expected.boundary_integral,
            )
            times = [f["time"] for f in history_frames(prefix)["frames"]]
            self.assertEqual(times, sorted(set(times)))

    def test_existing_collections_and_worker_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = defaults(2)
            spec.update(counts=[8, 6], t_end=0.04)
            s = create_session(spec)
            prefix = Path(directory) / "history"
            TemporalWriter(prefix, s).append(s)
            runner = SessionRunner()
            runner.start(s, prefix, 1)
            result = runner.results.get(timeout=10)
            runner.thread.join(timeout=10)
            self.assertFalse(result.error)
            self.assertEqual(
                history_frames(prefix)["formats"], ["vtk", "xdmf"]
            )
            other = create_session(spec)
            runner.start(other, prefix, 1)
            result = runner.results.get(timeout=10)
            runner.thread.join(timeout=10)
            self.assertIn("another run", result.error)
            self.assertEqual(other.steps, 0)
            with self.assertRaises(ValueError):
                runner.start(other, Path(directory) / "bad", 0)

    def test_exports_and_view_rendering_preserve_snapshot(self):
        import meshio

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for dim in (2, 3):
                for geometry in ("cartesian", "annular", "spherical"):
                    spec = defaults(dim, geometry)
                    spec.update(counts=[8, 6, 4][:dim], t_end=1e-6)
                    s = create_session(spec)
                    s.run()
                    original = s.state.copy()
                    export_session(folder / "frame.csv", s)
                    export_session(folder / "frame.h5", s)
                    export_session(folder / "frame.xdmf", s)
                    suffix = ".vtr" if geometry == "cartesian" else ".vts"
                    export_session(folder / ("frame" + suffix), s)
                    mesh = meshio.read(folder / "frame.xdmf")
                    np.testing.assert_array_equal(
                        mesh.cell_data["rho"][0],
                        s.state[0].ravel(),
                    )
                    np.testing.assert_array_equal(
                        Simulation.load(folder / "frame.h5").state,
                        original,
                    )
                    figure = Figure(figsize=(7, 7))
                    for mode in ("Map", "3D plane"):
                        draw_session(figure, s, mode=mode)
                        figure.savefig(folder / "frame.png")
                        self.assertGreater(
                            (folder / "frame.png").stat().st_size, 1000
                        )
                    np.testing.assert_array_equal(s.state, original)
            with self.assertRaises(ValueError):
                export_session(folder / "invalid.txt", s)

    def test_saved_face_conditions_are_reflected_in_the_editor_model(self):
        spec = defaults(2, "annular")
        spec["counts"] = [8, 6]
        spec["face_edits"] = dict(
            x_min=dict(type="inflow", primitive=[1.0, 20.0, 0.0, 101325.0]),
            x_max=dict(type="pressure_outlet", pressure=101325.0),
        )
        s = create_session(spec)
        restored = config_spec(s)
        self.assertEqual(restored["face_edits"]["x_min"]["type"], "inflow")
        self.assertEqual(restored["counts"], spec["counts"])


class DesktopGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            root = tk.Tk()
            root.destroy()
        except tk.TclError as error:
            raise unittest.SkipTest(str(error))
        from desktop.app import SimulationApp

        cls.app_class = SimulationApp

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.errors = []
        self.dialog = patch(
            "desktop.app.messagebox.showerror",
            side_effect=lambda *a, **k: self.errors.append(str(a)),
        )
        self.dialog.start()
        self.addCleanup(self.dialog.stop)
        self.app = self.app_class(2)
        self.app.vars["folder"].set(self.directory.name)
        self.app.vars["interval"].set("1")
        self.app.vars["nx"].set("12")
        self.app.vars["ny"].set("8")
        self.app.vars["t_end"].set(".04")
        self.app.update()
        self.addCleanup(self.close_app)

    def close_app(self):
        app = self.app
        if not app.closed:
            app.close()
            deadline = time.monotonic() + 15
            while not app.closed and time.monotonic() < deadline:
                app.update()
                time.sleep(0.01)

    def wait_idle(self):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            self.app.update()
            if not self.app.runner.running and self.app.runner.results.empty():
                if str(self.app.run_button["state"]) != "disabled":
                    break
            time.sleep(0.01)
        else:
            self.fail("Desktop simulation did not return to idle")
        self.assertFalse(self.errors)

    def test_run_browse_save_latest_export_displayed_and_resume(self):
        app = self.app
        expected = create_session(app.specification())
        expected.run()
        app.run_button.invoke()
        self.wait_idle()
        np.testing.assert_array_equal(app.session.state, expected.state)
        self.assertGreaterEqual(len(app.frames), 2)
        self.assertEqual(str(app.entries["nx"]["state"]), "disabled")
        app.frame_var.set(0)
        app._history_index()
        self.assertEqual(app.displayed.time, 0)
        checkpoint = Path(self.directory.name) / "current.h5"
        with patch(
            "desktop.app.filedialog.asksaveasfilename",
            return_value=str(checkpoint),
        ):
            app.file_button_5.invoke()
        saved = Simulation.load(checkpoint)
        self.assertEqual(saved.time, app.session.time)
        csv = Path(self.directory.name) / "displayed.csv"
        with patch(
            "desktop.app.filedialog.asksaveasfilename", return_value=str(csv)
        ):
            app.file_button_6.invoke()
        data = np.genfromtxt(csv, delimiter=",", names=True)
        np.testing.assert_array_equal(
            data["p"], app.displayed.primitive[-1].ravel()
        )
        image = Path(self.directory.name) / "displayed.png"
        with patch(
            "desktop.app.filedialog.asksaveasfilename", return_value=str(image)
        ):
            app.file_button_7.invoke()
        self.assertGreater(image.stat().st_size, 1000)
        saved.config["t_end"] = 0.08
        saved.run()
        app.vars["t_end"].set(".08")
        app.run_button.invoke()
        self.wait_idle()
        np.testing.assert_array_equal(app.session.state, saved.state)
        app.play_button.invoke()
        self.assertIsNotNone(app.play_job)
        app.new_button.invoke()
        self.assertIsNone(app.play_job)
        self.assertIsNone(app.session)
        self.assertEqual(str(app.entries["nx"]["state"]), "normal")

    def test_3d_axes_curved_surface_and_reactive_fields(self):
        app = self.app
        app._apply_spec(defaults(3, "spherical"))
        for a, n in zip("xyz", (8, 6, 4)):
            app.vars["n" + a].set(str(n))
        app.vars["t_end"].set("1e-6")
        app.vars["pressure"].set("1e6")
        app.reactive.set(True)
        app.run_button.invoke()
        self.wait_idle()
        self.assertIn("Reactant fraction", app.field_combo["values"])
        self.assertEqual(app.slice_var.get(), 2)
        app.axis_var.set("r")
        app._slice_axis(event=True)
        app.slice_var.set(5)
        app.field_var.set("Density")
        app.redraw()
        np.testing.assert_allclose(
            app.last_plane.values,
            app.displayed.primitive[0, :, :, 5],
        )
        app.mode_var.set("3D plane")
        app.redraw()
        self.assertEqual(app.figure.axes[0].name, "3d")
        radius = np.linalg.norm(app.last_plane.vertices, axis=-1)
        expected = 0.5 + (5 + 0.5) * 0.5 / 8
        np.testing.assert_allclose(radius, expected)
        self.assertFalse(self.errors)

    def test_pause_is_responsive_and_resume_uses_complete_steps(self):
        app = self.app
        app.vars["t_end"].set(".2")
        entered, release = Event(), Event()
        original = Simulation.step

        def controlled(session):
            original(session)
            if session.steps == 2:
                entered.set()
                release.wait(10)

        with patch.object(Simulation, "step", controlled):
            app.run_button.invoke()
            try:
                self.assertTrue(entered.wait(10))
                app.pause_button.invoke()
                app.update()
                self.assertIn("Pausing", app.status.get())
            finally:
                release.set()
            self.wait_idle()
        self.assertEqual(app.session.steps, 2)
        self.assertFalse(app.session.completed)
        self.assertEqual(app.displayed.steps, 2)
        self.assertEqual(app.frames[-1]["step"], 2)
        self.assertEqual(str(app.file_button_5["state"]), "normal")

    def test_invalid_inputs_and_corrupt_load_preserve_current_state(self):
        app = self.app
        app.vars["nx"].set("3.5")
        app.run_button.invoke()
        self.assertTrue(self.errors)
        self.assertIsNone(app.session)
        self.assertFalse(app.runner.running)
        self.errors.clear()
        app.vars["nx"].set("8")
        app.run_button.invoke()
        self.wait_idle()
        previous = app.session
        corrupt = Path(self.directory.name) / "corrupt.h5"
        corrupt.write_text("not a checkpoint")
        with patch(
            "desktop.app.filedialog.askopenfilename", return_value=str(corrupt)
        ):
            app.file_button_3.invoke()
        self.assertIs(app.session, previous)
        self.assertTrue(self.errors)
        self.assertEqual(str(app.run_button["state"]), "normal")

    def test_open_series_restore_faces_and_close_while_running(self):
        app = self.app
        spec = defaults(2, "annular")
        spec.update(counts=[8, 6], t_end=1e-5)
        s = create_session(spec)
        prefix = Path(self.directory.name) / "outside"
        s.run(output_prefix=prefix, output_every=1)
        app.load(str(prefix) + ".series.json", series=True)
        self.assertEqual(app.session.run_id, s.run_id)
        self.assertEqual(app.vars["geometry"].get(), "annular")
        app.face_var.set("r minimum")
        app._show_face()
        self.assertEqual(app.vars["face_type"].get(), "Outflow")
        target = s.time + 1e-5
        app.vars["t_end"].set(str(target))
        app.run_button.invoke()
        self.wait_idle()
        self.assertEqual(app.session.time, target)
        app.new_button.invoke()
        app._apply_spec(defaults(2))
        app.vars["folder"].set(self.directory.name)
        app.vars["t_end"].set("1")
        entered, release = Event(), Event()
        original = Simulation.step

        def controlled(session):
            original(session)
            entered.set()
            release.wait(10)

        with patch.object(Simulation, "step", controlled):
            app.run_button.invoke()
            try:
                self.assertTrue(entered.wait(10))
                app.close()
                self.assertTrue(app.closing)
            finally:
                release.set()
            deadline = time.monotonic() + 10
            while not app.closed and time.monotonic() < deadline:
                app.update()
                time.sleep(0.01)
            self.assertTrue(app.closed)
        self.assertFalse(app.runner.running)


if __name__ == "__main__":
    unittest.main()
