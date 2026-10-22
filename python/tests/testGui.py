"""Exercise real Tk widgets when a graphical display is available."""

import tkinter as tk
import unittest
from unittest.mock import patch

if __package__:
    from . import _bootstrap
else:
    import _bootstrap


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            root = tk.Tk()
            root.destroy()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Graphical display unavailable: {error}")
        from torchGui import CFDApp

        cls.app_class = CFDApp

    def setUp(self):
        self.app = self.app_class()
        self.app.update()
        self.addCleanup(self.app.destroy)
        self.app.cmb_preset.set("sod_standard")
        self.app.on_preset_selected()
        self.set_entry(self.app.ent_n, "20")
        self.set_entry(self.app.ent_tcorte, "0.01")

    @staticmethod
    def set_entry(entry, value):
        entry.delete(0, tk.END)
        entry.insert(0, value)

    def test_pause_and_reset_cancel_pending_animation(self):
        self.app.on_run()
        self.app.reset_animation()
        self.app.toggle_animation()
        self.assertIsNotNone(self.app.animation_after_id)
        self.app.toggle_animation()
        self.assertIsNone(self.app.animation_after_id)
        self.assertFalse(self.app.is_animating)
        self.app.toggle_animation()
        self.app.reset_animation()
        self.assertIsNone(self.app.animation_after_id)
        self.assertEqual(self.app.current_frame, 0)

    def test_zero_time_disables_playback(self):
        self.set_entry(self.app.ent_tcorte, "0")
        self.app.on_run()
        self.app.toggle_animation()
        self.assertFalse(self.app.is_animating)
        self.assertEqual(self.app.current_frame, 0)
        self.assertEqual(str(self.app.btn_play["state"]), "disabled")

    def test_rejected_run_preserves_results_and_coordinates(self):
        self.app.on_run()
        state, grid = self.app.u_evo, self.app.x_coords
        self.set_entry(self.app.ent_n, "0")
        with patch("torchGui.messagebox.showerror") as dialog:
            self.app.on_run()
        dialog.assert_called_once()
        self.assertIs(self.app.u_evo, state)
        self.assertIs(self.app.x_coords, grid)
        self.app.render_frame(0)

    def test_reactive_plot_and_time_label(self):
        self.app.cmb_preset.set("detonation_cj")
        self.app.on_preset_selected()
        self.set_entry(self.app.ent_n, "20")
        self.set_entry(self.app.ent_tcorte, "1e-5")
        self.app.on_run()
        self.assertIsNotNone(self.app.lambda_evo)
        self.assertEqual(len(self.app.ax_extra.lines), 1)
        self.assertIn("1e-05", self.app.ax_extra.get_title())


if __name__ == "__main__":
    unittest.main()
