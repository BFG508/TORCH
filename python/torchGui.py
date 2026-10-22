"""
GUI Application: Interactive Graphical User Interface for the 1D CFD Solver using Tkinter.
Includes animation controls, frame scrubbing, preset selection, live conservation audit,
and optional exact Toro Riemann solver validation overlay.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from cfdSolver import (
    run_simulation, run_simulation_wQR, compute_conservation_audit,
    PRESETS, get_preset, exact_riemann_sod, uconsv_to_uprim, gamma_calc
)

from cfdSolver.torchSolver import validate_grid

matplotlib.use("TkAgg")


class CFDApp(tk.Tk):
    """ Main application window for interactive 1D CFD simulations. """

    def __init__(self) -> None:
        super().__init__()
        self.title("INTERFACE 1D - CFD Solver (Euler + MUSCL + HLLC + Arrhenius/ZND + Audit)")
        self.geometry("1350x880")

        self.u_evo = None
        self.t_arr = None
        self.lambda_evo = None
        self.x_coords = None
        self.current_frame = 0
        self.is_animating = False
        self.animation_after_id = None

        self.create_widgets()

    def create_widgets(self) -> None:
        """ Constructs GUI layout components. """
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Control Panel
        control_frame = ttk.LabelFrame(
            main_container, text=" Simulation Setup ", padding=10
        )
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Preset Selection
        ttk.Label(
            control_frame, text="Preset Benchmark Profiles", font=("Arial", 10, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))

        f_preset = ttk.Frame(control_frame)
        f_preset.pack(fill=tk.X, pady=2)

        self.cmb_preset = ttk.Combobox(
            f_preset, values=["Custom Setup"] + list(PRESETS.keys()),
            state="readonly", width=22
        )
        self.cmb_preset.current(0)
        self.cmb_preset.pack(fill=tk.X)
        self.cmb_preset.bind("<<ComboboxSelected>>", self.on_preset_selected)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # Domain & Mesh Parameters
        ttk.Label(
            control_frame, text="Domain & Mesh", font=("Arial", 10, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))

        f_domain = ttk.Frame(control_frame)
        f_domain.pack(fill=tk.X, pady=2)

        ttk.Label(f_domain, text="Length (l):").grid(row=0, column=0, sticky=tk.W)
        self.ent_l = ttk.Entry(f_domain, width=10)
        self.ent_l.insert(0, "1.0")
        self.ent_l.grid(row=0, column=1, padx=5)

        ttk.Label(f_domain, text="Cells (N):").grid(row=1, column=0, sticky=tk.W)
        self.ent_n = ttk.Entry(f_domain, width=10)
        self.ent_n.insert(0, "100")
        self.ent_n.grid(row=1, column=1, padx=5)

        ttk.Label(f_domain, text="End time (t):").grid(row=2, column=0, sticky=tk.W)
        self.ent_tcorte = ttk.Entry(f_domain, width=10)
        self.ent_tcorte.insert(0, "0.2")
        self.ent_tcorte.grid(row=2, column=1, padx=5)

        ttk.Label(f_domain, text="CFL:").grid(row=3, column=0, sticky=tk.W)
        self.ent_cfl = ttk.Entry(f_domain, width=10)
        self.ent_cfl.insert(0, "0.5")
        self.ent_cfl.grid(row=3, column=1, padx=5)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # Initial States
        ttk.Label(
            control_frame, text="Initial States", font=("Arial", 10, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))

        f_states = ttk.Frame(control_frame)
        f_states.pack(fill=tk.X, pady=2)

        ttk.Label(f_states, text="Left (L)").grid(row=0, column=1)
        ttk.Label(f_states, text="Right (R)").grid(row=0, column=2)

        ttk.Label(f_states, text="Density (rho):").grid(row=1, column=0, sticky=tk.W)
        self.ent_rho_l = ttk.Entry(f_states, width=12)
        self.ent_rho_l.insert(0, "1.0")
        self.ent_rho_l.grid(row=1, column=1, padx=2)

        self.ent_rho_r = ttk.Entry(f_states, width=12)
        self.ent_rho_r.insert(0, "0.125")
        self.ent_rho_r.grid(row=1, column=2, padx=2)

        ttk.Label(f_states, text="Velocity (u):").grid(row=2, column=0, sticky=tk.W)
        self.ent_u_l = ttk.Entry(f_states, width=12)
        self.ent_u_l.insert(0, "0.0")
        self.ent_u_l.grid(row=2, column=1, padx=2)

        self.ent_u_r = ttk.Entry(f_states, width=12)
        self.ent_u_r.insert(0, "0.0")
        self.ent_u_r.grid(row=2, column=2, padx=2)

        ttk.Label(f_states, text="Pressure (p):").grid(row=3, column=0, sticky=tk.W)
        self.ent_p_l = ttk.Entry(f_states, width=12)
        self.ent_p_l.insert(0, "1.0")
        self.ent_p_l.grid(row=3, column=1, padx=2)

        self.ent_p_r = ttk.Entry(f_states, width=12)
        self.ent_p_r.insert(0, "0.1")
        self.ent_p_r.grid(row=3, column=2, padx=2)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # Numerical & Physical Options
        ttk.Label(
            control_frame, text="Numerical & Physics Options", font=("Arial", 10, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))

        f_opts = ttk.Frame(control_frame)
        f_opts.pack(fill=tk.X, pady=2)

        ttk.Label(f_opts, text="Mode:").grid(row=0, column=0, sticky=tk.W)
        self.cmb_mode = ttk.Combobox(
            f_opts, values=["Non-Reactive (Sod)", "Reactive (2H2+O2)"],
            state="readonly", width=18
        )
        self.cmb_mode.current(0)
        self.cmb_mode.grid(row=0, column=1, pady=2)

        ttk.Label(f_opts, text="Kinetics:").grid(row=1, column=0, sticky=tk.W)
        self.cmb_kinetics = ttk.Combobox(
            f_opts, values=["Arrhenius (1-step)", "ZND (2-stage)"],
            state="readonly", width=18
        )
        self.cmb_kinetics.current(0)
        self.cmb_kinetics.grid(row=1, column=1, pady=2)

        ttk.Label(f_opts, text="Limiter:").grid(row=2, column=0, sticky=tk.W)
        self.cmb_limiter = ttk.Combobox(
            f_opts, values=["VanLeer", "Albada", "MINBEE", "Superbee", "Unos"],
            state="readonly", width=18
        )
        self.cmb_limiter.current(0)
        self.cmb_limiter.grid(row=2, column=1, pady=2)

        ttk.Label(f_opts, text="Boundary:").grid(row=3, column=0, sticky=tk.W)
        self.cmb_bc = ttk.Combobox(
            f_opts, values=["Open", "Reflective", "Periodic"], state="readonly", width=18
        )
        self.cmb_bc.current(0)
        self.cmb_bc.grid(row=3, column=1, pady=2)

        self.var_exact_overlay = tk.BooleanVar(value=True)
        chk_exact = ttk.Checkbutton(
            control_frame, text="Overlay Exact Toro Solution", variable=self.var_exact_overlay,
            command=lambda: self.render_frame(self.current_frame)
        )
        chk_exact.pack(anchor=tk.W, pady=3)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        btn_run = ttk.Button(
            control_frame, text="▶ Run Simulation", command=self.on_run
        )
        btn_run.pack(fill=tk.X, ipady=4)

        # Animation Controls Box
        anim_frame = ttk.LabelFrame(control_frame, text=" Animation Controls ", padding=5)
        anim_frame.pack(fill=tk.X, pady=6)

        f_anim_btns = ttk.Frame(anim_frame)
        f_anim_btns.pack(fill=tk.X, pady=2)

        self.btn_play = ttk.Button(
            f_anim_btns, text="▶ Play", width=8, command=self.toggle_animation, state=tk.DISABLED
        )
        self.btn_play.pack(side=tk.LEFT, padx=2)

        self.btn_reset_anim = ttk.Button(
            f_anim_btns, text="↺ Reset", width=8, command=self.reset_animation, state=tk.DISABLED
        )
        self.btn_reset_anim.pack(side=tk.LEFT, padx=2)

        self.slider_frame = ttk.Scale(
            anim_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_move
        )
        self.slider_frame.pack(fill=tk.X, pady=5)
        self.slider_frame.config(state=tk.DISABLED)

        # Live Conservation Audit Panel
        audit_frame = ttk.LabelFrame(control_frame, text=" Live Conservation Audit ", padding=5)
        audit_frame.pack(fill=tk.X, pady=6)

        self.lbl_audit_mass = ttk.Label(audit_frame, text="Mass Change (dM): N/A")
        self.lbl_audit_mass.pack(anchor=tk.W)

        self.lbl_audit_mom = ttk.Label(audit_frame, text="Final Momentum: N/A")
        self.lbl_audit_mom.pack(anchor=tk.W)

        self.lbl_audit_energy = ttk.Label(audit_frame, text="Final Energy: N/A")
        self.lbl_audit_energy.pack(anchor=tk.W)

        # Right Matplotlib Plot Panel
        plot_frame = ttk.Frame(main_container)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_p = self.fig.add_subplot(221)
        self.ax_rho = self.fig.add_subplot(222)
        self.ax_u = self.fig.add_subplot(223)
        self.ax_extra = self.fig.add_subplot(224)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(
            self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_preset_selected(self, event=None) -> None:
        """ Fills parameters automatically based on selected preset profile. """
        preset_key = self.cmb_preset.get()
        if preset_key == "Custom Setup":
            return

        cfg = get_preset(preset_key)

        self.ent_l.delete(0, tk.END)
        self.ent_l.insert(0, "1.0")

        self.ent_n.delete(0, tk.END)
        self.ent_n.insert(0, str(cfg.n_cells))

        self.ent_tcorte.delete(0, tk.END)
        self.ent_tcorte.insert(0, str(cfg.t_end))

        self.ent_cfl.delete(0, tk.END)
        self.ent_cfl.insert(0, str(cfg.cfl))

        self.ent_rho_l.delete(0, tk.END)
        self.ent_rho_l.insert(0, str(cfg.rho_l))

        self.ent_rho_r.delete(0, tk.END)
        self.ent_rho_r.insert(0, str(cfg.rho_r))

        self.ent_u_l.delete(0, tk.END)
        self.ent_u_l.insert(0, str(cfg.u_l))

        self.ent_u_r.delete(0, tk.END)
        self.ent_u_r.insert(0, str(cfg.u_r))

        self.ent_p_l.delete(0, tk.END)
        self.ent_p_l.insert(0, str(cfg.p_l))

        self.ent_p_r.delete(0, tk.END)
        self.ent_p_r.insert(0, str(cfg.p_r))

        mode_idx = 0 if cfg.mode == "sod" else 1
        self.cmb_mode.current(mode_idx)

        lim_idx = ["VanLeer", "Albada", "MINBEE", "Superbee", "Unos"].index(
            cfg.limiter if cfg.limiter in ["VanLeer", "Albada", "MINBEE", "Superbee", "Unos"] else "VanLeer"
        )
        self.cmb_limiter.current(lim_idx)
        self.status_var.set(f"Loaded preset profile: '{cfg.name}'")

    def on_run(self) -> None:
        """ Triggers simulation execution and updates Matplotlib figures. """
        try:
            self.stop_animation()

            l_val = float(self.ent_l.get())
            n_val = int(self.ent_n.get())
            tcorte_val = float(self.ent_tcorte.get())
            cfl_val = float(self.ent_cfl.get())
            validate_grid(l_val, n_val, cfl_val, tcorte_val)

            rho_l = float(self.ent_rho_l.get())
            rho_r = float(self.ent_rho_r.get())
            u_l = float(self.ent_u_l.get())
            u_r = float(self.ent_u_r.get())
            p_l = float(self.ent_p_l.get())
            p_r = float(self.ent_p_r.get())

            exact_kwargs = dict(x0=max(1,min(n_val-1,int(np.round(0.5*n_val)))) * l_val/n_val, rho_l=rho_l,u_l=u_l,p_l=p_l,rho_r=rho_r,u_r=u_r,p_r=p_r)
            mode = self.cmb_mode.get()
            limiter = self.cmb_limiter.get()
            bc = self.cmb_bc.get()
            kinetics_choice = "znd" if "ZND" in self.cmb_kinetics.get() else "arrhenius"

            self.status_var.set("Running simulation...")
            self.update_idletasks()

            dx = l_val / float(n_val)
            x_coords = np.linspace(dx / 2.0, l_val - dx / 2.0, n_val)

            if mode == "Non-Reactive (Sod)":
                gamma = 1.4
                u_evo, t_arr = run_simulation(
                    l=l_val, n=n_val,
                    rho_l=rho_l, u_l=u_l, p_l=p_l,
                    rho_r=rho_r, u_r=u_r, p_r=p_r,
                    gamma=gamma, c_cfl=cfl_val, t_corte=tcorte_val,
                    limiter=limiter, reflective_or_open=bc
                )
                lambda_evo = None

            else:
                lambda_init = np.ones(n_val)
                u_evo, t_arr, lambda_evo = run_simulation_wQR(
                    l=l_val, n=n_val,
                    rho_l=rho_l, u_l=u_l, p_l=p_l,
                    rho_r=rho_r, u_r=u_r, p_r=p_r,
                    lambda_init=lambda_init, c_cfl=cfl_val, t_corte=tcorte_val,
                    limiter=limiter, reflective_or_open=bc,
                    kinetic_model=kinetics_choice
                )

            self.u_evo, self.t_arr, self.lambda_evo = u_evo, t_arr, lambda_evo
            self.x_coords, self.exact_kwargs = x_coords, exact_kwargs
            total_frames = len(self.t_arr) - 1
            self.slider_frame.config(state=tk.NORMAL, from_=0, to=total_frames)
            self.btn_play.config(state=tk.NORMAL if total_frames else tk.DISABLED)
            self.btn_reset_anim.config(state=tk.NORMAL)

            # Update live conservation audit sidebar
            audit = compute_conservation_audit(
                self.u_evo[:, :, 0], self.u_evo[:, :, -1], dx, 0.0, len(self.t_arr)-1
            )
            self.lbl_audit_mass.config(
                text=f"Mass Change (dM): {audit['mass_change_rel']:.2e}"
            )
            self.lbl_audit_mom.config(
                text=f"Final Momentum: {audit['final_momentum']:.2f}"
            )
            self.lbl_audit_energy.config(
                text=f"Final Energy: {audit['final_energy_j']:.2e} J"
            )

            self.slider_frame.set(total_frames)
            self.render_frame(total_frames)

            self.status_var.set(
                f"Simulation completed successfully ({len(self.t_arr)-1} timesteps, "
                f"t_final={self.t_arr[-1]:.6g} s)."
            )

        except Exception as err:
            messagebox.showerror("Simulation Error", str(err))
            self.status_var.set("Error occurred during simulation.")

    def render_frame(self, frame_idx: int) -> None:
        """ Renders a specific temporal frame of the simulation. """
        if self.u_evo is None or self.t_arr is None:
            return

        frame_idx = max(0, min(len(self.t_arr) - 1, frame_idx))
        self.current_frame = frame_idx
        t_curr = self.t_arr[frame_idx]

        u_consv_frame = self.u_evo[:, :, frame_idx]

        if self.lambda_evo is not None:
            lambda_frame = self.lambda_evo[frame_idx, :]
            gamma_frame = gamma_calc(lambda_frame)
            u_prim_frame = uconsv_to_uprim(u_consv_frame, gamma_frame)
            extra_data = lambda_frame
        else:
            gamma_frame = 1.4
            u_prim_frame = uconsv_to_uprim(u_consv_frame, gamma_frame)
            extra_data = None

        rho = u_prim_frame[0, :]
        u_vel = u_prim_frame[1, :]
        p_pres = u_prim_frame[2, :]

        self.ax_p.clear()
        self.ax_rho.clear()
        self.ax_u.clear()
        self.ax_extra.clear()

        x_min, x_max = min(self.x_coords), max(self.x_coords)

        # Plot exact Toro Riemann solution overlay for Sod if enabled
        if self.lambda_evo is None and self.var_exact_overlay.get() and t_curr > 0:
            x_dense = np.linspace(x_min, x_max, 500)
            rho_ex, u_ex, p_ex = exact_riemann_sod(x_dense, t=t_curr, **self.exact_kwargs)
            self.ax_p.plot(x_dense, p_ex, 'k--', linewidth=1.2, label='Exact (Toro)')
            self.ax_rho.plot(x_dense, rho_ex, 'k--', linewidth=1.2, label='Exact (Toro)')
            self.ax_u.plot(x_dense, u_ex, 'k--', linewidth=1.2, label='Exact (Toro)')

        self.ax_p.plot(self.x_coords, p_pres, 'r-o', markersize=3)
        self.ax_p.set_title(f"Pressure p [Pa] (t = {t_curr:.6g} s)")
        self.ax_p.set_xlabel("Position x [m]")
        self.ax_p.set_xlim(x_min, x_max)
        self.ax_p.grid(True)

        self.ax_rho.plot(self.x_coords, rho, 'b-o', markersize=3)
        self.ax_rho.set_title(f"Density rho [kg/m³] (t = {t_curr:.6g} s)")
        self.ax_rho.set_xlabel("Position x [m]")
        self.ax_rho.set_xlim(x_min, x_max)
        self.ax_rho.grid(True)

        self.ax_u.plot(self.x_coords, u_vel, 'g-o', markersize=3)
        self.ax_u.set_title(f"Velocity u [m/s] (t = {t_curr:.6g} s)")
        self.ax_u.set_xlabel("Position x [m]")
        self.ax_u.set_xlim(x_min, x_max)
        self.ax_u.grid(True)

        if extra_data is not None:
            self.ax_extra.plot(self.x_coords, extra_data, 'm-o', markersize=3)
            self.ax_extra.set_title(f"Reactant Fraction lambda [-] (t = {t_curr:.6g} s)")
            self.ax_extra.set_xlabel("Position x [m]")
            self.ax_extra.set_xlim(x_min, x_max)
            self.ax_extra.grid(True)
        else:
            self.ax_extra.set_title("Inert Sod Shock Flow")
            self.ax_extra.grid(True)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def stop_animation(self) -> None:
        """Cancel the pending frame so restarting cannot create concurrent loops."""
        self.is_animating = False
        if self.animation_after_id is not None:
            self.after_cancel(self.animation_after_id)
            self.animation_after_id = None
        self.btn_play.config(text="▶ Play")

    def toggle_animation(self) -> None:
        """ Starts or pauses real-time animation playback. """
        if self.u_evo is None or len(self.t_arr) < 2:
            return

        if self.is_animating:
            self.stop_animation()
        else:
            self.is_animating = True
            self.btn_play.config(text="⏸ Pause")
            self.animate_loop()

    def animate_loop(self) -> None:
        """ Loop for stepping through animation frames smoothly. """
        self.animation_after_id = None
        if not self.is_animating or self.t_arr is None:
            return

        if self.current_frame >= len(self.t_arr) - 1:
            self.current_frame = 0

        self.current_frame += 1
        self.slider_frame.set(self.current_frame)
        self.render_frame(self.current_frame)

        if self.current_frame < len(self.t_arr) - 1 and self.is_animating:
            self.animation_after_id = self.after(50, self.animate_loop)
        else:
            self.is_animating = False
            self.btn_play.config(text="▶ Play")

    def reset_animation(self) -> None:
        """ Resets playback to the initial time step frame. """
        self.stop_animation()
        self.current_frame = 0
        self.slider_frame.set(0)
        self.render_frame(0)

    def on_slider_move(self, val: str) -> None:
        """ Event handler for manual frame slider movement. """
        frame_idx = int(float(val))
        if frame_idx != self.current_frame:
            self.stop_animation()
            self.render_frame(frame_idx)


if __name__ == "__main__":
    app = CFDApp()
    app.mainloop()
