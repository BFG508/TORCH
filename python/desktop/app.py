"""Tk desktop for 2D/3D simulations and disk playback."""

from pathlib import Path
from queue import Empty
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

from cfdSolver import Simulation
from cfdSolver.torchGeometry import coordinates
from cfdSolver.torchSolver3D import LIMITERS

from .controller import SessionRunner
from .lifecycle import owned_resources, release
from .export import export_session
from .model import (
    config_spec,
    create_session,
    defaults,
    face_defaults,
    face_keys,
    history_frames,
    new_prefix,
    presets,
)
from .view import draw_session, fields
from .adaptive import slice_count
from cfdSolver.torchAdaptiveGeometry import settings

BOUNDARY_NAMES = {
    "Outflow": "outflow",
    "Slip wall": "wall",
    "Periodic": "periodic",
    "Prescribed inflow": "inflow",
    "Pressure outlet": "pressure_outlet",
    "Reservoir inlet": "stagnation_inlet",
}


class SimulationApp(tk.Tk):
    """Keep UI work on Tk and mutable state on one worker."""

    def __init__(self, dimension=2):
        super().__init__()
        self.title("TORCH | 2D / 3D Simulation Desktop")
        self.geometry("1420x920")
        self.minsize(1100, 760)
        self.runner = SessionRunner()
        self.session = None
        self.displayed = None
        self.prefix = None
        self.frames = []
        self.face_edits = {}
        self.last_plane = None
        self.last_frame = None
        self.playing = False
        self.play_job = None
        self.poll_job = None
        self.closing = False
        self.closed = False
        self.adjusting = False
        self.config_controls = []
        self.vars = {}
        self.entries = {}
        self.labels = {}
        self.face_key_map = {}
        self.mesh = []
        self.boundary_vars = {}
        self.status = tk.StringVar(value="Configure a simulation, then Run.")
        self.audit = tk.StringVar(value="No results yet")
        self.view_time = tk.StringVar(value="No recorded frames")
        self._widgets()
        self._apply_spec(defaults(dimension))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_job = self.after(40, self._poll)

    def _entry(
        self,
        frame,
        row,
        key,
        label,
        value="",
        choices=None,
        configuration=True,
    ):
        text = ttk.Label(frame, text=label)
        text.grid(row=row, column=0, sticky="w", padx=7, pady=4)
        self.labels[key] = text
        variable = tk.StringVar(value=str(value))
        if choices is None:
            widget = ttk.Entry(frame, textvariable=variable, width=23)
        else:
            widget = ttk.Combobox(
                frame,
                textvariable=variable,
                values=choices,
                state="readonly",
                width=21,
            )
        widget.grid(row=row, column=1, sticky="ew", padx=7, pady=4)
        self.vars[key], self.entries[key] = variable, widget
        if configuration:
            self.config_controls.append(widget)
        return widget

    def _widgets(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        left = ttk.Frame(self, padding=8, width=410)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.tabs = ttk.Notebook(left)
        self.tabs.grid(row=0, column=0, sticky="nsew")
        setup, mesh, boundary, files = [ttk.Frame(self.tabs) for _ in range(4)]
        for name, tab in zip(
            ("Setup", "Mesh", "Boundaries", "Files"),
            (setup, mesh, boundary, files),
        ):
            self.tabs.add(tab, text=name)
            tab.columnconfigure(1, weight=1)
        self._entry(setup, 0, "dimension", "Dimensions", "2D", ("2D", "3D"))
        self._entry(
            setup,
            1,
            "geometry",
            "Geometry",
            "cartesian",
            ("cartesian", "cylindrical", "annular", "spherical"),
        )
        self._entry(setup, 2, "preset", "Initial condition", "", ())
        self.entries["dimension"].bind("<<ComboboxSelected>>", self._domain)
        self.entries["geometry"].bind("<<ComboboxSelected>>", self._domain)
        self.entries["preset"].bind("<<ComboboxSelected>>", self._preset)
        self._entry(
            setup,
            3,
            "t_end",
            "End time [s]",
            0.05,
            configuration=False,
        )
        self._entry(setup, 4, "cfl", "CFL", 0.4)
        self._entry(setup, 5, "gamma", "Inert gamma", 1.4)
        self._entry(setup, 6, "limiter", "Limiter", "VanLeer", LIMITERS)
        self.reactive = tk.BooleanVar(value=False)
        check = ttk.Checkbutton(
            setup,
            text="Reactive flow",
            variable=self.reactive,
            command=self._states,
        )
        check.grid(row=7, column=0, columnspan=2, sticky="w", padx=7, pady=5)
        self.config_controls.append(check)
        self._entry(
            setup,
            8,
            "kinetic_model",
            "Chemistry",
            "arrhenius",
            ("arrhenius", "znd"),
        )
        self.entries["kinetic_model"].bind(
            "<<ComboboxSelected>>", lambda event: self._states()
        )
        for row, key, label, value in (
            (9, "fraction", "Initial unburned fraction", 1.0),
            (10, "beta", "Initial induction fraction", 1.0),
            (12, "density", "Density [kg/m³]", 1.0),
            (13, "u", "Local velocity 1 [m/s]", 0.0),
            (14, "v", "Local velocity 2 [m/s]", 0.0),
            (15, "w", "Local velocity 3 [m/s]", 0.0),
            (16, "pressure", "Static pressure [Pa]", 101325.0),
            (17, "rotation", "Rotation rate [rad/s]", 20.0),
        ):
            self._entry(setup, row, key, label, value)
        ttk.Label(
            setup,
            text="Uniform / geometric initial data",
            font=("", 10, "bold"),
        ).grid(row=11, column=0, columnspan=2, pady=(12, 5))
        ttk.Label(
            mesh,
            text="Cells, extent and lower coordinate on each axis.",
            wraplength=350,
        ).grid(row=0, column=0, columnspan=2, padx=7, pady=12)
        for d, axis in enumerate("xyz"):
            self._entry(mesh, 1 + 4 * d, "n" + axis, "Cells " + axis, 20)
            self._entry(mesh, 2 + 4 * d, "l" + axis, "Extent " + axis, 1.0)
            self._entry(mesh, 3 + 4 * d, "o" + axis, "Origin " + axis, 0.0)
        ttk.Label(
            mesh,
            text=(
                "Lengths: m. Angles: rad. Cylindrical 2D uses r,z; "
                "annular uses r,theta; spherical uses r,theta,phi. "
                "3D curved domains require positive inner radius."
            ),
            wraplength=350,
        ).grid(row=14, column=0, columnspan=2, padx=7, pady=15)
        execution = ttk.Frame(self.tabs)
        self.tabs.add(execution, text="Execution")
        execution.columnconfigure(1, weight=1)
        self._entry(
            execution, 0, "workers", "CPU workers", 1, configuration=False
        )
        self.adaptive = tk.BooleanVar(value=False)
        check = ttk.Checkbutton(
            execution,
            text="Adaptive mesh (AMR)",
            variable=self.adaptive,
            command=self._states,
        )
        check.grid(row=1, column=0, columnspan=2, sticky="w", padx=7, pady=5)
        self.config_controls.append(check)
        for row, (key, value) in enumerate(settings().items(), 2):
            self._entry(
                execution,
                row,
                "amr_" + key,
                key.replace("_", " ").capitalize(),
                value,
            )
        ttk.Label(
            execution,
            text=(
                "AMR uses limited linear reconstruction (order 2) or "
                "constant states (order 1). The base limiter applies to "
                "uniform meshes. A shared global time step preserves "
                "coarse/fine flux balances."
            ),
            wraplength=350,
        ).grid(row=10, column=0, columnspan=2, padx=7, pady=12)
        self._boundary_widgets(boundary)
        self._entry(
            files,
            0,
            "folder",
            "Results folder",
            "out/desktop",
            configuration=False,
        )
        ttk.Button(
            files, text="Choose folder", command=self.choose_folder
        ).grid(
            row=1,
            column=0,
            columnspan=2,
            pady=5,
        )
        self._entry(
            files,
            2,
            "interval",
            "Record every N steps",
            5,
            configuration=False,
        )
        for row, name, command in (
            (3, "Load restart", self.load_dialog),
            (4, "Open recorded series", self.series_dialog),
            (5, "Save current restart", self.save_dialog),
            (6, "Export displayed data", self.export_dialog),
            (7, "Export displayed figure", self.figure_dialog),
        ):
            button = ttk.Button(files, text=name, command=command)
            button.grid(
                row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=5
            )
            setattr(self, "file_button_" + str(row), button)
        ttk.Label(
            files,
            text=(
                "Recorded frames stay on disk. Save current restart uses "
                "the latest simulation state; displayed exports may use "
                "an earlier frame. Resume always continues the latest state."
            ),
            wraplength=350,
        ).grid(row=8, column=0, columnspan=2, padx=8, pady=15)
        actions = ttk.Frame(left)
        actions.grid(row=1, column=0, sticky="ew", pady=10)
        actions.columnconfigure((0, 1, 2), weight=1)
        self.run_button = ttk.Button(
            actions, text="Run / resume", command=self.run
        )
        self.run_button.grid(row=0, column=0, sticky="ew", padx=2)
        self.pause_button = ttk.Button(
            actions,
            text="Pause",
            command=self.pause,
            state="disabled",
        )
        self.pause_button.grid(row=0, column=1, sticky="ew", padx=2)
        self.new_button = ttk.Button(actions, text="New", command=self.new)
        self.new_button.grid(row=0, column=2, sticky="ew", padx=2)
        right = ttk.Frame(self, padding=(0, 8, 8, 8))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        controls = ttk.Frame(right)
        controls.grid(row=0, column=0, sticky="ew")
        self.field_var = tk.StringVar(value="Pressure")
        self.field_combo = ttk.Combobox(
            controls,
            textvariable=self.field_var,
            values=("Pressure",),
            state="readonly",
            width=19,
        )
        self.field_combo.pack(side="left", padx=4)
        self.field_combo.bind("<<ComboboxSelected>>", self.redraw)
        self.mode_var = tk.StringVar(value="Map")
        self.mode_combo = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=("Map", "3D plane"),
            state="readonly",
            width=12,
        )
        self.mode_combo.pack(side="left", padx=4)
        self.mode_combo.bind("<<ComboboxSelected>>", self.redraw)
        self.axis_var = tk.StringVar(value="z")
        self.axis_combo = ttk.Combobox(
            controls,
            textvariable=self.axis_var,
            values=tuple("xyz"),
            state="readonly",
            width=8,
        )
        self.axis_combo.pack(side="left", padx=4)
        self.axis_combo.bind("<<ComboboxSelected>>", self._slice_axis)
        self.slice_var = tk.IntVar(value=0)
        self.slice_scale = tk.Scale(
            controls,
            from_=0,
            to=19,
            orient="horizontal",
            length=180,
            variable=self.slice_var,
            command=self._slice_index,
            label="Slice index (0 based)",
        )
        self.slice_scale.pack(side="left", fill="x", expand=True)
        self.figure = Figure(figsize=(8.5, 6.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        toolbar = ttk.Frame(right)
        toolbar.grid(row=2, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar)
        timeline = ttk.Frame(right)
        timeline.grid(row=3, column=0, sticky="ew")
        timeline.columnconfigure(2, weight=1)
        self.play_button = ttk.Button(timeline, text="Play", command=self.play)
        self.play_button.grid(row=0, column=0, padx=4)
        self.latest_button = ttk.Button(
            timeline,
            text="Latest",
            command=self.latest,
        )
        self.latest_button.grid(row=0, column=1, padx=4)
        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = tk.Scale(
            timeline,
            from_=0,
            to=0,
            orient="horizontal",
            showvalue=False,
            variable=self.frame_var,
            command=self._history_index,
        )
        self.frame_scale.grid(row=0, column=2, sticky="ew")
        ttk.Label(right, textvariable=self.view_time).grid(
            row=4,
            column=0,
            sticky="w",
            pady=4,
        )
        ttk.Label(self, textvariable=self.audit, padding=(12, 4)).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
        )
        ttk.Label(self, textvariable=self.status, padding=(12, 4)).grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
        )
        self._states()

    def _boundary_widgets(self, frame):
        self.face_var = tk.StringVar(value="x_min")
        self.face_combo = ttk.Combobox(
            frame,
            textvariable=self.face_var,
            values=face_keys(2),
            state="readonly",
            width=20,
        )
        self.face_combo.grid(row=0, column=0, columnspan=2, padx=8, pady=10)
        self.face_combo.bind("<<ComboboxSelected>>", self._show_face)
        self.config_controls.append(self.face_combo)
        self._entry(
            frame,
            1,
            "face_type",
            "Condition",
            "Outflow",
            tuple(BOUNDARY_NAMES),
        )
        self.entries["face_type"].bind("<<ComboboxSelected>>", self._face_type)
        for row, key, label, value in (
            (2, "face_density", "Inlet / backflow density", 1.0),
            (3, "face_u", "Local velocity 1 [m/s]", 10.0),
            (4, "face_v", "Local velocity 2 [m/s]", 0.0),
            (5, "face_w", "Local velocity 3 [m/s]", 0.0),
            (6, "face_pressure", "Static pressure [Pa]", 101325.0),
            (7, "face_p0", "Total pressure [Pa]", 110000.0),
            (8, "face_t0", "Total temperature [K]", 300.0),
            (9, "face_r", "Gas constant [J/(kg K)]", 287.0),
            (10, "face_lambda", "Incoming unburned fraction", 1.0),
            (11, "face_beta", "Incoming induction fraction", 1.0),
        ):
            self._entry(frame, row, key, label, value)
        self.backflow = tk.BooleanVar(value=False)
        self.backflow_check = ttk.Checkbutton(
            frame,
            text="Specify pressure-outlet backflow",
            variable=self.backflow,
            command=self._face_type,
        )
        self.backflow_check.grid(row=12, column=0, columnspan=2, pady=7)
        self.config_controls.append(self.backflow_check)
        self.apply_face_button = ttk.Button(
            frame,
            text="Apply face condition",
            command=self.apply_face,
        )
        self.apply_face_button.grid(row=13, column=0, columnspan=2, pady=7)
        self.config_controls.append(self.apply_face_button)
        self.face_note = tk.StringVar()
        ttk.Label(frame, textvariable=self.face_note, wraplength=350).grid(
            row=14,
            column=0,
            columnspan=2,
            padx=8,
            pady=10,
        )

    def _apply_spec(self, spec):
        self.adjusting = True
        self.vars["dimension"].set(str(spec["dimension"]) + "D")
        for key in (
            "geometry",
            "preset",
            "t_end",
            "cfl",
            "gamma",
            "limiter",
            "kinetic_model",
            "density",
            "pressure",
            "rotation",
            "fraction",
            "beta",
        ):
            self.vars[key].set(str(spec[key]))
        self.reactive.set(spec["reactive"])
        self.adaptive.set(spec.get("adaptive", False))
        self.vars["workers"].set(str(spec.get("workers", 1)))
        for key, value in settings(**spec.get("amr", {})).items():
            self.vars["amr_" + key].set(str(value))
        for d, axis in enumerate("xyz"):
            for key, values, fallback in (
                ("n", spec["counts"], 2),
                ("l", spec["lengths"], 1.0),
                ("o", spec["origins"], 0.0),
            ):
                self.vars[key + axis].set(
                    str(values[d] if d < len(values) else fallback)
                )
            self.vars["uvw"[d]].set(
                str(
                    spec["velocities"][d]
                    if d < len(spec["velocities"])
                    else 0.0
                )
            )
        self.face_edits = dict(spec.get("face_edits", {}))
        names = coordinates(
            dict(
                dimension=spec["dimension"],
                geometry=spec["geometry"],
            )
        )
        for d, name in enumerate(names):
            axis = "xyz"[d]
            unit = "rad" if name in ("theta", "phi") else "m"
            self.labels["n" + axis].configure(text="Cells " + name)
            self.labels["l" + axis].configure(
                text="Extent " + name + " [" + unit + "]"
            )
            self.labels["o" + axis].configure(
                text="Origin " + name + " [" + unit + "]"
            )
            for key in ("uvw"[d], "face_" + "uvw"[d]):
                self.labels[key].configure(text="Velocity " + name + " [m/s]")
        self.face_key_map = {
            name + " " + side: key
            for name, pair in zip(
                names,
                (
                    face_keys(spec["dimension"])[i: i + 2]
                    for i in range(0, 2 * spec["dimension"], 2)
                ),
            )
            for side, key in zip(("minimum", "maximum"), pair)
        }
        self.entries["preset"].configure(
            values=presets(max(2, spec["dimension"]), spec["geometry"]),
        )
        self.face_combo.configure(values=list(self.face_key_map))
        self.face_var.set(next(iter(self.face_key_map)))
        self.adjusting = False
        self._states()
        self._show_face()

    def _domain(self, event=None):
        if self.adjusting:
            return
        dim = int(self.vars["dimension"].get()[0])
        self._apply_spec(defaults(dim, self.vars["geometry"].get()))

    def _preset(self, event=None):
        if self.adjusting:
            return
        name = self.vars["preset"].get()
        self.face_edits = {}
        if name in ("2d_rde_chamber", "3d_ignition"):
            self.reactive.set(True)
            self.vars["t_end"].set("1e-5")
        else:
            self.reactive.set(False)
            self.vars["t_end"].set(
                ".05" if self.vars["geometry"].get() == "cartesian" else "1e-4"
            )
        self._states()
        self._show_face()

    def specification(self):
        dim = int(self.vars["dimension"].get()[0])
        result = defaults(max(2, dim), self.vars["geometry"].get())
        result["dimension"] = dim
        for key in ("geometry", "preset", "limiter", "kinetic_model"):
            result[key] = self.vars[key].get()
        for key in (
            "t_end",
            "cfl",
            "gamma",
            "density",
            "pressure",
            "rotation",
            "fraction",
            "beta",
        ):
            result[key] = float(self.vars[key].get())
        result["counts"] = [int(self.vars["n" + a].get()) for a in "xyz"[:dim]]
        result["lengths"] = [
            float(self.vars["l" + a].get()) for a in "xyz"[:dim]
        ]
        result["origins"] = [
            float(self.vars["o" + a].get()) for a in "xyz"[:dim]
        ]
        result["velocities"] = [float(self.vars[a].get()) for a in "uvw"[:dim]]
        result["reactive"] = self.reactive.get()
        result["workers"] = int(self.vars["workers"].get())
        result["adaptive"] = self.adaptive.get()
        result["amr"] = {
            key: (float if "threshold" in key else int)(
                self.vars["amr_" + key].get()
            )
            for key in settings()
        }
        result["face_edits"] = self.face_edits.copy()
        return result

    def face_key(self):
        value = self.face_var.get()
        return self.face_key_map.get(value, value)

    def _show_face(self, event=None):
        try:
            spec = self.specification()
            key = self.face_key()
            face = face_defaults(spec)[face_keys(spec["dimension"]).index(key)]
            self.vars["face_type"].set(
                next(
                    label
                    for label, value in BOUNDARY_NAMES.items()
                    if value == face["type"]
                )
            )
            data = face.get("primitive", face.get("backflow"))
            if data is None:
                d = face_keys(spec["dimension"]).index(key) // 2
                velocity = [0.0] * spec["dimension"]
                velocity[d] = 10.0 if key.endswith("min") else -10.0
                data = [1.0] + velocity + [101325.0]
            for key, value in zip(
                ["face_density"]
                + ["face_" + a for a in "uvw"[: spec["dimension"]]]
                + ["face_pressure"],
                data,
            ):
                self.vars[key].set(str(value))
            for key, source, fallback in (
                ("face_p0", "total_pressure", 110000.0),
                ("face_t0", "total_temperature", 300.0),
                ("face_r", "gas_constant", 287.0),
                ("face_lambda", "lambda", 1.0),
                ("face_beta", "beta", 1.0),
            ):
                self.vars[key].set(str(face.get(source, fallback)))
            if face["type"] == "pressure_outlet":
                self.vars["face_pressure"].set(str(face["pressure"]))
            self.backflow.set("backflow" in face)
            self._face_type()
        except (ValueError, KeyError):
            self.face_note.set(
                "Complete the mesh inputs before editing faces."
            )

    def _face_type(self, event=None):
        kind = BOUNDARY_NAMES[self.vars["face_type"].get()]
        primitive = kind == "inflow" or (
            kind == "pressure_outlet" and self.backflow.get()
        )
        active = set()
        if primitive:
            active.update(["face_density", "face_u", "face_v", "face_w"])
        if primitive or kind == "pressure_outlet":
            active.add("face_pressure")
        if kind == "stagnation_inlet":
            active.update(["face_p0", "face_t0", "face_r"])
        if kind in ("inflow", "stagnation_inlet", "pressure_outlet"):
            if self.reactive.get():
                active.add("face_lambda")
                if self.vars["kinetic_model"].get() == "znd":
                    active.add("face_beta")
        if int(self.vars["dimension"].get()[0]) != 3:
            active.discard("face_w")
        locked = self.session is not None or self.runner.running
        for key in (k for k in self.entries if k.startswith("face_")):
            if key != "face_type":
                self.entries[key].configure(
                    state=(
                        "normal"
                        if key in active and not locked
                        else "disabled"
                    ),
                )
        self.backflow_check.configure(
            state=(
                "normal"
                if kind == "pressure_outlet" and not locked
                else "disabled"
            ),
        )
        self.face_note.set(
            "Periodic applies to both faces of this axis. Local velocities "
            "follow the geometry's coordinate order. Apply before Run."
        )

    def apply_face(self):
        try:
            spec = self.specification()
            dim = spec["dimension"]
            key = self.face_key()
            kind = BOUNDARY_NAMES[self.vars["face_type"].get()]
            face = dict(type=kind)
            if kind == "inflow" or (
                kind == "pressure_outlet" and self.backflow.get()
            ):
                p = [float(self.vars["face_density"].get())]
                p += [float(self.vars["face_" + a].get()) for a in "uvw"[:dim]]
                p += [float(self.vars["face_pressure"].get())]
                face["primitive" if kind == "inflow" else "backflow"] = p
            if kind == "pressure_outlet":
                face["pressure"] = float(self.vars["face_pressure"].get())
            if kind == "stagnation_inlet":
                for target, source in (
                    ("total_pressure", "face_p0"),
                    ("total_temperature", "face_t0"),
                    ("gas_constant", "face_r"),
                ):
                    face[target] = float(self.vars[source].get())
            if kind in ("inflow", "pressure_outlet", "stagnation_inlet"):
                face["lambda"] = float(self.vars["face_lambda"].get())
                face["beta"] = float(self.vars["face_beta"].get())
            edits = self.face_edits.copy()
            edits[key] = face
            opposite = key[:2] + ("max" if key.endswith("min") else "min")
            if kind == "periodic":
                edits[opposite] = dict(type="periodic")
            elif (
                face_defaults(spec)[face_keys(dim).index(opposite)]["type"]
                == "periodic"
            ):
                edits[opposite] = dict(type="outflow")
            spec["face_edits"] = edits
            create_session(spec)
            self.face_edits = edits
            self.status.set(
                "Applied " + key + ": " + self.vars["face_type"].get()
            )
        except (ValueError, TypeError, KeyError) as exc:
            self._error(exc)

    def _states(self):
        busy = self.runner.running
        locked = busy or self.session is not None
        for widget in self.config_controls:
            widget.configure(
                state=(
                    "disabled"
                    if locked
                    else (
                        "readonly"
                        if isinstance(widget, ttk.Combobox)
                        else "normal"
                    )
                )
            )
        dim = (
            int(self.vars["dimension"].get()[0])
            if "dimension" in self.vars
            else 2
        )
        if not locked:
            for key in ("nz", "lz", "oz", "w"):
                self.entries[key].configure(
                    state="normal" if dim == 3 else "disabled"
                )
            uniform = self.vars["preset"].get() in (
                "Uniform",
                "Radial blast",
                "Rotating equilibrium",
            )
            for key in ("density", "pressure", "u", "v", "w", "rotation"):
                self.entries[key].configure(
                    state="normal" if uniform else "disabled"
                )
            rotation = self.vars["preset"].get() == "Rotating equilibrium"
            self.entries["rotation"].configure(
                state="normal" if rotation else "disabled"
            )
            if rotation:
                for key in "uvw":
                    self.entries[key].configure(state="disabled")
            if dim != 3:
                self.entries["w"].configure(state="disabled")
            reactive = self.reactive.get()
            self.entries["gamma"].configure(
                state="disabled" if reactive else "normal"
            )
            self.entries["fraction"].configure(
                state="normal" if reactive else "disabled"
            )
            self.entries["kinetic_model"].configure(
                state="readonly" if reactive else "disabled"
            )
            self.entries["beta"].configure(
                state=(
                    "normal"
                    if (reactive and self.vars["kinetic_model"].get() == "znd")
                    else "disabled"
                )
            )
            self._face_type()
        if not locked:
            for key in settings():
                self.entries["amr_" + key].configure(
                    state="normal" if self.adaptive.get() else "disabled"
                )
            self.entries["limiter"].configure(
                state="disabled" if self.adaptive.get() else "readonly"
            )
        self.entries["workers"].configure(
            state="disabled" if busy else "normal"
        )
        self.entries["t_end"].configure(state="disabled" if busy else "normal")
        for key in ("folder", "interval"):
            self.entries[key].configure(state="disabled" if busy else "normal")
        self.run_button.configure(state="disabled" if busy else "normal")
        self.pause_button.configure(state="normal" if busy else "disabled")
        self.new_button.configure(state="disabled" if busy else "normal")
        for row in (3, 4):
            getattr(self, "file_button_" + str(row)).configure(
                state="disabled" if busy else "normal",
            )
        self.file_button_5.configure(
            state=(
                "normal"
                if self.session is not None and not busy
                else "disabled"
            ),
        )
        for row in (6, 7):
            getattr(self, "file_button_" + str(row)).configure(
                state=(
                    "normal"
                    if self.displayed is not None and not busy
                    else "disabled"
                ),
            )
        available = bool(self.frames) and not busy
        self.frame_scale.configure(state="normal" if available else "disabled")
        self.play_button.configure(state="normal" if available else "disabled")
        self.latest_button.configure(
            state="normal" if available else "disabled"
        )

    def run(self):
        if self.runner.running:
            return
        self.stop_playback()
        try:
            interval = int(self.vars["interval"].get())
            if interval < 1:
                raise ValueError("Record every N steps must be positive")
            if self.session is None:
                candidate = create_session(self.specification())
                prefix = new_prefix(self.vars["folder"].get())
            else:
                candidate = self.session
                target = float(self.vars["t_end"].get())
                if not np.isfinite(target) or target < candidate.time:
                    raise ValueError(
                        "End time cannot precede the current state"
                    )
                if target == candidate.time:
                    raise ValueError("Set a later end time, or choose New")
                candidate.config["t_end"] = target
                prefix = self.prefix or new_prefix(self.vars["folder"].get())
            candidate.set_workers(int(self.vars["workers"].get()))
            self.session, self.prefix = candidate, prefix
            self.runner.start(candidate, prefix, interval)
            self.status.set("Running; Pause stops after a complete step.")
            self._states()
        except (ValueError, OSError, TypeError, KeyError) as exc:
            self._error(exc)
            self._states()

    def pause(self):
        self.runner.request_pause()
        self.status.set("Pausing after the current step...")

    def _poll(self):
        self.poll_job = None
        if self.closing:
            if self.runner.running:
                self.poll_job = self.after(40, self._poll)
            else:
                self.destroy()
            return
        try:
            progress = self.runner.progress.get_nowait()
            action = "Pausing" if self.runner.pause.is_set() else "Running"
            self.status.set(
                f"{action}: t={progress.time:.6g} s, step {progress.steps}"
            )
            if progress.frame != self.last_frame and progress.frame.exists():
                self.last_frame = progress.frame
                self.show_session(Simulation.load(progress.frame))
        except Empty:
            pass
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.pause()
            self._error(exc)
        try:
            result = self.runner.results.get_nowait()
            self.session = result.session
            self._refresh_history()
            self.show_session(self.session)
            if result.error:
                self._error(result.error)
            else:
                self.status.set(
                    "Completed. Results: " + str(self.prefix.parent)
                    if self.session.completed
                    else "Paused; ready to resume."
                )
            self._states()
        except Empty:
            pass
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self._error(exc)
            self._states()
        self.poll_job = self.after(40, self._poll)

    def _refresh_history(self):
        if self.prefix is None:
            return
        data = history_frames(self.prefix)
        self.frames = data["frames"]
        self.adjusting = True
        self.frame_scale.configure(to=max(0, len(self.frames) - 1))
        self.frame_var.set(len(self.frames) - 1)
        self.adjusting = False

    def show_session(self, session):
        center = (
            self.displayed is None
            or self.displayed.config["dimension"]
            != session.config["dimension"]
        )
        self.displayed = session
        names = fields(session)
        self.field_combo.configure(values=names)
        if self.field_var.get() not in names:
            self.field_var.set("Pressure")
        axes = coordinates(session.config)
        self.axis_combo.configure(
            values=axes,
            state=(
                "readonly" if session.config["dimension"] == 3 else "disabled"
            ),
        )
        if self.axis_var.get() not in axes:
            self.axis_var.set(axes[-1])
        if center:
            normal = axes.index(self.axis_var.get())
            self.slice_var.set(slice_count(session, normal) // 2)
        self._slice_axis()
        scale = max(abs(session.initial_integral[0]), 1e-15)
        mass_residual = session.balance_residual[0] / scale
        self.audit.set(
            f"Displayed mass: {session.integral[0]:.6g} | "
            f"relative mass balance: {mass_residual:.3e} | "
            f"energy balance: {session.balance_residual[-1]:.3e}"
        )
        if hasattr(session, "mesh"):
            note = (
                " | refinement deferred: cell limit"
                if session.budget_limited
                else ""
            )
            self.audit.set(
                self.audit.get()
                + f" | leaves: {len(session.mesh.keys)}"
                + note
            )
        self.view_time.set(
            f"Displayed: t={session.time:.6g} s, step {session.steps}"
        )

    def _slice_axis(self, event=None):
        if self.displayed is None:
            return
        c = self.displayed.config
        axes = coordinates(c)
        normal = axes.index(self.axis_var.get())
        count = slice_count(self.displayed, normal)
        index = (
            count // 2
            if event is not None
            else min(max(0, self.slice_var.get()), count - 1)
        )
        self.adjusting = True
        self.slice_scale.configure(
            to=count - 1,
            state="normal" if c["dimension"] == 3 else "disabled",
        )
        self.slice_var.set(index)
        self.adjusting = False
        self.redraw()

    def _slice_index(self, value=None):
        if not self.adjusting:
            self.redraw()

    def redraw(self, event=None):
        if self.displayed is None or self.adjusting:
            return
        try:
            normal = coordinates(self.displayed.config).index(
                self.axis_var.get()
            )
            self.last_plane = draw_session(
                self.figure,
                self.displayed,
                self.field_var.get(),
                normal,
                self.slice_var.get(),
                self.mode_var.get(),
            )
            self.canvas.draw_idle()
        except (ValueError, IndexError) as exc:
            self._error(exc)

    def _history_index(self, value=None):
        if self.adjusting or not self.frames or self.runner.running:
            return
        index = min(len(self.frames) - 1, max(0, self.frame_var.get()))
        try:
            path = self.prefix.parent / (self.frames[index]["stem"] + ".h5")
            self.show_session(Simulation.load(path))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.stop_playback()
            self._error(exc)

    def play(self):
        if self.play_job is not None:
            self.stop_playback()
            return
        if not self.frames or self.runner.running:
            return
        self.playing = True
        self.play_button.configure(text="Pause playback")
        self._play_next()

    def _play_next(self):
        self.play_job = None
        index = self.frame_var.get() + 1
        if index >= len(self.frames):
            index = 0
        self.frame_var.set(index)
        self._history_index()
        if self.playing:
            self.play_job = self.after(250, self._play_next)

    def stop_playback(self):
        self.playing = False
        if self.play_job is not None:
            self.after_cancel(self.play_job)
            self.play_job = None
        self.play_button.configure(text="Play")

    def latest(self):
        self.stop_playback()
        if self.frames:
            self.frame_var.set(len(self.frames) - 1)
            self._history_index()

    def new(self):
        if self.runner.running:
            return
        self.stop_playback()
        if self.session is not None:
            self.session.close()
        self.session = self.displayed = None
        self.prefix = self.last_frame = None
        self.frames = []
        self.last_plane = None
        self.figure.clear()
        self.canvas.draw_idle()
        self.adjusting = True
        self.frame_scale.configure(to=0)
        self.frame_var.set(0)
        self.adjusting = False
        if self.vars["dimension"].get() == "1D":
            self._apply_spec(defaults(2))
        self._states()
        self.status.set("Configure a new simulation, then Run.")
        self.audit.set("No results yet")
        self.view_time.set("No recorded frames")

    def load(self, path, series=False):
        if self.runner.running:
            raise RuntimeError("Pause before loading another simulation")
        self.stop_playback()
        if series:
            suffix = ".series.json"
            if not str(path).endswith(suffix):
                raise ValueError("Choose a .series.json collection")
            prefix = Path(str(path)[: -len(suffix)])
            data = history_frames(prefix)
            session = Simulation.load(
                prefix.parent / (data["frames"][-1]["stem"] + ".h5")
            )
            if session.run_id != data["run_id"]:
                raise ValueError(
                    "Series identity does not match its final frame"
                )
        else:
            session = Simulation.load(path)
            prefix = new_prefix(self.vars["folder"].get())
        self.new()
        self._apply_spec(config_spec(session))
        self.session, self.prefix = session, prefix
        if series:
            self._refresh_history()
        self.show_session(session)
        self._states()
        self.status.set("Loaded. Resume uses the saved mesh and physics.")

    def choose_folder(self):
        if self.runner.running:
            return
        value = filedialog.askdirectory(parent=self)
        if value:
            self.vars["folder"].set(value)

    def load_dialog(self):
        path = filedialog.askopenfilename(
            parent=self,
            filetypes=[("TORCH restart", "*.h5")],
        )
        if path:
            try:
                self.load(path)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self._error(exc)

    def series_dialog(self):
        path = filedialog.askopenfilename(
            parent=self,
            filetypes=[("TORCH series", "*.series.json")],
        )
        if path:
            try:
                self.load(path, series=True)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self._error(exc)

    def save_dialog(self):
        if self.session is None or self.runner.running:
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".h5",
            filetypes=[("TORCH restart", "*.h5")],
        )
        if path:
            try:
                self.session.save(path)
                self.status.set("Saved current restart: " + path)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self._error(exc)

    def export_dialog(self):
        if self.displayed is None or self.runner.running:
            return
        extension = (
            ".vtr"
            if self.displayed.config.get("geometry", "cartesian")
            == "cartesian"
            else ".vts"
        )
        if hasattr(self.displayed, "mesh"):
            extension = ".vtu"
        path = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".xdmf",
            filetypes=[
                ("XDMF", "*.xdmf"),
                ("VTK", "*" + extension),
                ("CSV", "*.csv"),
                ("HDF5 snapshot", "*.h5"),
            ],
        )
        if path:
            try:
                export_session(path, self.displayed)
                self.status.set("Exported displayed frame: " + path)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self._error(exc)

    def figure_dialog(self):
        if self.displayed is None or self.runner.running:
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("SVG", "*.svg"),
            ],
        )
        if path:
            try:
                self.figure.savefig(path, dpi=200)
                self.status.set("Exported displayed figure: " + path)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self._error(exc)

    def _error(self, error):
        self.status.set("Error: " + str(error))
        messagebox.showerror("TORCH", str(error), parent=self)

    def destroy(self):
        if self.closed:
            return
        if self.runner.running:
            self.close()
            return
        self.stop_playback()
        if self.runner.thread is not None:
            self.runner.thread.join()
        if self.poll_job is not None:
            self.after_cancel(self.poll_job)
            self.poll_job = None
        idle = getattr(self.canvas, "_idle_draw_id", None)
        if idle is not None:
            self.canvas.get_tk_widget().after_cancel(idle)
            self.canvas._idle_draw_id = None
        if self.session is not None:
            self.session.close()
        resources, widgets = owned_resources(self, self.canvas)
        self.closed = True
        super().destroy()
        release(resources, widgets)

    def close(self):
        if self.closed:
            return
        self.stop_playback()
        self.closing = True
        self.runner.request_pause()
        if not self.runner.running:
            if self.poll_job is not None:
                self.after_cancel(self.poll_job)
            self.destroy()
