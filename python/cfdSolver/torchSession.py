"""Step-boundary simulation sessions for bounded-memory execution and restart.

Legacy drivers remain unchanged. Sessions reuse their numerical kernels and
store only the current state, composition and integrated conservation balances.
"""

import json
import uuid
from functools import cached_property
import numpy as np
from . import torchSolver as one
from . import torchSolver2D as two
from . import torchSolver3D as three


class Simulation:
    """Cartesian or orthogonal metric simulation, resumable between steps."""

    def __init__(
        self,
        primitive,
        lengths,
        *,
        t_end,
        cfl=0.4,
        gamma=1.4,
        boundaries=None,
        limiter="VanLeer",
        reactive=False,
        kinetic_model="arrhenius",
        lambda_init=None,
        beta_init=None,
        inlet_primitive=None,
        controller="Sound Speed Controlled",
        variant="Variante 1 - Original",
        wavespeed="Pressure Based",
        max_steps=100000,
        geometry="cartesian",
        origins=None,
        face_boundaries=None,
    ):
        primitive = np.array(primitive, dtype=float, copy=True)
        dim = primitive.ndim - 1
        if dim not in (1, 2, 3) or primitive.shape[0] != dim + 2:
            raise ValueError(
                "Require primitives with shape (dimension+2, ... reversed spatial axes)"
            )
        lengths = list(np.atleast_1d(lengths))
        counts = list(primitive.shape[1:][::-1])
        if len(lengths) != dim:
            raise ValueError("Lengths must be in x,y,z order, one per active axis")
        boundaries = list(boundaries) if boundaries is not None else ["Open"] * dim
        self.config = dict(
            dimension=dim,
            counts_xyz=counts,
            lengths_xyz=lengths,
            boundaries_xyz=boundaries,
            t_end=t_end,
            cfl=cfl,
            gamma=gamma,
            limiter=limiter,
            reactive=bool(reactive),
            kinetic_model=kinetic_model,
            inlet_primitive=list(
                [1.0, 0.0, 100.0, 101325.0]
                if inlet_primitive is None
                else inlet_primitive
            ),
            controller=controller,
            variant=variant,
            wavespeed=wavespeed,
            max_steps=max_steps,
            geometry=geometry,
            origins_xyz=(
                list(np.atleast_1d(origins)) if origins is not None else [0.0] * dim
            ),
        )
        if geometry != "cartesian" or face_boundaries is not None:
            from .torchBoundaries import normalize_faces

            if (
                boundaries == ["Open"] * dim
                and geometry != "cartesian"
                and (face_boundaries is None or isinstance(face_boundaries, dict))
            ):
                provided_faces = face_boundaries or {}
                face_boundaries = {}
                if self.config["origins_xyz"][0] == 0:
                    face_boundaries["x_min"] = "wall"
                if geometry == "spherical" and dim >= 2:
                    if self.config["origins_xyz"][1] == 0:
                        face_boundaries["y_min"] = "wall"
                    if np.isclose(
                        self.config["origins_xyz"][1] + lengths[1],
                        np.pi,
                        rtol=0,
                        atol=1e-14,
                    ):
                        face_boundaries["y_max"] = "wall"
                face_boundaries.update(provided_faces)
            self.config["face_boundaries"] = normalize_faces(
                self.config, face_boundaries
            )
        self._validate_config()
        if not np.isfinite(primitive).all() or np.any(primitive[[0, -1]] <= 0):
            raise ValueError(
                "Initial density and pressure must be positive and primitives finite"
            )
        self.lam = self._fraction(lambda_init, "lambda") if reactive else None
        self.beta = (
            self._fraction(beta_init, "beta")
            if reactive and kinetic_model == "znd"
            else None
        )
        if (
            not reactive
            and lambda_init is not None
            or beta_init is not None
            and self.beta is None
        ):
            raise ValueError(
                "Composition requires reactive chemistry; beta requires ZND"
            )
        g = self.gamma
        self.state = primitive.copy()
        self.state[1:-1] *= primitive[0]
        self.state[-1] = primitive[-1] / (g - 1) + 0.5 * primitive[0] * np.sum(
            primitive[1:-1] ** 2, axis=0
        )
        if not one.physical_state(self.state):
            raise ValueError("Nonphysical initial conservative state")
        self.time, self.steps = 0.0, 0
        self.run_id = str(uuid.uuid4())
        self.initial_integral = self.integral.copy()
        self.boundary_integral = np.zeros(dim + 2)
        self.source_integral = np.zeros(dim + 2)

    def _validate_config(self):
        c = self.config
        dim = c["dimension"]
        from .torchGeometry import validate_geometry
        from .torchBoundaries import normalize_faces

        if (
            not isinstance(dim, (int, np.integer))
            or dim not in (1, 2, 3)
            or isinstance(dim, bool)
        ):
            raise ValueError("Invalid dimension")
        for key in ("counts_xyz", "lengths_xyz", "boundaries_xyz"):
            if len(c[key]) != dim:
                raise ValueError("Incomplete mesh/boundary metadata")
        for length, n in zip(c["lengths_xyz"], c["counts_xyz"]):
            one.validate_grid(length, n, c["cfl"], c["t_end"])
        validate_geometry(c)
        if c.get("geometry", "cartesian") != "cartesian" or "face_boundaries" in c:
            c["face_boundaries"] = normalize_faces(c, c.get("face_boundaries"))
            if (c["controller"], c["variant"], c["wavespeed"]) != (
                "Sound Speed Controlled",
                "Variante 1 - Original",
                "Pressure Based",
            ):
                raise ValueError(
                    "Metric/face sessions use MUSCL-Rusanov/SSPRK2; legacy HLLC switches are unsupported"
                )
        if not np.isfinite(c["gamma"]) or c["gamma"] <= 1:
            raise ValueError("gamma must be finite and greater than one")
        if c["limiter"] not in three.LIMITERS or c["kinetic_model"] not in (
            "arrhenius",
            "znd",
        ):
            raise ValueError("Unknown limiter or chemistry model")
        if (
            not isinstance(c["max_steps"], (int, np.integer))
            or isinstance(c["max_steps"], bool)
            or c["max_steps"] < 1
        ):
            raise ValueError("Invalid maximum steps")
        for axis, boundary in enumerate(c["boundaries_xyz"]):
            if boundary not in ("Open", "Reflective", "Periodic") and not (
                dim == 2 and axis == 1 and boundary == "Inflow"
            ):
                raise ValueError("Unsupported session boundary")
        one.validate_options_1d(
            c["limiter"], c["controller"], c["variant"], c["wavespeed"], "Open"
        )
        if dim == 2 and "Inflow" in c["boundaries_xyz"]:
            inlet = np.asarray(c["inlet_primitive"])
            if (
                inlet.shape != (4,)
                or not np.isfinite(inlet).all()
                or inlet[0] <= 0
                or inlet[3] <= 0
                or inlet[2] < 0
            ):
                raise ValueError("Invalid inlet primitives")

    def _fraction(self, values, name):
        shape = tuple(self.config["counts_xyz"][::-1])
        result = (
            np.ones(shape)
            if values is None
            else np.array(values, dtype=float, copy=True)
        )
        if (
            result.shape != shape
            or not np.isfinite(result).all()
            or np.any((result < 0) | (result > 1))
        ):
            raise ValueError(f"{name} must have the spatial shape and values in [0,1]")
        return result

    @property
    def gamma(self):
        return (
            one.gamma_calc(self.lam)
            if self.config["reactive"]
            else self.config["gamma"]
        )

    @property
    def spacing(self):
        return np.array(self.config["lengths_xyz"]) / self.config["counts_xyz"]

    @property
    def integral(self):
        if self.config.get("geometry", "cartesian") != "cartesian":
            return (
                (self.state * self.metrics.volumes[None])
                .reshape(self.config["dimension"] + 2, -1)
                .sum(axis=1)
            )
        return self.state.reshape(self.config["dimension"] + 2, -1).sum(
            axis=1
        ) * np.prod(self.spacing)

    @cached_property
    def metrics(self):
        from .torchGeometry import Metrics

        return Metrics(self.config)

    @property
    def uses_metric_kernel(self):
        return (
            self.config.get("geometry", "cartesian") != "cartesian"
            or "face_boundaries" in self.config
        )

    @property
    def primitive(self):
        p = self.state.copy()
        p[1:-1] /= self.state[0]
        p[-1] = (self.gamma - 1) * (
            self.state[-1] - 0.5 * np.sum(self.state[1:-1] ** 2, axis=0) / self.state[0]
        )
        return p

    @property
    def balance_residual(self):
        return (
            self.integral
            - self.initial_integral
            + self.boundary_integral
            - self.source_integral
        )

    @property
    def completed(self):
        return self.time == self.config["t_end"]

    def _dt(self, state, gamma):
        c = self.config
        dim = c["dimension"]
        h = self.spacing
        bc = c["boundaries_xyz"]
        if dim == 1:
            padded = one.bc_creator(state, bc[0])
            gp = gamma if np.isscalar(gamma) else one.scalar_bc(gamma, bc[0])
            waves = one.wave_speed_estimate(padded, gp, c["wavespeed"])[3]
            return min(
                one.delta_t_estimate(waves, h[0], c["cfl"]),
                one.hydro_dt_1d(padded, gp, h[0], c["cfl"]),
            )
        if dim == 2:
            padded = two.apply_boundary_conditions_2d(state, *bc, c["inlet_primitive"])
            gp = two.pad_scalar_2d(
                np.broadcast_to(gamma, state.shape[1:]), *bc, inlet_value=1.4
            )
            return two.compute_cfl_dt_2d(
                two.uconsv_to_uprim_2d(padded, gp), *h, c["cfl"], gp
            )
        p = three.uconsv_to_uprim_3d(state, gamma)
        sound = np.sqrt(gamma * p[-1] / p[0])
        return c["cfl"] / sum(
            np.max(abs(p[m]) + sound) / dx for m, dx in enumerate(h, 1)
        )

    def _reaction(self, state, lam, beta, dt):
        dim = self.config["dimension"]
        model = self.config["kinetic_model"]
        if dim == 1:
            lam, state, beta = one.new_lambda_n_e_calc(lam, dt, state, model, beta)
            return state, lam, beta if model == "znd" else None
        if dim == 2:
            return two.reaction_step_2d(state, lam, beta, dt, model)
        return three.reaction_step_3d(state, lam, beta, dt, model)

    def step(self):
        """Advance one complete split step; failed steps leave session data intact."""
        if self.completed:
            return
        if self.uses_metric_kernel:
            self._metric_step()
            return
        c = self.config
        dim = c["dimension"]
        h = self.spacing
        bc = c["boundaries_xyz"]
        if self.steps >= c["max_steps"]:
            raise RuntimeError("Maximum steps reached before requested final time")
        dt = min(self._dt(self.state, self.gamma), c["t_end"] - self.time)
        if not np.isfinite(dt) or dt <= 0 or self.time + dt == self.time:
            raise RuntimeError("Integration stalled before requested final time")
        start = self.state
        old = start
        lam = self.lam
        beta = self.beta
        if c["reactive"]:
            for _ in range(60):
                old, lam, beta = self._reaction(start, self.lam, self.beta, dt / 2)
                allowed = self._dt(old, one.gamma_calc(lam))
                if dt <= allowed * (1 + 1e-12):
                    break
                dt = min(dt / 2, allowed)
            else:
                raise RuntimeError("Could not find stable reactive step")
        gamma = one.gamma_calc(lam) if c["reactive"] else c["gamma"]
        if dim == 1:
            padded = one.bc_creator(old, bc[0])
            gp = gamma if np.isscalar(gamma) else one.scalar_bc(gamma, bc[0])
            waves = one.wave_speed_estimate(padded, gp, c["wavespeed"])[3]
            reconstructed = one.muscl_application(
                padded, dt, h[0], gp, c["cfl"], c["limiter"], c["controller"], bc[0]
            )
            flux = one.hllc_flux_calc_for_muscl(reconstructed, waves, gp, c["variant"])
            state, flux = one.conservative_update_1d(padded, flux, dt, h[0], gp)
            fluxes = [flux]
            if c["reactive"]:
                lam = one.advect_scalar_1d(lam, old, state, flux[0], dt / h[0], bc[0])
                if beta is not None:
                    beta = one.advect_scalar_1d(
                        beta, old, state, flux[0], dt / h[0], bc[0]
                    )
        elif dim == 2:
            state, *fluxes = two.muscl_hancock_step_2d(
                old,
                *h,
                dt,
                gamma,
                c["limiter"],
                *bc,
                return_fluxes=True,
                inlet_primitive=c["inlet_primitive"],
            )
            if c["reactive"]:
                lam = two.advect_scalar_2d(
                    lam,
                    old,
                    state,
                    fluxes[0][0],
                    fluxes[1][0],
                    dt / h[0],
                    dt / h[1],
                    *bc,
                )
                if beta is not None:
                    beta = two.advect_scalar_2d(
                        beta,
                        old,
                        state,
                        fluxes[0][0],
                        fluxes[1][0],
                        dt / h[0],
                        dt / h[1],
                        *bc,
                    )
        else:
            state, fluxes = three._hydro(
                old, gamma, h, dt, c["limiter"], tuple(bc[::-1])
            )
            if c["reactive"]:
                lam = three._advect(lam, old, state, fluxes, dt, h, tuple(bc[::-1]))
                if beta is not None:
                    beta = three._advect(
                        beta, old, state, fluxes, dt, h, tuple(bc[::-1])
                    )
        outward = np.zeros(dim + 2)
        volume = np.prod(h)
        for direction, flux in enumerate(fluxes):
            axis = dim - direction
            faces = np.take(flux, -1, axis=axis) - np.take(flux, 0, axis=axis)
            outward += (
                dt * volume / h[direction] * faces.reshape(dim + 2, -1).sum(axis=1)
            )
        source = np.zeros(dim + 2)
        if c["reactive"]:
            hydro = state
            state, lam, beta = self._reaction(hydro, lam, beta, dt / 2)
            source = volume * (old - start + state - hydro).reshape(dim + 2, -1).sum(
                axis=1
            )
        if not one.physical_state(state):
            raise RuntimeError("Reaction produced a nonphysical state")
        self.state, self.lam, self.beta = state, lam, beta
        self.boundary_integral += outward
        self.source_integral += source
        self.time = min(self.time + dt, c["t_end"])
        self.steps += 1

    def _metric_step(self):
        from .torchFiniteVolume import cfl_dt, advance

        c = self.config
        if self.steps >= c["max_steps"]:
            raise RuntimeError("Maximum steps reached before requested final time")
        start = self.state
        dt = min(cfl_dt(start, self.lam, c, self.metrics), c["t_end"] - self.time)
        for retry in range(20):
            if not np.isfinite(dt) or dt <= 0 or self.time + dt == self.time:
                raise RuntimeError("Integration stalled before requested final time")
            old, lam, beta = start, self.lam, self.beta
            if c["reactive"]:
                old, lam, beta = self._reaction(start, self.lam, self.beta, dt / 2)
                allowed = cfl_dt(old, lam, c, self.metrics)
                if dt > allowed * (1 + 1e-12):
                    dt = min(dt / 2, allowed)
                    continue
            try:
                hydro, lam, beta, boundary, source = advance(
                    old, lam, beta, dt, c, self.metrics
                )
            except RuntimeError:
                dt /= 2
                continue
            state = hydro
            if c["reactive"]:
                state, lam, beta = self._reaction(hydro, lam, beta, dt / 2)
                source += (
                    ((old - start + state - hydro) * self.metrics.volumes[None])
                    .reshape(c["dimension"] + 2, -1)
                    .sum(axis=1)
                )
            if not one.physical_state(state):
                raise RuntimeError("Reaction produced a nonphysical state")
            self.state, self.lam, self.beta = state, lam, beta
            self.boundary_integral += boundary
            self.source_integral += source
            self.time = min(self.time + dt, c["t_end"])
            self.steps += 1
            return
        raise RuntimeError("Could not find a positive metric/face step")

    def run(
        self,
        *,
        step_limit=None,
        checkpoint=None,
        checkpoint_every=0,
        output_prefix=None,
        output_every=1,
        formats=("vtk", "xdmf"),
    ):
        """Run to configured end time, or pause after step_limit additional steps.

        Periodic outputs/checkpoints never shorten a numerical step. Restart
        therefore preserves the original time-step sequence and final balances.
        """
        for value in (step_limit, checkpoint_every, output_every):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError("Step intervals must be nonnegative integers")
        if checkpoint_every and not checkpoint:
            raise ValueError("checkpoint_every requires a checkpoint destination")
        if output_prefix and output_every < 1:
            raise ValueError("output_every must be positive")
        from .torchIO import write_checkpoint, TemporalWriter

        writer = TemporalWriter(output_prefix, self, formats) if output_prefix else None
        if writer:
            writer.append(self)
        count = 0
        while not self.completed and (step_limit is None or count < step_limit):
            self.step()
            count += 1
            if checkpoint and checkpoint_every and self.steps % checkpoint_every == 0:
                write_checkpoint(checkpoint, self)
            if writer and self.steps % output_every == 0:
                writer.append(self)
        if checkpoint:
            write_checkpoint(checkpoint, self)
        if writer:
            writer.append(self)
        return self

    def save(self, path):
        from .torchIO import write_checkpoint

        write_checkpoint(path, self)

    @classmethod
    def load(cls, path, *, t_end=None):
        from .torchIO import read_checkpoint

        result = read_checkpoint(path)
        if t_end is not None:
            if not np.isfinite(t_end) or t_end < result.time:
                raise ValueError(
                    "New end time must be finite and not precede checkpoint"
                )
            result.config["t_end"] = t_end
        return result

    @classmethod
    def preset(cls, name, counts, *, lengths=None, **options):
        """Create a session from an existing benchmark, using x,y,z counts."""
        counts = list(np.atleast_1d(counts))
        dim = len(counts)
        lengths = lengths if lengths is not None else [1.0] * dim
        if dim == 1:
            cfg = one.get_preset(name)
            n = counts[0]
            divider = max(1, min(n - 1, int(np.round(0.5 * n))))
            p = np.empty((3, n))
            p[:, :divider] = np.array([cfg.rho_l, cfg.u_l, cfg.p_l])[:, None]
            p[:, divider:] = np.array([cfg.rho_r, cfg.u_r, cfg.p_r])[:, None]
            defaults = dict(
                t_end=cfg.t_end,
                cfl=cfg.cfl,
                limiter=cfg.limiter,
                reactive=cfg.mode == "reactive",
            )
        elif dim == 2:
            cfg = two.get_2d_preset(name, *counts)
            p = np.stack([cfg["rho"], cfg["u"], cfg["v"], cfg["p"]])
            defaults = dict(
                t_end=cfg["t_end"],
                boundaries=[
                    {"Transmissive": "Open", "Inflow": "Inflow"}.get(cfg[k], cfg[k])
                    for k in ("bc_x", "bc_y")
                ],
                reactive=cfg["is_reactive"],
            )
        elif dim == 3:
            cfg = three.get_3d_preset(name, *counts)
            p = cfg["primitive"]
            defaults = dict(t_end=cfg["t_end"], reactive=cfg["is_reactive"])
        else:
            raise ValueError("Only dimensions 1,2,3 are supported")
        defaults.update(options)
        return cls(p, lengths, **defaults)
