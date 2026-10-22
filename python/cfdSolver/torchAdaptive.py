"""Conservative 1D/2D/3D dyadic AMR with a shared-face SSPRK2 update."""

import copy
import numpy as np

from .torchAdaptiveGeometry import LeafMesh, children, settings
from .torchBoundaries import ghost, normalize_faces
from .torchFiniteVolume import primitive, conservative
from .torchFlux import rusanov_flux as rusanov
from .torchSession import Simulation
from .torchSolver import gamma_calc, physical_state, new_lambda_n_e_calc


def face_values(mesh, state, fractions, config, reconstruct=True):
    g = np.broadcast_to(
        gamma_calc(fractions[0]) if config["reactive"] else config["gamma"],
        state.shape[1:],
    )
    p = primitive(state, g)
    left, right = (
        p[:, np.maximum(mesh.left, 0)],
        p[:, np.maximum(mesh.right, 0)],
    )
    gl, gr = (
        g[np.maximum(mesh.left, 0)].copy(),
        g[np.maximum(mesh.right, 0)].copy(),
    )
    ql = [q[np.maximum(mesh.left, 0)].copy() for q in fractions]
    qr = [q[np.maximum(mesh.right, 0)].copy() for q in fractions]
    for k in range(config["dimension"]):
        for side in (-1, 1):
            mask = (mesh.directions == k) & (mesh.sides == side)
            if not mask.any():
                continue
            face = config["face_boundaries"][2 * k + (side == 1)]
            if side == -1:
                pp, gg, qs = ghost(
                    right[:, mask],
                    gr[mask],
                    [q[mask] for q in qr],
                    face,
                    k,
                    side,
                    config["reactive"],
                )
                left[:, mask], gl[mask] = pp, gg
                for a, b in zip(ql, qs):
                    a[mask] = b
            else:
                pp, gg, qs = ghost(
                    left[:, mask],
                    gl[mask],
                    [q[mask] for q in ql],
                    face,
                    k,
                    side,
                    config["reactive"],
                )
                right[:, mask], gr[mask] = pp, gg
                for a, b in zip(qr, qs):
                    a[mask] = b
    if reconstruct and config["amr"]["order"] == 2:
        gradient = mesh.gradients(p, left, right)
        for values, indices, offset in (
            (left, mesh.left, mesh.offset_left),
            (right, mesh.right, mesh.offset_right),
        ):
            mask = indices >= 0
            values[:, mask] += np.einsum(
                "nd,ndv->vn", offset[mask], gradient[indices[mask]]
            )
        # A wall reflects the reconstructed interior state, maintaining
        # exactly zero mass/energy/species flux even on fine wall faces.
        for k in range(config["dimension"]):
            for side in (-1, 1):
                face = config["face_boundaries"][2 * k + (side == 1)]
                if face["type"] != "wall":
                    continue
                mask = (mesh.directions == k) & (mesh.sides == side)
                if side == -1:
                    left[:, mask] = right[:, mask]
                    left[k + 1, mask] *= -1
                    gl[mask] = gr[mask]
                else:
                    right[:, mask] = left[:, mask]
                    right[k + 1, mask] *= -1
                    gr[mask] = gl[mask]
    return left, right, gl, gr, ql, qr


def source(mesh, p):
    c, d = mesh.config, mesh.dimension
    result = np.zeros_like(p)
    geometry = c["geometry"]
    if geometry == "cartesian":
        return result
    rho, pressure = p[0], p[-1]
    radial = mesh.divergence[0]
    result[1] = pressure * radial
    if geometry == "spherical":
        invr = radial / 2
        result[1] += rho * np.sum(p[2:-1] ** 2, axis=0) * invr
        if d >= 2:
            cotr = mesh.divergence[1]
            result[2] = pressure * cotr - rho * p[1] * p[2] * invr
            if d == 3:
                result[2] += rho * p[3] ** 2 * cotr
                result[3] = -rho * p[3] * (p[1] * invr + p[2] * cotr)
    elif d >= 2 and not (geometry == "cylindrical" and d == 2):
        result[1] += rho * p[2] ** 2 * radial
        result[2] = -rho * p[1] * p[2] * radial
    return result


def rhs(mesh, state, fractions, config, execution, first_order=False):
    left, right, gl, gr, ql, qr = face_values(
        mesh, state, fractions, config, not first_order
    )
    flux = np.empty_like(left)
    tasks = [
        (k, start, min(start + 8192, len(ids)), ids)
        for k in range(config["dimension"])
        for ids in [np.flatnonzero(mesh.directions == k)]
        for start in range(0, len(ids), 8192)
    ]

    def evaluate(task):
        k, start, stop, ids = task
        ids = ids[start:stop]
        return ids, rusanov(left[:, ids], right[:, ids], gl[ids], gr[ids], k)

    for ids, values in execution.map(evaluate, tasks):
        flux[:, ids] = values
    g = gamma_calc(fractions[0]) if config["reactive"] else config["gamma"]
    src = source(mesh, primitive(state, g))
    change = src * mesh.volumes
    species = [np.zeros_like(q) for q in fractions]
    weighted = flux * mesh.face_areas
    boundary = np.zeros(state.shape[0])
    for indices, sign, side in ((mesh.left, -1, 1), (mesh.right, 1, -1)):
        mask = indices >= 0
        np.add.at(change.T, indices[mask], sign * weighted.T[mask])
        boundary += side * np.sum(weighted[:, mesh.sides == side], axis=1)
        for q, a, b in zip(species, ql, qr):
            transported = weighted[0] * np.where(flux[0] >= 0, a, b)
            np.add.at(q, indices[mask], sign * transported[mask])
    return (
        change / mesh.volumes,
        [q / mesh.volumes for q in species],
        boundary,
        np.sum(src * mesh.volumes, axis=1),
    )


def cfl(mesh, state, fractions, config):
    left, right, gl, gr, _, _ = face_values(mesh, state, fractions, config)
    rate = np.zeros(len(mesh.keys))
    for k in range(config["dimension"]):
        mask = mesh.directions == k
        speed = np.maximum(
            abs(left[k + 1, mask])
            + np.sqrt(gl[mask] * left[-1, mask] / left[0, mask]),
            abs(right[k + 1, mask])
            + np.sqrt(gr[mask] * right[-1, mask] / right[0, mask]),
        )
        weighted = speed * mesh.face_areas[mask]
        for indices in (mesh.left[mask], mesh.right[mask]):
            active = indices >= 0
            np.add.at(rate, indices[active], weighted[active])
    return config["cfl"] / np.max(0.5 * rate / mesh.volumes)


def advance(mesh, state, fractions, dt, config, execution):
    conserved = [state[0] * q for q in fractions]
    for first in (False, True):
        k1, q1, b1, s1 = rhs(mesh, state, fractions, config, execution, first)
        stage = state + dt * k1
        if not physical_state(stage):
            continue
        stageq = [(q + dt * k) / stage[0] for q, k in zip(conserved, q1)]
        if any(
            not np.isfinite(q).all() or np.any((q < -1e-12) | (q > 1 + 1e-12))
            for q in stageq
        ):
            continue
        stageq = [np.clip(q, 0, 1) for q in stageq]
        k2, q2, b2, s2 = rhs(mesh, stage, stageq, config, execution, first)
        final = 0.5 * (state + stage + dt * k2)
        if not physical_state(final):
            continue
        finalq = [
            0.5 * (q + stage[0] * f + dt * k) / final[0]
            for q, f, k in zip(conserved, stageq, q2)
        ]
        if any(
            not np.isfinite(q).all() or np.any((q < -1e-12) | (q > 1 + 1e-12))
            for q in finalq
        ):
            continue
        return (
            final,
            [np.clip(q, 0, 1) for q in finalq],
            0.5 * dt * (b1 + b2),
            0.5 * dt * (s1 + s2),
        )
    raise RuntimeError("Adaptive update lost positivity")


def remesh(mesh, state, fractions, config, refine=None, coarsen=True):
    """Transactional sibling coarsening and face-balanced refinement."""
    left, right, gl, gr, _, _ = face_values(
        mesh, state, fractions, config, False
    )
    indicator = np.zeros(len(mesh.keys))
    jumps = np.max(
        abs(left[[0, -1]] - right[[0, -1]])
        / np.maximum(abs(left[[0, -1]]) + abs(right[[0, -1]]), 1e-14),
        axis=0,
    )
    for indices in (mesh.left, mesh.right):
        mask = indices >= 0
        np.maximum.at(indicator, indices[mask], jumps[mask])
    options = config["amr"]
    tag = (
        indicator > options["refine_threshold"]
        if refine is None
        else np.asarray(refine, dtype=bool)
    )
    if tag.shape != indicator.shape:
        raise ValueError("Refinement mask must match leaves")
    tag = (tag | (mesh.levels < options["min_level"])) & (
        mesh.levels < options["max_level"]
    )
    i, j = mesh.left[mesh.interior], mesh.right[mesh.interior]
    for _ in range(options["max_level"] + 1):
        old = tag.copy()
        a, b = mesh.levels[i] + tag[i], mesh.levels[j] + tag[j]
        tag[j[a > b + 1]] = True
        tag[i[b > a + 1]] = True
        if np.array_equal(old, tag):
            break
    groups = {}
    if coarsen:
        for idx, key in enumerate(mesh.keys):
            if (
                key[0] <= options["min_level"]
                or tag[idx]
                or indicator[idx] >= options["coarsen_threshold"]
            ):
                continue
            parent = (int(key[0]) - 1, *(int(v) // 2 for v in key[1:]))
            groups.setdefault(parent, []).append(idx)
    accepted = {}
    for parent, ids in groups.items():
        if len(ids) != 2**mesh.dimension:
            continue
        membership = np.zeros(len(tag), dtype=bool)
        membership[ids] = True
        neighbor = np.concatenate(
            (
                j[membership[i] & ~membership[j]],
                i[membership[j] & ~membership[i]],
            )
        )
        if np.any(mesh.levels[neighbor] + tag[neighbor] > parent[0] + 1):
            continue
        accepted[parent] = ids
    removed = {idx for ids in accepted.values() for idx in ids}
    count = (
        len(tag)
        + int(tag.sum()) * (2**mesh.dimension - 1)
        - len(accepted) * (2**mesh.dimension - 1)
    )
    budget_limited = count > options["max_cells"]
    if budget_limited:
        # Coarsening stays valid when refinements are deferred; capacity
        # never causes a partial or unbalanced mesh transaction.
        tag[:] = False
    if not np.any(tag) and not accepted:
        return mesh, state, fractions, budget_limited, False
    data = {}
    for parent, ids in accepted.items():
        weights = mesh.volumes[ids]
        u = np.sum(state[:, ids] * weights, axis=1) / np.sum(weights)
        qs = [
            np.sum(state[0, ids] * q[ids] * weights)
            / np.sum(state[0, ids] * weights)
            for q in fractions
        ]
        data[parent] = (u, qs)
    # A limited conservative reconstruction preserves linear fields at
    # refinement. Correct the volume-weighted mean on curved cells.
    ul, ur = conservative(left, gl), conservative(right, gr)
    gradient = (
        mesh.gradients(state, ul, ur)
        if np.any(tag) and options["order"] == 2
        else None
    )
    for idx, key_array in enumerate(mesh.keys):
        if idx in removed:
            continue
        key = tuple(int(v) for v in key_array)
        if not tag[idx]:
            data[key] = (state[:, idx], [q[idx] for q in fractions])
            continue
        group = children(key)
        level = key[0] + 1
        low = (
            mesh.origin
            + np.array([k[1:] for k in group])
            * np.array(config["lengths_xyz"])
            / np.array(config["counts_xyz"])
            / 2**level
        )
        high = (
            low
            + np.array(config["lengths_xyz"])
            / np.array(config["counts_xyz"])
            / 2**level
        )
        from .torchAdaptiveGeometry import volume

        weights = volume(config, low, high)
        values = np.repeat(state[:, idx, None], len(group), axis=1)
        if gradient is not None:
            change = (0.5 * (low + high) - mesh.centers[idx]) @ gradient[idx]
            change -= np.sum(change * weights[:, None], axis=0) / np.sum(
                weights
            )
            trial = values + change.T
            if physical_state(trial):
                values = trial
        for k, value in zip(group, values.T):
            data[k] = (value, [q[idx] for q in fractions])
    keys = np.array(list(data), dtype=np.int64)
    candidate = LeafMesh(config, keys)
    out = np.stack([data[tuple(key)][0] for key in candidate.keys], axis=1)
    qs = [
        np.array([data[tuple(key)][1][a] for key in candidate.keys])
        for a in range(len(fractions))
    ]
    changed = not np.array_equal(mesh.keys, candidate.keys)
    return candidate if changed else mesh, out, qs, budget_limited, changed


class AdaptiveSimulation(Simulation):
    """Leaf simulation with fixed root counts in configuration metadata.
    """

    @classmethod
    def from_uniform(cls, session, **options):
        if isinstance(session, cls):
            raise ValueError("Session is already adaptive")
        result = cls.__new__(cls)
        result.config = copy.deepcopy(session.config)
        result.config["amr"] = settings(**options)
        result.config["face_boundaries"] = normalize_faces(
            result.config, result.config.get("face_boundaries")
        )
        result._validate_config()
        result.mesh = LeafMesh(result.config)
        result.state = session.state.reshape(
            result.config["dimension"] + 2, -1
        ).copy()
        result.lam = (
            None if session.lam is None else session.lam.ravel().copy()
        )
        result.beta = (
            None if session.beta is None else session.beta.ravel().copy()
        )
        result.time, result.steps, result.run_id = (
            session.time,
            session.steps,
            session.run_id,
        )
        for key in (
            "initial_integral",
            "boundary_integral",
            "source_integral",
        ):
            setattr(result, key, getattr(session, key).copy())
        result.regrids, result.mesh_revision, result.budget_limited = (
            0,
            0,
            False,
        )
        result.set_workers(session.workers)
        for _ in range(result.config["amr"]["max_level"]):
            result.regrid()
        return result

    @property
    def integral(self):
        return np.sum(self.state * self.mesh.volumes, axis=1)

    def _fraction(self, values, name):
        q = np.asarray(values, dtype=float)
        if (
            q.shape != (len(self.mesh.keys),)
            or not np.isfinite(q).all()
            or np.any((q < 0) | (q > 1))
        ):
            raise ValueError(f"Invalid adaptive {name}")
        return q.copy()

    def regrid(self, refine=None, coarsen=True):
        fractions = [q for q in (self.lam, self.beta) if q is not None]
        mesh, state, qs, budget, changed = remesh(
            self.mesh, self.state, fractions, self.config, refine, coarsen
        )
        self.mesh, self.state = mesh, state
        self.lam = qs[0] if qs else None
        self.beta = qs[1] if len(qs) > 1 else None
        self.regrids += 1
        self.mesh_revision += int(changed)
        self.budget_limited = budget

    def _reaction_flat(self, state, fractions, dt):
        normal = np.stack(
            (
                state[0],
                state[1],
                state[-1] - 0.5 * np.sum(state[2:-1] ** 2, axis=0) / state[0],
            )
        )
        lam, reacted, beta = new_lambda_n_e_calc(
            fractions[0],
            dt,
            normal,
            self.config["kinetic_model"],
            fractions[1] if len(fractions) > 1 else None,
        )
        result = state.copy()
        result[-1] += reacted[-1] - normal[-1]
        return result, [lam] + ([beta] if len(fractions) > 1 else [])

    def step(self):
        if self.completed:
            return
        c = self.config
        if self.steps >= c["max_steps"]:
            raise RuntimeError("Maximum steps reached")
        initial = [q for q in (self.lam, self.beta) if q is not None]
        dt = min(
            cfl(self.mesh, self.state, initial, c), c["t_end"] - self.time
        )
        for _ in range(24):
            if not np.isfinite(dt) or dt <= 0 or self.time + dt == self.time:
                raise RuntimeError("Adaptive integration stalled")
            old, fractions = self.state, initial
            if c["reactive"]:
                old, fractions = self._reaction_flat(
                    self.state, initial, dt / 2
                )
                allowed = cfl(self.mesh, old, fractions, c)
                if dt > allowed * (1 + 1e-12):
                    dt = min(dt / 2, allowed)
                    continue
            try:
                hydro, qs, boundary, src = advance(
                    self.mesh, old, fractions, dt, c, self.execution
                )
            except RuntimeError:
                dt /= 2
                continue
            state = hydro
            if c["reactive"]:
                state, qs = self._reaction_flat(hydro, qs, dt / 2)
                src += np.sum(
                    (old - self.state + state - hydro) * self.mesh.volumes,
                    axis=1,
                )
            if not physical_state(state):
                raise RuntimeError("Nonphysical adaptive reaction")
            mesh = self.mesh
            budget = self.budget_limited
            changed = False
            do_regrid = (self.steps + 1) % c["amr"]["regrid_every"] == 0
            if do_regrid:
                mesh, state, qs, budget, changed = remesh(mesh, state, qs, c)
            self.mesh, self.state = mesh, state
            self.lam = qs[0] if qs else None
            self.beta = qs[1] if len(qs) > 1 else None
            self.boundary_integral += boundary
            self.source_integral += src
            self.time = min(self.time + dt, c["t_end"])
            self.steps += 1
            self.regrids += int(do_regrid)
            self.mesh_revision += int(changed)
            self.budget_limited = budget
            return
        raise RuntimeError("Could not find a positive adaptive step")
