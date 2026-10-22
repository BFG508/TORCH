"""Metric-aware MUSCL/Rusanov finite volumes with SSPRK2 and face
boundaries.

Second-order primitive reconstruction on uniform logical grids;
conservative
species remain first order. Failed reconstruction falls back to first
order.
"""

import numpy as np
from .torchBoundaries import ghost, normalize_faces
from .torchGeometry import Metrics
from .torchSolver import gamma_calc, physical_state


def primitive(state, gamma):
    p = state.copy()
    p[1:-1] /= state[0]
    p[-1] = (gamma - 1) * (
        state[-1] - 0.5 * np.sum(state[1:-1] ** 2, axis=0) / state[0]
    )
    return p


def conservative(p, gamma):
    u = p.copy()
    u[1:-1] *= p[0]
    u[-1] = p[-1] / (gamma - 1) + 0.5 * p[0] * np.sum(p[1:-1] ** 2, axis=0)
    return u


def face_states(p, gamma, fractions, c, direction):
    dim = c["dimension"]
    axis = dim - 1 - direction
    faces = c.get("face_boundaries") or normalize_faces(c)
    pp, gg, qq = [], [], [[] for _ in fractions]
    for end, side in ((0, -1), (-1, 1)):
        face = faces[2 * direction + (side == 1)]
        index = -1 if side == -1 else 0
        if face["type"] != "periodic":
            index = end
        pf = np.take(p, index, axis=axis + 1)
        gf = np.take(gamma, index, axis=axis)
        qf = [np.take(q, index, axis=axis) for q in fractions]
        if face["type"] != "periodic":
            pf, gf, qf = ghost(
                pf, gf, qf, face, direction, side, c["reactive"]
            )
        pp.append(np.expand_dims(pf, axis + 1))
        gg.append(np.expand_dims(gf, axis))
        for group, q in zip(qq, qf):
            group.append(np.expand_dims(q, axis))
    return (
        np.concatenate((pp[0], p, pp[1]), axis=axis + 1),
        np.concatenate((gg[0], gamma, gg[1]), axis=axis),
        [
            np.concatenate((group[0], q, group[1]), axis=axis)
            for q, group in zip(fractions, qq)
        ],
    )


def cfl_dt(state, lam, c, metrics=None):
    metrics = metrics or Metrics(c)
    gamma = np.broadcast_to(
        gamma_calc(lam) if c["reactive"] else c["gamma"], state.shape[1:]
    )
    p = primitive(state, gamma)
    rate = np.zeros(state.shape[1:])
    for d, area in enumerate(metrics.areas):
        axis = c["dimension"] - 1 - d
        padded, gp, _ = face_states(p, gamma, [], c, d)
        speed = abs(padded[d + 1]) + np.sqrt(gp * padded[-1] / padded[0])
        indices = np.arange(state.shape[axis + 1] + 1)
        face_speed = np.maximum(
            np.take(speed, indices, axis=axis),
            np.take(speed, indices + 1, axis=axis),
        )
        rate += (
            np.maximum(
                np.take(area * face_speed, indices[:-1], axis=axis),
                np.take(area * face_speed, indices[1:], axis=axis),
            )
            / metrics.volumes
        )
    return c["cfl"] / np.max(rate)


def limited(minus, plus, name):
    r = minus / np.where(abs(plus) < 1e-14, 1e-14, plus)
    pos = np.maximum(r, 0.0)
    if name == "VanLeer":
        phi = 2 * pos / (1 + pos)
    elif name == "Albada":
        phi = (pos * pos + pos) / (pos * pos + 1)
    elif name == "MINBEE":
        phi = np.maximum(0, np.minimum(r, 1))
    elif name == "Superbee":
        phi = np.maximum(0, np.maximum(np.minimum(2 * r, 1), np.minimum(r, 2)))
    else:
        phi = np.maximum(
            0,
            np.minimum(
                np.minimum(2, 2 * r), np.minimum((2 + r) / 3, (1 + 2 * r) / 3)
            ),
        )
    return phi * plus


def rhs(state, fractions, c, metrics, first_order=False):
    dim = c["dimension"]
    gamma = np.broadcast_to(
        gamma_calc(fractions[0]) if c["reactive"] else c["gamma"],
        state.shape[1:],
    )
    p = primitive(state, gamma)
    source = metrics.source(p)
    change = source.copy()
    species = [np.zeros_like(q) for q in fractions]
    outward = np.zeros(dim + 2)
    for d, area in enumerate(metrics.areas):
        axis = dim - 1 - d
        pp, gg, qq = face_states(p, gamma, fractions, c, d)
        slope = np.zeros_like(pp)
        middle = [slice(None)] * (dim + 1)
        middle[axis + 1] = slice(1, -1)
        before, after = middle.copy(), middle.copy()
        before[axis + 1] = slice(None, -2)
        after[axis + 1] = slice(2, None)
        if not first_order:
            slope[tuple(middle)] = limited(
                pp[tuple(middle)] - pp[tuple(before)],
                pp[tuple(after)] - pp[tuple(middle)],
                c["limiter"],
            )
            invalid = np.any(
                (pp[[0, -1]] - 0.5 * abs(slope[[0, -1]])) <= 1e-12, axis=0
            )
            slope *= (~invalid)[None]
            boundaries = c.get("face_boundaries") or normalize_faces(c)
            if boundaries[2 * d]["type"] == "periodic":
                for target, source_index in ((0, -2), (-1, 1)):
                    sl = [slice(None)] * (dim + 1)
                    sl[axis + 1] = target
                    slope[tuple(sl)] = np.take(
                        slope, source_index, axis=axis + 1
                    )
        indices = np.arange(state.shape[axis + 1] + 1)
        left = np.take(pp + 0.5 * slope, indices, axis=axis + 1)
        right = np.take(pp - 0.5 * slope, indices + 1, axis=axis + 1)
        gl, gr = np.take(gg, indices, axis=axis), np.take(
            gg, indices + 1, axis=axis
        )
        boundaries = c.get("face_boundaries") or normalize_faces(c)
        for end, side in ((0, 0), (-1, 1)):
            if boundaries[2 * d + side]["type"] == "wall":
                index = [slice(None)] * (dim + 1)
                index[axis + 1] = end
                scalar_index = tuple(index[1:])
                if side == 0:
                    left[tuple(index)] = right[tuple(index)]
                    gl[scalar_index] = gr[scalar_index]
                    index[0] = d + 1
                    left[tuple(index)] *= -1
                else:
                    right[tuple(index)] = left[tuple(index)]
                    gr[scalar_index] = gl[scalar_index]
                    index[0] = d + 1
                    right[tuple(index)] *= -1
        ul, ur = conservative(left, gl), conservative(right, gr)
        fl, fr = ul * left[d + 1], ur * right[d + 1]
        fl[d + 1] += left[-1]
        fr[d + 1] += right[-1]
        fl[-1] += left[d + 1] * left[-1]
        fr[-1] += right[d + 1] * right[-1]
        speed = np.maximum(
            abs(left[d + 1]) + np.sqrt(gl * left[-1] / left[0]),
            abs(right[d + 1]) + np.sqrt(gr * right[-1] / right[0]),
        )
        flux = 0.5 * (fl + fr - speed[None] * (ur - ul))
        weighted = flux * area[None]
        change -= np.diff(weighted, axis=axis + 1) / metrics.volumes[None]
        outward += (
            (
                np.take(weighted, -1, axis=axis + 1)
                - np.take(weighted, 0, axis=axis + 1)
            )
            .reshape(dim + 2, -1)
            .sum(axis=1)
        )
        for j, q in enumerate(qq):
            ql, qr = np.take(q, indices, axis=axis), np.take(
                q, indices + 1, axis=axis
            )
            transport = flux[0] * np.where(flux[0] >= 0, ql, qr) * area
            species[j] -= np.diff(transport, axis=axis) / metrics.volumes
    integrated_source = (
        (source * metrics.volumes[None]).reshape(dim + 2, -1).sum(axis=1)
    )
    return change, species, outward, integrated_source


def advance(state, lam, beta, dt, c, metrics=None):
    metrics = metrics or Metrics(c)
    fractions = [q for q in (lam, beta) if q is not None]
    for first_order in (False, True):
        conserved = [state[0] * q for q in fractions]
        k1, q1, b1, s1 = rhs(state, fractions, c, metrics, first_order)
        stage = state + dt * k1
        if not physical_state(stage):
            continue
        stage_q = [(q + dt * k) / stage[0] for q, k in zip(conserved, q1)]
        if any(np.any((q < -1e-12) | (q > 1 + 1e-12)) for q in stage_q):
            continue
        stage_q = [np.clip(q, 0, 1) for q in stage_q]
        k2, q2, b2, s2 = rhs(stage, stage_q, c, metrics, first_order)
        result = 0.5 * (state + stage + dt * k2)
        if not physical_state(result):
            continue
        final_q = [
            (0.5 * (q + stage[0] * f + dt * k)) / result[0]
            for q, f, k in zip(conserved, stage_q, q2)
        ]
        if any(np.any((q < -1e-12) | (q > 1 + 1e-12)) for q in final_q):
            continue
        final_q = [np.clip(q, 0, 1) for q in final_q]
        return (
            result,
            final_q[0] if final_q else None,
            final_q[1] if len(final_q) > 1 else None,
            0.5 * dt * (b1 + b2),
            0.5 * dt * (s1 + s2),
        )
    raise RuntimeError("Metric/face update lost positivity; reduce CFL")
