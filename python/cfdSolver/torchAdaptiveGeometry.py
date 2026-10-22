"""Balanced dyadic leaf meshes and exact orthogonal box metrics."""

from itertools import product
import numpy as np


def settings(**overrides):
    result = dict(
        max_level=2,
        min_level=0,
        refine_threshold=0.08,
        coarsen_threshold=0.02,
        regrid_every=3,
        max_cells=100000,
        order=2,
    )
    if set(overrides) - set(result):
        raise ValueError("Unknown adaptivity option")
    result.update(overrides)
    for key in (
        "max_level",
        "min_level",
        "regrid_every",
        "max_cells",
        "order",
    ):
        value = result[key]
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError("AMR counts and levels must be integers")
        result[key] = int(value)
    if not (
        0 <= result["min_level"] <= result["max_level"] <= 8
        and result["regrid_every"] >= 1
        and 2 <= result["max_cells"] <= 2_000_000
        and result["order"] in (1, 2)
    ):
        raise ValueError("Invalid AMR levels, interval, capacity or order")
    low, high = result["coarsen_threshold"], result["refine_threshold"]
    if not np.isfinite([low, high]).all() or not 0 <= low < high <= 1:
        raise ValueError("Require 0 <= coarsen threshold < refine <= 1")
    return result


def children(key):
    level, *index = key
    return [
        (level + 1, *(2 * i + b for i, b in zip(index, bits[::-1])))
        for bits in product((0, 1), repeat=len(index))
    ]


def canonical(keys):
    keys = np.asarray(keys, dtype=np.int64)
    return keys[
        np.lexsort(
            tuple(keys[:, k] for k in range(1, keys.shape[1])) + (keys[:, 0],)
        )
    ]


def volume(c, low, high):
    d, g = c["dimension"], c.get("geometry", "cartesian")
    width = high - low
    if g == "cartesian":
        return np.prod(width, axis=1)
    shell2 = (high[:, 0] ** 2 - low[:, 0] ** 2) / 2
    if g == "spherical":
        shell3 = (high[:, 0] ** 3 - low[:, 0] ** 3) / 3
        if d == 1:
            return 4 * np.pi * shell3
        result = shell3 * (np.cos(low[:, 1]) - np.cos(high[:, 1]))
        return result * (width[:, 2] if d == 3 else 2 * np.pi)
    if d == 1:
        return 2 * np.pi * shell2
    if g == "cylindrical" and d == 2:
        return 2 * np.pi * shell2 * width[:, 1]
    return shell2 * np.prod(width[:, 1:], axis=1)


def area(c, direction, position, low, high):
    d, g = c["dimension"], c.get("geometry", "cartesian")
    width = high - low
    if g == "cartesian":
        return np.prod(np.delete(width, direction, axis=1), axis=1)
    r0, r1 = low[:, 0], high[:, 0]
    shell2 = (r1**2 - r0**2) / 2
    if d == 1:
        return (
            4 * np.pi * position**2
            if g == "spherical"
            else 2 * np.pi * position
        )
    if g == "spherical":
        azimuth = width[:, 2] if d == 3 else 2 * np.pi
        if direction == 0:
            return (
                position**2
                * (np.cos(low[:, 1]) - np.cos(high[:, 1]))
                * azimuth
            )
        if direction == 1:
            return shell2 * np.sin(position) * azimuth
        return shell2 * width[:, 1]
    if g == "cylindrical" and d == 2:
        return (
            2 * np.pi * position * width[:, 1]
            if direction == 0
            else 2 * np.pi * shell2
        )
    if direction == 0:
        return position * np.prod(width[:, 1:], axis=1)
    if direction == 1:
        return width[:, 0] * (width[:, 2] if d == 3 else 1)
    return shell2 * width[:, 1]


class LeafMesh:
    """A forest of quad/octrees, with one shared flux per subface."""

    def __init__(self, config, keys=None, validate=True):
        self.config = config
        self.dimension = d = config["dimension"]
        self.options = settings(**config["amr"])
        self.max_level = self.options["max_level"]
        counts = np.array(config["counts_xyz"], dtype=np.int64)
        if np.prod(counts) > self.options["max_cells"]:
            raise ValueError("AMR capacity is smaller than the base mesh")
        if (
            np.prod(counts) * 2 ** (d * self.options["min_level"])
            > self.options["max_cells"]
        ):
            raise ValueError("AMR capacity cannot cover the minimum level")
        if keys is None:
            indices = np.indices(tuple(counts[::-1])).reshape(d, -1)[::-1].T
            keys = np.column_stack(
                (np.zeros(len(indices), dtype=int), indices)
            )
        keys = np.asarray(keys)
        if (
            keys.ndim != 2
            or keys.shape[1] != d + 1
            or keys.dtype.kind not in "iu"
            or len(keys) == 0
            or len(keys) > self.options["max_cells"]
        ):
            raise ValueError("Invalid adaptive leaf keys/count")
        self.keys = canonical(keys)
        self.levels = self.keys[:, 0]
        if np.any((self.levels < 0) | (self.levels > self.max_level)):
            raise ValueError("Leaf level is outside configured bounds")
        if np.any(self.keys[:, 1:] < 0) or np.any(
            self.keys[:, 1:] >= counts * (2 ** self.levels[:, None])
        ):
            raise ValueError("Leaf indices are outside the domain")
        self.lookup = {tuple(key): i for i, key in enumerate(self.keys)}
        if len(self.lookup) != len(self.keys):
            raise ValueError("Duplicate leaf keys")
        if validate:
            remaining = set(self.lookup)
            for level in range(self.max_level, 0, -1):
                parents = {
                    (level - 1, *(v // 2 for v in key[1:]))
                    for key in remaining
                    if key[0] == level
                }
                for parent in parents:
                    group = set(children(parent))
                    if group <= remaining:
                        if parent in remaining:
                            raise ValueError("Overlapping adaptive cells")
                        remaining.difference_update(group)
                        remaining.add(parent)
            roots = {
                tuple(key)
                for key in np.column_stack(
                    (
                        np.zeros(int(np.prod(counts)), dtype=int),
                        np.indices(tuple(counts[::-1])).reshape(d, -1)[::-1].T,
                    )
                )
            }
            if remaining != roots:
                raise ValueError(
                    "Adaptive leaves overlap or leave domain holes"
                )
        scale = 2 ** (self.max_level - self.levels)
        self.integer_low = self.keys[:, 1:] * scale[:, None]
        self.integer_high = self.integer_low + scale[:, None]
        self.total = counts * 2**self.max_level
        self.unit = np.array(config["lengths_xyz"]) / self.total
        self.origin = np.array(config["origins_xyz"])
        self.low = self.origin + self.integer_low * self.unit
        self.high = self.origin + self.integer_high * self.unit
        self.centers = 0.5 * (self.low + self.high)
        self.widths = self.high - self.low
        self.volumes = volume(config, self.low, self.high)
        if not np.isfinite(self.volumes).all() or np.any(self.volumes <= 0):
            raise ValueError("Nonpositive adaptive volumes")
        self.divergence = np.stack(
            [
                (
                    area(config, k, self.high[:, k], self.low, self.high)
                    - area(config, k, self.low[:, k], self.low, self.high)
                )
                / self.volumes
                for k in range(d)
            ]
        )
        self._topology()

    def _find(self, point):
        for level in range(self.max_level + 1):
            size = 2 ** (self.max_level - level)
            index = tuple(np.floor(point / size).astype(int))
            found = self.lookup.get((level, *index))
            if found is not None:
                return found
        raise ValueError("Adaptive mesh has an uncovered neighbor")

    def _topology(self):
        c, d = self.config, self.dimension
        periodic = [
            c["face_boundaries"][2 * k]["type"] == "periodic" for k in range(d)
        ]
        left, right, directions, lows, highs, positions, sides = (
            [] for _ in range(7)
        )

        def emit(i, j, k, lo, hi, position, side=0):
            left.append(i)
            right.append(j)
            directions.append(k)
            lows.append(lo.copy())
            highs.append(hi.copy())
            positions.append(position)
            sides.append(side)

        def match(i, k, lo, hi, position):
            point = 0.5 * (lo + hi)
            point[k] = (position + 0.5) % self.total[k]
            j = self._find(point)
            if np.all(
                self.integer_low[j, np.arange(d) != k] <= lo[np.arange(d) != k]
            ) and np.all(
                self.integer_high[j, np.arange(d) != k]
                >= hi[np.arange(d) != k]
            ):
                emit(i, j, k, lo, hi, position)
                return
            transverse = [a for a in range(d) if a != k]
            middle = (lo + hi) // 2
            for bits in product((0, 1), repeat=len(transverse)):
                lower, upper = lo.copy(), hi.copy()
                for a, bit in zip(transverse, bits):
                    if bit:
                        lower[a] = middle[a]
                    else:
                        upper[a] = middle[a]
                match(i, k, lower, upper, position)

        for i in range(len(self.keys)):
            for k in range(d):
                lo, hi = (
                    self.integer_low[i].copy(),
                    self.integer_high[i].copy(),
                )
                position = hi[k]
                if position == self.total[k] and not periodic[k]:
                    emit(i, -1, k, lo, hi, position, 1)
                else:
                    match(i, k, lo, hi, position)
                if lo[k] == 0 and not periodic[k]:
                    emit(-1, i, k, lo, hi, 0, -1)
        self.left, self.right = np.array(left), np.array(right)
        self.directions, self.sides = np.array(directions), np.array(sides)
        self.face_low = self.origin + np.array(lows) * self.unit
        self.face_high = self.origin + np.array(highs) * self.unit
        self.face_positions = (
            self.origin[self.directions]
            + np.array(positions) * self.unit[self.directions]
        )
        self.face_areas = np.empty(len(left))
        face_center = 0.5 * (self.face_low + self.face_high)
        face_center[np.arange(len(left)), self.directions] = (
            self.face_positions
        )
        self.offset_left = face_center - self.centers[np.maximum(self.left, 0)]
        self.offset_right = (
            face_center - self.centers[np.maximum(self.right, 0)]
        )
        for k in range(d):
            mask = self.directions == k
            self.face_areas[mask] = area(
                c,
                k,
                self.face_positions[mask],
                self.face_low[mask],
                self.face_high[mask],
            )
            valid = mask & (self.left >= 0)
            self.offset_left[valid, k] = 0.5 * self.widths[self.left[valid], k]
            valid = mask & (self.right >= 0)
            self.offset_right[valid, k] = (
                -0.5 * self.widths[self.right[valid], k]
            )
        self.interior = (self.left >= 0) & (self.right >= 0)
        if np.any(
            abs(
                self.levels[self.left[self.interior]]
                - self.levels[self.right[self.interior]]
            )
            > 1
        ):
            raise ValueError("Adaptive mesh must be face-balanced 2:1")
        self.displacement = self.offset_left - self.offset_right
        lower = self.left < 0
        upper = self.right < 0
        self.displacement[lower] = -2 * self.offset_right[lower]
        self.displacement[upper] = 2 * self.offset_left[upper]
        self._gradient_matrix()

    def _gradient_matrix(self):
        d = self.dimension
        delta = self.displacement
        self.weights = 1 / np.sum(delta**2, axis=1)
        matrix = np.zeros((len(self.keys), d, d))
        contribution = (
            self.weights[:, None, None] * delta[:, :, None] * delta[:, None, :]
        )
        for indices in (self.left, self.right):
            mask = indices >= 0
            np.add.at(matrix, indices[mask], contribution[mask])
        self.inverse = np.linalg.inv(matrix)

    def gradients(self, values, left, right, bounded=True):
        """Least-squares slopes with a cellwise extrema limiter."""
        d = self.dimension
        difference = right - left
        vector = np.zeros((len(self.keys), d, values.shape[0]))
        contribution = (
            self.weights[:, None, None]
            * self.displacement[:, :, None]
            * difference.T[:, None, :]
        )
        low, high = values.T.copy(), values.T.copy()
        for indices, neighbor in ((self.left, right), (self.right, left)):
            mask = indices >= 0
            np.add.at(vector, indices[mask], contribution[mask])
            np.minimum.at(low, indices[mask], neighbor.T[mask])
            np.maximum.at(high, indices[mask], neighbor.T[mask])
        gradient = np.einsum("nij,njk->nik", self.inverse, vector)
        if bounded:
            theta = np.ones_like(values.T)
            for indices, offset in (
                (self.left, self.offset_left),
                (self.right, self.offset_right),
            ):
                mask = indices >= 0
                ids = indices[mask]
                change = np.einsum("nd,ndv->nv", offset[mask], gradient[ids])
                allowed = np.where(
                    change >= 0,
                    high[ids] - values.T[ids],
                    low[ids] - values.T[ids],
                )
                ratio = np.ones_like(change)
                np.divide(
                    allowed, change, out=ratio, where=abs(change) > 1e-14
                )
                np.minimum.at(theta, ids, np.clip(ratio, 0, 1))
            gradient *= theta[:, None, :]
        return gradient
