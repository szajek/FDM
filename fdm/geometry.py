import math

import numpy as np
import scipy.spatial

COORDS_HASH_ACCURACY = 7
COORDS_EQ_ACCURACY = 1e-8
INFINITY = 9e9


def calculate_length(components):
    return sum([c ** 2 for c in components]) ** .5


def calculate_points_delta(point_1, point_2):
    return tuple(c_1 - c_2 for c_1, c_2 in zip(point_1, point_2))


def calculate_distance(start, end):
    return calculate_length(calculate_points_delta(start, end))


class Point:
    __slots__ = 'x', 'y', 'z', '_hash'

    _hash_pool = {}

    def __init__(self, x=0., y=0., z=0.):
        self.x = x
        self.y = y
        self.z = z

        self._hash = None

    def translate(self, vector):
        dx, dy, dz = vector.components
        return Point(self.x + dx, self.y + dy, self.z + dz)

    def __add__(self, other):
        if isinstance(other, Vector):
            return self.translate(other)
        else:
            raise NotImplementedError

    def __iadd__(self, other):
        return self.__add__(self, other)

    def __sub__(self, other):
        if isinstance(other, Vector):
            return self.translate(-other)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Point(*(c*other for c in self))
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        return self.__mul__(self, other)

    def __neg__(self):
        return Point(*(-c for c in self))

    def __hash__(self):
        if self._hash is None:
            self._hash = self._hash_pool.setdefault(tuple(self), self._create_hash())
        return self._hash

    def _create_hash(self):
        return hash(
            (
                round(self.x, COORDS_HASH_ACCURACY),
                round(self.y, COORDS_HASH_ACCURACY),
                round(self.z, COORDS_HASH_ACCURACY))
        )

    def __eq__(self, other):
        if isinstance(other, Point):
            return all([abs(c1 - c2) < COORDS_EQ_ACCURACY for c1, c2 in zip(self, other)])
        else:
            return False

    def __iter__(self):
        return iter([self.x, self.y, self.z])

    def __repr__(self):
        return "Point({},{},{})".format(
            *self
        )


class Vector:
    __slots__ = 'start', 'end', '_components'

    def __init__(self, start, end):
        self.start = start
        self.end = end

        self._components = None

    def _calculate_components(self):
        return calculate_points_delta(self.end, self.start)

    def __neg__(self):
        return Vector(self.end, self.start)

    @property
    def components(self):
        if self._components is None:
            self._components = self._calculate_components()
        return self._components

    @property
    def length(self):
        return calculate_length(self.components)

    def __iter__(self):
        return [self.start, self.end].__iter__()

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.start == other.start and self.end == other.end


NEGATIVE_UTOPIA_COORDINATES = -INFINITY, -INFINITY, -INFINITY
NEGATIVE_UTOPIA = Point(*NEGATIVE_UTOPIA_COORDINATES)


class FreeVector(Vector):
    def __init__(self, point):
        Vector.__init__(self, Point(), point)


class BoundaryBox:
    def __init__(self, _min, _max):
        self.min = _min
        self.max = _max

    @property
    def dimensions(self):
        return tuple(map(self._calculate_dimension, range(len(self.min))))

    def _calculate_dimension(self, direction):
        return self.max[direction] - self.min[direction]

    @classmethod
    def from_points(cls, points):
        _min, _max = calculate_extreme_coordinates(points)
        return cls(_min, _max)


def calculate_extreme_coordinates(points):
    return tuple(zip(*[(min(coords), max(coords)) for coords in zip(*map(list, points))]))


class IndexedPoints:
    def __init__(self, base_points, points_to_look_for):
        self._base_points = base_points
        self._points_to_look_for = points_to_look_for

        self._points_to_look_for_indices = dict(((point, i) for i, point in enumerate(points_to_look_for)))

        self._base_points_array = np.array([list(point) for point in self._base_points])
        self._look_for_points_array = np.array([list(point) for point in self._points_to_look_for])

        self._find_points_indices()

    def _find_points_indices(self):

        ckd_tree = scipy.spatial.cKDTree(self._base_points_array)
        distances, close_points_indexes = ckd_tree.query(self._look_for_points_array, k=2)

        self._indices = [
            self._find_point_index(i, close_points_indexes[i][0])
            for i in range(len(self._points_to_look_for))
        ]

    def _find_point_index(self, i, the_closest_point):
        x = self._look_for_points_array[i][0]
        idx = the_closest_point

        bind_with_left = idx == len(self._base_points_array) - 1 or x < self._base_points_array[idx][0]
        i1, i2 = (idx - 1, idx) if bind_with_left else (idx, idx + 1)
        d1, d2 = abs(self._base_points_array[[i1, i2], 0] - [x, x])

        return i1 + (d1 / (d1 + d2))

    def get_index(self, point):
        return self._indices[self._points_to_look_for_indices[point]]

    def get_point(self, index):
        assert math.fmod(index, 1) == 0, "Point index must be integer"
        return self._base_points[index]