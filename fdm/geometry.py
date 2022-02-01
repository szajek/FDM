import math

import numpy
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
    __slots__ = 'x', 'y', 'z', '_coords', '_hash', 'index'

    _hash_pool = {}

    def __init__(self, x=0., y=0., z=0., index=None):
        self.x, self.y, self.z = self._coords = (x, y, z)

        self._hash = None
        self.index = index

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
            self._hash = self._hash_pool.setdefault(self._get_rounded_coords(), self._create_hash())
        return self._hash

    def _create_hash(self):
        return hash(self._get_rounded_coords())

    def _get_rounded_coords(self):
        return (
            round(self.x, COORDS_HASH_ACCURACY) if self.x != -1. else 'one',
            round(self.y, COORDS_HASH_ACCURACY) if self.y != -1. else 'one',
            round(self.z, COORDS_HASH_ACCURACY) if self.z != -1. else 'one'
        )

    def __eq__(self, other):
        if isinstance(other, Point):
            return hash(self) == hash(other)
        else:
            return False

    def __iter__(self):
        return iter(self._coords)

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

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector(other*self.start, other*self.end)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        return self.__mul__(self, other)

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


class PointBeyondDomainException(Exception):
    pass


def create_close_point_finder(base_points, tolerance=1e-6):
    dim = detect_dimension(points_to_coords_array(base_points))
    finder = _close_point_finders[dim](base_points, tolerance)
    return CachedClosePointFinder1d(finder)


class ClosePointsFinder2d:
    def __init__(self, base_points, tolerance=1e-6):
        self._base_points = base_points
        self._base_points_number = len(base_points)
        self._tolerance = tolerance

        self._base_points_array = points_to_coords_array(base_points)

        self._triangle = scipy.spatial.Delaunay(self._base_points_array[:, :2])

    def __call__(self, point):
        return self._find(point)

    def _find(self, point):
        x, y, z = point
        simplex_number = self._triangle.find_simplex(np.array([x, y]), tol=self._tolerance)
        if simplex_number == -1:
            raise PointBeyondDomainException("Simplex has not been found for {}".format(str(point)))

        indices = self._triangle.simplices[simplex_number]

        return {self._base_points[base_idx]: calculate_distance(self._base_points[base_idx], point)
                for base_idx in indices
                if base_idx < self._base_points_number}


class CachedClosePointFinder1d(object):
    def __init__(self, finder):
        self._finder = finder

        self._cache = {}

    def __call__(self, point):
        try:
            return self._cache[point]
        except KeyError:
            return self._cache.setdefault(point, self._finder(point))


class ClosePointsFinder1d(object):
    def __init__(self, unsorted_base_points, tolerance=1e-6):
        sorted_xs, sorted_base_points = self._sort_points(unsorted_base_points)
        self._base_points = sorted_base_points
        self._tolerance = tolerance

        self._base_xs = np.array(sorted_xs)

    @staticmethod
    def _sort_points(unsorted_base_points):
        unsorted_xs = np.array([p.x for p in unsorted_base_points])
        return tuple(zip(*sorted(zip(unsorted_xs, unsorted_base_points))))

    def __call__(self, point):
        x = point.x
        point_1_idx = np.argmin(np.abs(self._base_xs - x))
        point_1_x = self._base_xs[point_1_idx]
        point_2_idx = self._find_second_point(point_1_idx, point_1_x, x)
        point_2_x = self._base_xs[point_2_idx]
        return {
            self._base_points[point_1_idx]: abs(x - point_1_x),
            self._base_points[point_2_idx]: abs(x - point_2_x),
        }

    def _find_second_point(self, closest_point_idx, closest_point_x, x):
        n = len(self._base_points) - 1
        if closest_point_idx == 0:
            if x - closest_point_x < -self._tolerance:
                raise PointBeyondDomainException("Point x={} beyond domain".format(str(x)))
            else:
                return 1
        elif closest_point_idx == n:
            if x - closest_point_x > self._tolerance:
                raise PointBeyondDomainException("Point x={} beyond domain".format(str(x)))
            else:
                return n - 1
        else:
            if x - closest_point_x >= 0.:
                return closest_point_idx + 1
            else:
                return closest_point_idx - 1


_close_point_finders = {
    1: ClosePointsFinder1d,
    2: ClosePointsFinder2d,
}


def detect_dimension(points_coords_array):
    for i, s in enumerate(reversed(np.sum(np.abs(points_coords_array), axis=0))):
        if s > 0.:
            return 3 - i
    else:
        return 1


def points_to_coords_array(points):
    return np.array([list(point) for point in points])

