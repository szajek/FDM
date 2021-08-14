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
            self._hash = self._hash_pool.setdefault(self._get_rounded_coords(), self._create_hash())
        return self._hash

    def _create_hash(self):
        return hash(self._get_rounded_coords())

    def _get_rounded_coords(self):
        return (
            round(self.x, COORDS_HASH_ACCURACY),
            round(self.y, COORDS_HASH_ACCURACY),
            round(self.z, COORDS_HASH_ACCURACY))

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


def detect_dimension(points_coords_array):
    for i, s in enumerate(reversed(np.sum(np.abs(points_coords_array), axis=0))):
        if s > 0.:
            return 3 - i
    else:
        return 1


def points_to_coords_array(points):
    return np.array([list(point) for point in points])


class ClosePointsFinder:
    def __init__(self, base_points, points_to_look_for, tolerance=1e-6):

        assert len(points_to_look_for) > 0, "No 'look for' points are provided."

        self._base_points = base_points
        self._base_points_number = len(base_points)
        self._points_to_look_for = points_to_look_for
        self._tolerance = tolerance

        self._base_points_array = points_to_coords_array(base_points)
        self._look_for_points_array = points_to_coords_array(points_to_look_for)
        self._look_for_point_to_indices = {p: i for i, p in enumerate(points_to_look_for)}

        self._correct_base_point_acc_space_dimension()
        self._compute()

    def _correct_base_point_acc_space_dimension(self):
        space_dimension = detect_dimension(np.vstack((self._base_points_array, self._look_for_points_array)))
        if space_dimension == 1:
            add_points = self._create_virtual_base_points_for_one_dimension()
            add_points_array = np.array([list(point) for point in add_points])
            self._base_points_array = np.vstack((self._base_points_array, add_points_array))
        elif space_dimension == 3:
            raise AttributeError("3D space is not serviced yet.")

    def _create_virtual_base_points_for_one_dimension(self):
        points = list(sorted(self._base_points, key=lambda item: list(item)[0]))
        return [Point((points[i].x + points[i+1].x)/2., 1.) for i in range(len(points)-1)]

    def _compute(self):

        self._triangle = scipy.spatial.Delaunay(self._base_points_array[:, :2])
        self._simplices = self._triangle.find_simplex(self._look_for_points_array[:, :2], tol=self._tolerance)
        self._distances = scipy.spatial.distance.cdist(self._look_for_points_array, self._base_points_array)

    def _find(self, point):
        look_for_idx = self._look_for_point_to_indices[point]

        simplex_number = self._simplices[look_for_idx]
        assert simplex_number != -1, "Simplex has not been found for {}".format(str(point))

        indices = self._triangle.simplices[simplex_number]
        return {self._base_points[base_idx]: self._distances[look_for_idx][base_idx] for base_idx in indices
                if base_idx < self._base_points_number}

    def __call__(self, point):
        return self._find(point)
