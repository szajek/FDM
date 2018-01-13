import functools

from fdm.utils import Immutable


COORDS_HASH_ACCURACY = 7
COORDS_EQ_ACCURACY = 1e-8
INFINITY = 9e9


@functools.total_ordering
class Point(metaclass=Immutable):
    def __init__(self, x=0., y=0., z=0.):
        self.x = x
        self.y = y
        self.z = z

        self._hash = self._create_hash()

    def translate(self, vector):
        return Point(*(c + d for c, d in zip(self, vector.components)))

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

    def __gt__(self, other):
        if isinstance(other, Point):
            return Vector(NEGATIVE_UTOPIA, self).length > Vector(NEGATIVE_UTOPIA, other).length
        else:
            raise NotImplementedError

    def __iter__(self):
        return iter([self.x, self.y, self.z])

    def __repr__(self):
        return "Point({},{},{})".format(
            *self
        )


NEGATIVE_UTOPIA = Point(-INFINITY, -INFINITY, -INFINITY)


class Vector(metaclass=Immutable):
    def __init__(self, start, end):
        self.start = start
        self.end = end

        self.components = self._calculate_components()

    def _calculate_components(self):
        return (
            self.end.x - self.start.x,
            self.end.y - self.start.y,
            self.end.z - self.start.z,
        )

    def __neg__(self):
        return Vector(self.end, self.start)

    @property
    def length(self):
        return sum([c**2 for c in self.components])**.5

    def __iter__(self):
        return [self.start, self.end].__iter__()

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.start == other.start and self.end == other.end


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
