import math

import numpy as np
import scipy.spatial

from .geometry import Point
from .utils import Immutable

__all__ = ['Mesh', 'Mesh1DBuilder', ]


NODE_TOLERANCE = 1e-4


class Mesh(metaclass=Immutable):
    def __init__(self, nodes, virtual_nodes=()):
        self.nodes = tuple(nodes)
        self.virtual_nodes = tuple(virtual_nodes)  # todo: introduce Node class with 'is_virtual' method; refactor fdm.equation.model_to_equation
        self.all_nodes = list(sorted(self.nodes + self.virtual_nodes, key=lambda item: item.x))


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


def create_weights_distributor(indexed_points):

    def distribute(point, value):
        pos = indexed_points.get_index(point)
        idx = int(pos)
        modulo = math.fmod(pos, 1.)

        n1 = indexed_points.get_point(idx)

        if modulo < NODE_TOLERANCE:
            return {n1: value}
        elif abs(1. - modulo) < NODE_TOLERANCE:
            n2 = indexed_points.get_point(idx + 1)
            return {n2: value}
        else:
            n2 = indexed_points.get_point(idx + 1)
            _w1, _w2 = (1. - modulo), modulo
            return {n1: _w1*value, n2: _w2*value}
    return distribute


class Mesh1DBuilder:
    def __init__(self, length, start=0.):
        self._length = length
        self._start = start

        self._nodes = []
        self._virtual_nodes = []

    def add_uniformly_distributed_nodes(self, number):
        if number < 2:
            raise AttributeError("Number of point must be at least 2")
        section_length = self._length / (number - 1)

        for node_num in range(number):
            self.add_node_by_coordinate(self._start + node_num*section_length)

        return self

    def add_node_by_coordinate(self, coord):
        node = Point(coord)
        self._nodes.append(node)
        return node

    def add_virtual_nodes(self, *coords):
        for c in coords:
            self._virtual_nodes.append(Point(c))
        return self

    def create(self):
        return Mesh(
            self._nodes,
            self._virtual_nodes
        )
