import math
from functools import lru_cache

import numpy as np

from .utils import Immutable
from .geometry import Point, BoundaryBox

__all__ = ['Mesh', 'Mesh1DBuilder', ]


NODE_TOLERANCE = 1e-4


class Mesh(metaclass=Immutable):
    def __init__(self, nodes, virtual_nodes=()):
        self.nodes = tuple(nodes)
        self.virtual_nodes = tuple(virtual_nodes)
        self.all_nodes = list(sorted(self.nodes + self.virtual_nodes))

        self._grid_points = np.array([node.x for node in self.all_nodes])

    @lru_cache(maxsize=1024, typed=False)
    def _position_by_point(self, point):

        indicator = self._grid_points - point.x

        indicator[indicator == 0.] = 1.
        indicator = np.divide(indicator, np.abs(indicator))

        try:
            i = np.where(np.isnan(indicator) | (indicator == 1))[0][0]
        except IndexError:
            raise IndexError("{} is outside the mesh.".format(point))

        x1 = self._grid_points[i - 1]
        x2 = self._grid_points[i]
        dx = x2 - x1
        return i - 1 + (point.x - x1) / dx

    @lru_cache(maxsize=1024, typed=False)
    def distribute_to_points(self, point, value):
        pos = self._position_by_point(point)
        idx = int(pos)
        modulo = math.fmod(pos, 1.)

        n1 = self.all_nodes[idx]

        if modulo < NODE_TOLERANCE:
            return {n1: value}
        else:
            n2 = self.all_nodes[idx + 1]
            _w1, _w2 = (1. - modulo), modulo
            return {n1: _w1*value, n2: _w2*value}

    @property
    def boundary_box(self):
        return BoundaryBox.from_points(self.nodes)


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
