from .geometry import (Point, Vector)
from .utils import Immutable

__all__ = ['Mesh', 'Mesh1DBuilder', ]


NODE_TOLERANCE = 1e-4


class Mesh(metaclass=Immutable):
    def __init__(self, nodes, virtual_nodes=(), additional_nodes=()):
        self.real_nodes = tuple(nodes)
        self.virtual_nodes = tuple(virtual_nodes)
        self.additional_nodes = tuple(additional_nodes)


class Mesh1DBuilder:
    def __init__(self, length, start=0.):
        self._length = length
        self._start = start

        self._nodes = []
        self._virtual_nodes = []
        self._additional_nodes = []

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

    def add_middle_nodes(self):
        for i in range(len(self._nodes) - 1):
            p1, p2 = self._nodes[i: i + 2]
            v = Vector(p1, p2)
            p = p1 + v*0.5
            self._additional_nodes.append(p)

        return self

    def create(self):
        return Mesh(
            self._nodes,
            self._virtual_nodes,
            self._additional_nodes
        )
