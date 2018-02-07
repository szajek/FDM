from .geometry import Point
from .utils import Immutable

__all__ = ['Mesh', 'Mesh1DBuilder', ]


NODE_TOLERANCE = 1e-4


class Mesh(metaclass=Immutable):
    def __init__(self, nodes, virtual_nodes=()):
        self.real_nodes = tuple(nodes)
        self.virtual_nodes = tuple(virtual_nodes)


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
