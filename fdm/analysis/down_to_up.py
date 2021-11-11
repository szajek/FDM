import numpy

from fdm.geometry import ClosePointsFinder
from fdm.analysis.utils import create_weights_distributor


def get_solvers():
    return {}


class ArrayBuilder(object):
    def __init__(self, variables):
        self._variables = variables
        self._points = tuple(variables.keys())
        self._size = len(variables)

        self._array = numpy.identity(self._size)

    def apply(self, element):
        modifier = self._create_modifier(element)
        self._array = output = numpy.dot(modifier, self._array)
        return output

    def _create_modifier(self, element):
        modifier = numpy.zeros((self._size, self._size))
        for point in self._points:
            idx = self._variables[point]
            scheme = self._expand_scheme_for(point, element)
            for p in scheme:
                modifier[idx, self._variables[p]] = scheme[p]
        return modifier

    def _expand_scheme_for(self, point, element):
        scheme = element.expand(point)
        if len(scheme):
            scheme = distribute_scheme_to_nodes(self._points, scheme)
        return scheme

    def get(self):
        return self._array


def distribute_scheme_to_nodes(nodes, scheme):
    free_points = tuple(scheme)
    distributor = create_weights_distributor(
        ClosePointsFinder(nodes, free_points)
    )
    return scheme.distribute(distributor)
