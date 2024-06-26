import abc
import collections
import itertools

import numpy as np

from fdm import Scheme
from fdm.analysis.analyzer import (
    AnalysisType, create_linear_system_solver, create_eigenproblem_solver
)
from fdm.analysis.tools import create_weights_distributor
from fdm.geometry import create_close_point_finder

LinearSystemEquation = collections.namedtuple(
    'LinearEquation', ('scheme', 'free_value')
)
EigenproblemEquation = collections.namedtuple(
    'EigenproblemEquation', ('scheme_A', 'scheme_B')
)


def get_solvers():
    return {
        AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS: create_linear_system_solver(
            input_builder(LinearSystemEquation, (SchemeWriter, FreeValueWriter))
        ),
        AnalysisType.EIGENPROBLEM: create_eigenproblem_solver(
            input_builder(EigenproblemEquation, (SchemeWriter, SchemeWriter)),
        )}


def input_builder(equation, scheme_writers):
    def build(model, ordered_nodes, variables):
        equations = [equation(*data) for data in expand_template(model.template, model.mesh.real_nodes, ordered_nodes)]
        writer = EquationWriter(*(builder(variables) for builder in scheme_writers))
        A, b = writer.write(*zip(*equations))
        return A, b

    return build


class EquationWriter:
    def __init__(self, *writers):
        self._writers = writers

    def write(self, *items):
        return [writer.write(*items) for writer, items in zip(self._writers, items)]


class Writer(metaclass=abc.ABCMeta):
    def __init__(self, variables):
        self._variables = variables

        self._size = len(variables)
        self._array = self._create_array()
        self._counter = 0

    def write(self, *schemes):
        list(map(self._write_next, schemes))
        return self._array

    @abc.abstractmethod
    def _write_next(self, item):
        raise NotImplementedError

    @abc.abstractmethod
    def _create_array(self):
        raise NotImplementedError


class SchemeWriter(Writer):
    def _create_array(self):
        return np.zeros((self._size, self._size))

    def _write_next(self, scheme):
        for point, weight in scheme.items():
            if point not in self._variables:
                raise AttributeError("No point in mapper found: %s" % str(point))
            self._array[self._counter, self._variables.get(point)] = weight
        self._counter += 1


class FreeValueWriter(Writer):
    def _create_array(self):
        return np.zeros(self._size)

    def _write_next(self, value):
        self._array[self._counter] = value
        self._counter += 1


def expand_template(template, for_points, all_points):
    return _map_data_to_points(
        all_points,
        [template.expand(point) for point in for_points]
    )


def _map_data_to_points(points, expanded_data):
    distributor = create_weights_distributor(
        create_close_point_finder(points)
    )

    def distribute(item):
        return item.distribute(distributor) if isinstance(item, Scheme) else item

    return [[distribute(item) for item in items]
            for items in expanded_data
            ]


def _extract_points_from_data(expanded_data):
    def _extract_schemes(items):
        return [item for item in items if isinstance(item, Scheme)]

    return list(
        set(
            itertools.chain(
                *[scheme.keys() for scheme in
                  itertools.chain(*[_extract_schemes(items) for items in expanded_data])]
            )
        )
    )
