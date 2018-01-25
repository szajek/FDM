import abc
import collections
import enum
import itertools

import numpy as np

from fdm.equation import create_weights_distributor, Scheme
from fdm.geometry import IndexedPoints

np.set_printoptions(suppress=True, linewidth=500, threshold=np.nan)

__all__ = ['solve', 'AnalysisType']


class AnalysisType(enum.Enum):
    SYSTEM_OF_LINEAR_EQUATIONS = 0
    EIGENPROBLEM = 1


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


class EquationWriter:
    def __init__(self, *writers):
        self._writers = writers

    def write(self, *items):
        return [writer.write(*items) for writer, items in zip(self._writers, items)]


class Output(collections.Mapping):
    def __init__(self, full_output, variable_number, point_to_variable_mapper):
        self._full_output = full_output
        self._point_to_variable_mapper = point_to_variable_mapper

        self._real_output = full_output[:variable_number]
        self._virtual_output = full_output[variable_number:]

    @property
    def real(self):
        return self._real_output

    def __getitem__(self, key):
        if self._point_to_variable_mapper.get(key, key) >= len(self._full_output):
            pass
        return self._full_output[self._point_to_variable_mapper.get(key, key)]

    def __iter__(self):
        return self._real_output.__iter__()

    def __len__(self):
        return len(self._real_output)

    def __repr__(self):
        return "{name}: real: {real}; virtual: {virtual}".format(
            name=self.__class__.__name__, real=self._real_output, virtual=self._virtual_output)


LinearSystemEquation = collections.namedtuple('LinearEquation', ('scheme', 'free_value'))
EigenproblemEquation = collections.namedtuple('EigenproblemEquation', ('scheme_A', 'scheme_B'))


LinearSystemResults = collections.namedtuple("LinearSystemResults", ('displacement',))
EigenproblemResults = collections.namedtuple("EigenproblemResults", ('eigenvalues', 'eigenvectors',))


def expand_template(template, points):
    return _map_data_to_points(
        points,
        [template.expand(point) for point in points]
    )


def _map_data_to_points(points, expanded_data):
    free_points = _extract_points_from_data(expanded_data)

    distributor = create_weights_distributor(
        IndexedPoints(points, free_points)
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


def create_variables(mesh):
    return {p: i for i, p in enumerate(mesh.real_nodes + mesh.virtual_nodes)}


class Analyser:
    def __init__(self, equation, input_builders, solver, output_parser):
        self._equation = equation
        self._input_builders = input_builders
        self._solver = solver
        self._output_parser = output_parser

    def solve(self, model):
        variables = create_variables(model.mesh)
        return self._output_parser(
                self._solver(*
                             self._build_input(
                                 self._build_equations(model),
                                 variables
                             )
                             ),
                len(model.mesh.real_nodes),
                variables
            )

    def _build_input(self, equations, variables):
        return EquationWriter(*(builder(variables) for builder in self._input_builders)).write(*zip(*equations))

    def _build_equations(self, model):
        return [self._equation(*data) for data in expand_template(model.template, model.mesh.all_nodes)]


def linear_system_solver(A, b):
    return np.linalg.solve(A, b[np.newaxis].T)


def linear_system_output_parser(field, variable_number, variables):
    return LinearSystemResults(
        Output(field, variable_number, variables)
    )


def eigenproblem_solver(A, B):
    matrix = np.dot(B, np.linalg.inv(A))
    return np.linalg.eig(matrix)


def eigenproblem_output_parser(row_output, variable_number, variables):
    def correct_eval(value):
        return -1./value

    evals, evects = row_output
    return EigenproblemResults(
        [correct_eval(eval) for eval in evals],
        [Output(evects[:, i], variable_number, variables)
         for i in range(evects.shape[1])]
    )


_solvers = {
    AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS: Analyser(
        LinearSystemEquation,
        (SchemeWriter, FreeValueWriter),
        linear_system_solver,
        linear_system_output_parser,
    ),
    AnalysisType.EIGENPROBLEM: Analyser(
        EigenproblemEquation,
        (SchemeWriter, SchemeWriter),
        eigenproblem_solver,
        eigenproblem_output_parser,
    ),
}


def solve(solver, model):
    return _solvers[solver].solve(model)
