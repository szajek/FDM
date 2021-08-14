import abc
import collections
import enum
import itertools

import numpy as np
import scipy.linalg

from fdm.equation import create_weights_distributor, Scheme
from fdm.geometry import ClosePointsFinder

np.set_printoptions(suppress=True, linewidth=500, threshold=np.nan)

__all__ = ['solve', 'AnalysisType']


EPS = np.finfo(np.float64).eps


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
        ClosePointsFinder(points, free_points)
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


def create_variables(ordered_nodes):
    return {p: i for i, p in enumerate(ordered_nodes)}


class OrderedNodes(collections.Sequence):
    def __init__(self, mesh):
        self.indices_for_real = []
        self._items = self._create(mesh)

    def _create(self, mesh):
        self.indices_for_real = list(range(len(mesh.real_nodes)))
        return mesh.real_nodes + mesh.virtual_nodes

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


def null_input_modifier(ordered_nodes,  *args):
    return args


def null_spy(tag, item):
    return


class Analyser:
    def __init__(self, equation, input_builders, solver, output_parser,
                 input_modifier=null_input_modifier):

        self._equation = equation
        self._input_builders = input_builders
        self._input_modifier = input_modifier
        self._solver = solver
        self._output_parser = output_parser

        self._spy = null_spy

    def install_spy(self, spy):
        self._spy = spy or null_spy

    def __call__(self, model):
        ordered_nodes = OrderedNodes(model.mesh)
        variables = create_variables(ordered_nodes)
        solver_input = self._input_modifier(
                                 ordered_nodes,
                                 *self._build_input(
                                     self._build_equations(model.template, ordered_nodes),
                                     variables
                                    )
                                )
        self._spy('solver_input', solver_input)
        return self._output_parser(
                self._solver(*solver_input),
                len(model.mesh.real_nodes),
                variables
            )

    def solve(self, model):
        return self.__call__(model)

    def _build_input(self, equations, variables):
        return EquationWriter(*(builder(variables) for builder in self._input_builders)).write(*zip(*equations))

    def _build_equations(self, template, ordered_nodes):
        return [self._equation(*data) for data in expand_template(template, ordered_nodes)]


def linear_system_solver(A, b):
    return np.linalg.solve(A, b[np.newaxis].T)


def linear_system_output_parser(field, variable_number, variables):
    return LinearSystemResults(
        Output(field, variable_number, variables)
    )


def eigenproblem_input_modifier(ordered_nodes, A, B):
    indices = np.ix_(ordered_nodes.indices_for_real, ordered_nodes.indices_for_real)
    return A[indices], B[indices]


def eigenproblem_solver(A, B):
    evals, evects = scipy.linalg.eig(A, b=B)

    idx = evals.argsort()[::-1]
    return evals[idx], evects[:, idx]


def eigenproblem_output_parser(row_output, variable_number, variables):
    def correct_eval(value):
        return -value.real

    def are_increasing(a, b):
        return b - a > 0.

    def invert_to_positive(e):
        s1 = np.sum(e)
        s2 = np.sum(np.negative(e))
        return e if s1 > s2 else np.negative(e)

    def normalize(v):
        extreme = max([np.max(v), abs(np.min(v))])
        return v if abs(extreme) < EPS else v / extreme

    def extract_real_part(value):
        return np.array([v.real for v in value])

    def correct_evec(value):
        corrected = invert_to_positive(normalize(extract_real_part(value)))
        return corrected if are_increasing(*corrected[:2]) else np.negative(corrected)

    evals, evects = row_output

    eigenvectors = [correct_evec(evects[:, i]) for i in range(evects.shape[1])]

    converted = [(correct_eval(eval), Output(evec, variable_number, variables))
                 for eval, evec in zip(evals, eigenvectors) if eval.real not in [np.inf, -np.inf]]

    return EigenproblemResults(*zip(*converted))


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
        input_modifier=eigenproblem_input_modifier,
    ),
}


def solve(_type, model, spy=None):
    solver = _solvers[_type]
    solver.install_spy(spy)
    return solver(model)

