import collections
import enum

import numpy
import numpy as np
import scipy.linalg
from fdm.analysis.tools import (apply_statics_bc, apply_dynamics_bc)

__all__ = ['AnalysisType', 'Analyser', 'create_linear_system_solver',
           'create_linear_system_solver', 'create_eigenproblem_solver']


np.set_printoptions(suppress=True, linewidth=500, threshold=np.nan)


class AnalysisType(enum.Enum):
    SYSTEM_OF_LINEAR_EQUATIONS = 0
    EIGENPROBLEM = 1


EPS = np.finfo(np.float64).eps


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


LinearSystemResults = collections.namedtuple("LinearSystemResults", ('displacement',))
EigenproblemResults = collections.namedtuple("EigenproblemResults", ('eigenvalues', 'eigenvectors',))


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


def create_linear_system_solver(input_builder, output_modifier=None):
    return Analyser(
        input_builder,
        _solvers['scipy.sparse.linalg.spsolve'],
        linear_system_output_parser,
        apply_statics_bc,
        output_modifier=output_modifier,
        output_verification=linear_system_verification
    )


def create_eigenproblem_solver(input_builder):
    return Analyser(
        input_builder,
        eigenproblem_solver,
        eigenproblem_output_parser,
        apply_dynamics_bc,
        input_modifier=eigenproblem_input_modifier,
    )


class Analyser:
    def __init__(self, input_builder, solver, output_parser, bc_applicator,
                 input_modifier=null_input_modifier, output_modifier=None, output_verification=None):

        self._input_builder = input_builder
        self._input_modifier = input_modifier
        self._output_modifier = output_modifier
        self._bc_applicator = bc_applicator
        self._solver = solver
        self._output_parser = output_parser
        self._output_verification = output_verification

        self._spy = null_spy

    def install_spy(self, spy):
        self._spy = spy or null_spy

    def __call__(self, model):
        ordered_nodes = OrderedNodes(model.mesh)
        variables = create_variables(ordered_nodes)

        A, b = self._input_builder(model, ordered_nodes, variables)

        real_variable_number = len(model.mesh.real_nodes)
        A = A[:real_variable_number, :]
        b = b[:real_variable_number]

        A, b = self._bc_applicator(variables, A, b, model.bcs)

        solver_input = self._input_modifier(ordered_nodes, A, b)
        self._spy('solver_input', solver_input)

        raw_output = self._solver(*solver_input)

        if self._output_verification:
            self._output_verification(raw_output, A, b)

        if self._output_modifier:
            raw_output = self._output_modifier(raw_output, ordered_nodes, variables)

        return self._output_parser(
                raw_output,
                len(model.mesh.real_nodes),
                variables
            )

    def solve(self, model):
        return self.__call__(model)


def linear_system_solver_sparse(A, b):  # with pivoting
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve

    b = b.reshape(-1, 1)
    A = csc_matrix(A, dtype=float)
    B = csc_matrix(b, dtype=float)
    x = spsolve(A, B, permc_spec='MMD_AT_PLUS_A')

    return x.reshape(-1, 1)


def linear_system_solver_lu_factor(A, b):  # with pivoting
    lu, piv = scipy.linalg.lu_factor(A)
    return scipy.linalg.lu_solve((lu, piv), b)


def linear_system_solver_standard(A, b):
    return np.linalg.solve(A, b[np.newaxis].T)


_solvers = {
    'numpy.linarg.solve': linear_system_solver_standard,
    'scipy.linalg.lu_solve': linear_system_solver_lu_factor,
    'scipy.sparse.linalg.spsolve': linear_system_solver_sparse,
}


def linear_system_output_parser(field, variable_number, variables):
    return LinearSystemResults(
        Output(field, variable_number, variables)
    )


def linear_system_verification(output, A, b):
    Ax = numpy.dot(A, output).flatten()
    if not numpy.allclose(Ax, b, atol=1e-5):
        print('!!!! SYSTEM OF EQUATIONS NOT SOLVED !!!!')
        for i, (a, b) in enumerate(zip(Ax, b)):
            print(i, a, b)
        raise ValueError


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


