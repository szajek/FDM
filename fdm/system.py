import abc
import collections
import enum
import itertools

import numpy as np
import sys

from .equation import Delta, DynamicLinearEquationTemplate

__all__ = ['LinearEquation', 'solve', 'VirtualValueStrategy']

LinearEquation = collections.namedtuple('LinearEquation', ('coefficients', 'free_value'))


class VirtualValueStrategy(enum.Enum):
    SYMMETRY = 0
    AS_IN_BORDER = 1


def extract_virtual_nodes(equation, domain, strategy):
    nodes_number = len(domain.nodes)
    last_node_idx = (nodes_number - 1)

    def detect_node_location(node_id):
        return -1 if node_id < 0. else 1. if node_id >= nodes_number else 0.

    def find_symmetric_node(node_id, location):
        return abs(node_id) if location == -1 else last_node_idx - (node_id - last_node_idx)

    def find_boundary_node(location):
        return 0 if location == -1 else last_node_idx

    def find_corresponding_node(node_id, location):
        if strategy == VirtualValueStrategy.SYMMETRY:
            return find_symmetric_node(node_id, location)
        elif strategy == VirtualValueStrategy.AS_IN_BORDER:
            return find_boundary_node(location)
        else:
            raise NotImplementedError

    def create_virtual_node_if_needed(node_id):
        location = detect_node_location(node_id)
        if location:
            return VirtualNode(node_id, find_corresponding_node(node_id, location))

    return list(filter(None, [create_virtual_node_if_needed(node_id) for node_id in equation.coefficients.keys()]))


def template_to_equation(template, model, node_address, delta=None):
    delta = Delta.from_connections(*model.domain.get_connections(node_address)) if delta is None else delta
    return LinearEquation(
        template.operator(node_address).to_coefficients(
            delta
        ),
        template.free_value(node_address)
    )


def model_to_equations(model):

    def create_equation(node_address):
        return template_to_equation(model.bcs.get(node_address, model.equation), model, node_address)

    return [create_equation(i) for i, node in enumerate(model.domain.nodes)]


def virtual_nodes_to_equations(virtual_nodes, renumerator, model):  # todo: remove bcs
    def to_equality_equation(vn):
        variable_number = renumerator.get(vn.address)
        return LinearEquation(
            {
                variable_number: 1.,
                vn.corresponding_address: -1.
            },
            0.
        )

    def form_template(template, address):
        return template_to_equation(template, model, address, delta=1.)

    def to_equation(vn):
        if vn.address in model.bcs:
            return form_template(model.bcs[vn.address], vn.address)
        else:
            return to_equality_equation(vn)

    return list(map(to_equation, virtual_nodes))


VirtualNode = collections.namedtuple('VirtualNode', ('address', 'corresponding_address', ))


class EquationWriter:
    def __init__(self, equation, renumerator):
        self._equation = equation
        self._renumerator = renumerator

    def to_coefficients_array(self, size):
        row = self._create_row(size)
        for variable_number, coefficient in self._equation.coefficients.items():
            variable_number = self._renumerator.get(variable_number, variable_number)
            row[int(variable_number)] = coefficient
        return row

    def to_free_value(self):
        return self._equation.free_value

    def _create_row(self, size):
        return np.zeros(size)


class Output(collections.Mapping):

    @property
    def real(self):
        return self._real_output

    def __getitem__(self, key):
        if self._address_forwarder.get(key, key) >= len(self._full_output):
            pass
        return self._full_output[self._address_forwarder.get(key, key)]

    def __iter__(self):
        return self._real_output.__iter__()

    def __len__(self):
        return len(self._real_output)

    def __init__(self, full_output, variable_number, address_forwarder):
        self._full_output = full_output
        self._address_forwarder = address_forwarder

        self._real_output = full_output[:variable_number]
        self._virtual_output = full_output[variable_number:]

    def __repr__(self):
        return "{name}: real: {real}; virtual: {virtual}".format(
            name=self.__class__.__name__, real=self._real_output, virtual=self._virtual_output)


def _solve(solver, model, strategy=VirtualValueStrategy.SYMMETRY):

    def create_virtual_nodes():
        return set(sum([extract_virtual_nodes(equation, model.domain, strategy) for equation in real_equations], []))

    def create_address_forwarder():
        return collections.OrderedDict(
            [(vn.address, real_variables_number + i) for i, vn in enumerate(virtual_nodes)])

    def fill_arrays(weights, free_vector):

        for i, writer in enumerate(create_equation_writers()):
            weights[i] = writer.to_coefficients_array(all_variable_number)
            free_vector[i] = writer.to_free_value()
        return weights, free_vector

    def create_empty_arrays():
        return np.zeros((all_variable_number, all_variable_number)), np.zeros(all_variable_number)

    def create_equation_writers():
        return map(lambda eq: EquationWriter(eq, address_forwarder), all_equations)

    real_equations = model_to_equations(model)
    real_variables_number = len(real_equations)

    virtual_nodes = create_virtual_nodes()
    address_forwarder = create_address_forwarder()
    virtual_nodes_equations = virtual_nodes_to_equations(virtual_nodes, address_forwarder, model)

    all_equations = real_equations + virtual_nodes_equations
    all_variable_number = len(all_equations)

    weights_array, free_vector_array = fill_arrays(*create_empty_arrays())

    return Output(solver(weights_array, free_vector_array), real_variables_number, address_forwarder)


def create_linear_system_of_equations_solver():
    def _solve(A, b):
        np.set_printoptions(suppress=True, linewidth=500, threshold=np.nan)
        return np.linalg.solve(A, b[np.newaxis].T)
    return _solve


def create_eigenproblem_solver():
    def _solve(A, b):
        mass = np.diag(np.ones(b.size))
        matrix = np.dot(A, np.linalg.inv(mass))
        eval, evect = np.linalg.eig(matrix)
        return evect[:, 0]
    return _solve


_solvers = {
    'linear_system_of_equations': create_linear_system_of_equations_solver(),
    'eigenproblem': create_eigenproblem_solver(),
}


def solve(solver, *args, **kwargs):
    return _solve(_solvers[solver], *args, **kwargs)