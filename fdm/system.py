import collections

import numpy as np

from fdm.equation import LinearEquation

np.set_printoptions(suppress=True, linewidth=500, threshold=np.nan)


__all__ = ['solve']


def model_to_equations(model):

    def from_template(point, template):
        template = model.bcs.get(point, template)
        assert template is not None, "No template found for: {}".format(point)
        return LinearEquation(
                template.operator(point).to_mesh(model.mesh),
                template.free_value(point)
                )

    def generate_for(nodes, template=None):
        return [from_template(node, template) for node in nodes]

    return generate_for(model.mesh.nodes, model.equation) + generate_for(model.mesh.virtual_nodes)


class EquationWriter:
    def __init__(self, equation, mapper):
        self._equation = equation
        self._mapper = mapper

    def to_coefficients_array(self, size):
        row = self._create_row(size)
        for point, weight in self._equation.scheme.items():
            if point not in self._mapper:
                raise AttributeError("No point in mapper found: %s" % str(point))
            row[self._mapper.get(point)] = weight
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
        if self._point_to_variable_mapper.get(key, key) >= len(self._full_output):
            pass
        return self._full_output[self._point_to_variable_mapper.get(key, key)]

    def __iter__(self):
        return self._real_output.__iter__()

    def __len__(self):
        return len(self._real_output)

    def __init__(self, full_output, variable_number, point_to_variable_mapper):
        self._full_output = full_output
        self._point_to_variable_mapper = point_to_variable_mapper

        self._real_output = full_output[:variable_number]
        self._virtual_output = full_output[variable_number:]

    def __repr__(self):
        return "{name}: real: {real}; virtual: {virtual}".format(
            name=self.__class__.__name__, real=self._real_output, virtual=self._virtual_output)


def create_variables(domain):
    return {p: i for i, p in enumerate(domain.nodes + domain.virtual_nodes)}


def _solve(solver, model):

    def create_empty_arrays():
        return np.zeros((variables_number, variables_number)), np.zeros(variables_number)

    def create_equation_writers():
        return map(lambda eq: EquationWriter(eq, variables), equations)

    def fill_arrays(weights, free_vector):
        for i, writer in enumerate(create_equation_writers()):
            weights[i] = writer.to_coefficients_array(variables_number)
            free_vector[i] = writer.to_free_value()
        return weights, free_vector

    variables = create_variables(model.mesh)
    variables_number = len(variables)

    equations = model_to_equations(model)

    return Output(
        solver(*fill_arrays(*create_empty_arrays())),
        len(model.mesh.nodes),
        variables
    )


def create_linear_system_of_equations_solver():
    def _solve(A, b):
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