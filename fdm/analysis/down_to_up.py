import abc

import numpy

from fdm.geometry import ClosePointsFinder
from fdm.analysis.tools import create_weights_distributor

from fdm.analysis.analyzer import (
    AnalysisType, create_linear_system_solver, create_eigenproblem_solver
)


def get_solvers():
    return {
        AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS: create_linear_system_solver(
            input_builder()
        ),
        AnalysisType.EIGENPROBLEM: create_eigenproblem_solver(
            input_builder(),
        )}


def input_builder():
    def build(model, nodes, variables):
        matrix_builder = MatrixBuilder(variables)
        matrix_template = [(template_nodes, elements) for template_nodes, (elements, _) in model.template]
        A = compose_array(matrix_builder, matrix_template)

        vector_builder = VectorBuilder(variables)
        vector_template = [(template_nodes, [function]) for template_nodes, (_, function) in model.template]
        b = compose_array(vector_builder, vector_template)
        b = numpy.ravel(b)

        return A, b

    return build


def compose_array(builder, template):
    array = numpy.zeros(builder.size)
    for nodes, elements in template:
        builder.restore()
        _array = build_array(builder, nodes, elements)
        numpy.add(array, _array, out=array)
    return array


def build_array(builder, nodes, elements):
    last_element_idx = len(elements) - 1
    for i, element in enumerate(elements):
        apply_nodes = nodes if i == last_element_idx else None
        builder.apply(element, apply_nodes)
    return builder.get()


class ArrayBuilder(metaclass=abc.ABCMeta):
    def __init__(self, variables):
        self._variables = variables

        self._array = None
        self.restore()

        self._points = tuple(variables.keys())

    @property
    def size(self):
        return self._array.shape

    def restore(self):
        n = len(self._variables)
        self._array = self._create_array(n)

    @abc.abstractmethod
    def _create_array(self, size):
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, element, nodes=None):
        raise NotImplementedError

    def get(self):
        return self._array


class VectorBuilder(ArrayBuilder):
    def __init__(self, variables):
        super().__init__(variables)

    def _create_array(self, size):
        return numpy.zeros((size, 1))

    def apply(self, calculator, nodes=None):
        points = self._points if nodes is None else nodes
        for point in points:
            idx = self._variables[point]
            self._array[idx, 0] = calculator(point)
        return self._array


class MatrixBuilder(ArrayBuilder):
    def __init__(self, variables):
        super().__init__(variables)

    def _create_array(self, size):
        return numpy.identity(size)

    def apply(self, element, nodes=None):
        modifier = self._create_modifier(element, nodes)
        self._array = output = numpy.dot(modifier, self._array)
        return output

    def _create_modifier(self, element, nodes=None):
        modifier = numpy.zeros(self.size)
        points = self._points if nodes is None else nodes
        for point in points:
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


def distribute_scheme_to_nodes(nodes, scheme):
    free_points = tuple(scheme)
    distributor = create_weights_distributor(
        ClosePointsFinder(nodes, free_points)
    )
    return scheme.distribute(distributor)


def flatten_equation(equation):
    return [equation]
