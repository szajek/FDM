import unittest

import dicttools
import numpy as np

from fdm.mesh import Mesh1DBuilder
from fdm.equation import Operator, Stencil, Number, LinearEquationTemplate
from fdm.geometry import Point
from fdm.model import create_bc, Model, VirtualNodeStrategy
from fdm.system import solve


def _create_linear_function(length, node_number, a, b):
    def calc(point):
        return a * point.x + b

    return calc


def _create_mesh(length, node_number):
    builder = Mesh1DBuilder(length)
    builder.add_uniformly_distributed_nodes(node_number)
    builder.add_virtual_nodes(-0.2, length + 0.2)
    domain = builder.create()
    return domain


def _create_equation(linear_operator, free_vector):
    return LinearEquationTemplate(
        linear_operator,
        free_vector
    )


def _create_fixed_and_free_end_bc(length, approx_size):
    return {
        Point(0): create_bc('dirichlet'),
        Point(length): create_bc('neumann', Stencil.backward(span=approx_size), value=0.)
    }


def _create_fixed_ends_bc(length, approx_size):
    return {
        Point(0): create_bc('dirichlet', value=0.),
        Point(length): create_bc('dirichlet', value=0.)
    }


_bcs = {
    'fixed_free': _create_fixed_and_free_end_bc,
    'fixed_fixed': _create_fixed_ends_bc,
}


def _create_bc(_type, length, approx_size):
    return _bcs[_type](length, approx_size)


def _create_virtual_nodes_bc(strategy, *coords):
    return {Point(x): create_bc('virtual_node', x, strategy) for x in coords}


def _create_standard_operator(approx_span, A, E):
    return Operator(
        Stencil.central(span=approx_span),
        Number(A) * Number(E) * Operator(
            Stencil.central(span=approx_span),
        )
    )


def _solve_for_classical(analysis_type, mesh, approx_span, bc_type, load_function_coefficients, cross_section=1.):
    node_number = len(mesh.nodes)
    length = mesh.boundary_box.dimensions[0]
    a, b = load_function_coefficients
    result = solve(
        analysis_type,
        Model(
            _create_equation(
                _create_standard_operator(approx_span, A=cross_section, E=1.),
                _create_linear_function(length, node_number, a=a, b=b)
            ),
            mesh,
            dicttools.merge(
                _create_bc(bc_type, length, approx_span),
                _create_virtual_nodes_bc(VirtualNodeStrategy.SYMMETRY, -0.2, length + 0.2),
            )
        )
    )
    return result


class TrussStaticEquationFiniteDifferencesTest(unittest.TestCase):
    def test_ConstantSectionAndYoungModulus_ReturnCorrectDisplacement(self):
        node_numbers = 6
        mesh = _create_mesh(length=1., node_number=node_numbers)
        approx_span = 1. / (node_numbers - 1)

        result = _solve_for_classical(
            'linear_system_of_equations', mesh, approx_span, 'fixed_free', load_function_coefficients=(-1., 0.),
        )

        expected = np.array(
            [
                [0.],
                [0.08],
                [0.152],
                [0.208],
                [0.24],
                [0.24],
            ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-6)

    def test_VariedSection_ReturnCorrectDisplacement(self):
        node_number, length = 6, 1.
        approx_span = 1. / (node_number - 1)

        mesh = _create_mesh(length, node_number)

        def cross_section(point):
            return 2. - (point.x / length) * 1.

        result = _solve_for_classical(
            'linear_system_of_equations',
            mesh, approx_span, 'fixed_free',
            load_function_coefficients=(0., -1.),
            cross_section=cross_section,
        )

        expected = np.array(
            [[-3.92668354e-16],
             [8.42105263e-02],
             [1.54798762e-01],
             [2.08132095e-01],
             [2.38901326e-01],
             [2.38901326e-01],
             ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-6)


class TrussDynamicEigenproblemEquationFiniteDifferencesTest(TrussStaticEquationFiniteDifferencesTest):
    @unittest.skip("Wrong BC applied")
    def test_ConstantSectionAndYoung_ReturnCorrectDisplacement(self):
        node_number = 6
        mesh = _create_mesh(length=1., node_number=node_number)
        approx_span = 1. / (node_number - 1)
        ro = 2.

        result = _solve_for_classical('eigenproblem', mesh, approx_span, 'fixed_fixed',
                                      load_function_coefficients=(-ro, 0.))

        expected = np.array([0., -0.3717, -0.6015, -0.6015, -0.3717, 0.], )

        np.testing.assert_allclose(expected, result, atol=1e-4)
