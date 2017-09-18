import unittest
import numpy as np

from fdm.domain import Grid1DBuilder
from fdm.equation import Operator, Stencil, Number, LinearEquationTemplate, NodeFunction
from fdm.model import create_bc, Model
from fdm.system import solve


def _create_domain(length, node_number):
    domain_builder = Grid1DBuilder(length)
    domain_builder.add_uniformly_distributed_nodes(node_number)
    return domain_builder.create()


def _create_linear_function(length, node_number, a, b):
    def calc(node_address):
        x = (node_address / (node_number - 1) * length)
        return a*x + b
    return calc


def _create_domain(length, node_number):
    domain_builder = Grid1DBuilder(length)
    domain_builder.add_uniformly_distributed_nodes(node_number)
    return domain_builder.create()


def _create_equation(linear_operator, free_vector):
    return LinearEquationTemplate(
        linear_operator,
        free_vector
    )


def _create_fixed_and_free_end_bc(node_number):
    return {
        0: create_bc('dirichlet'),
        node_number - 1: create_bc('neumann', Stencil.backward(), value=0.)
    }


def _create_fixed_ends_bc(node_number):
    return {
        0: create_bc('dirichlet', value=0.),
        node_number - 1: create_bc('dirichlet', value=0.)
    }


_bcs = {
    'fixed_free': _create_fixed_and_free_end_bc,
    'fixed_fixed': _create_fixed_ends_bc,
}


def _create_bc(_type, node_number):
    return _bcs[_type](node_number)


def _create_standard_operator(A, E):
    return Operator(
        Stencil.central(1.),
        Number(A) * Number(E) * Operator(
            Stencil.central(1.),
        )
    )


def _solve_for_classical(analysis_type, domain, bc_type, load_function_coefficients, cross_section=1.):
    node_number = len(domain.nodes)
    length = domain.boundary_box.dimensions[0]
    a, b = load_function_coefficients
    result = solve(
        analysis_type,
        Model(
            _create_equation(
                _create_standard_operator(A=cross_section, E=1.),
                _create_linear_function(length, node_number, a=a, b=b)
            ),
            domain,
            _create_bc(bc_type, node_number))
    )
    return result


class TrussStaticEquationFiniteDifferencesTest(unittest.TestCase):
    def test_ConstantSectionAndYoungModulus_ReturnCorrectDisplacement(self):
        domain = _create_domain(length=1., node_number=6)

        result = _solve_for_classical(
            'linear_system_of_equations', domain, 'fixed_free', load_function_coefficients=(-1., 0.))

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

        domain = _create_domain(length, node_number)

        def cross_section(node_address):
            x = domain.get_by_address(node_address).x
            return 2. - (x / length) * 1.

        result = _solve_for_classical(
            'linear_system_of_equations',
            domain, 'fixed_free',
            load_function_coefficients=(0., -1.),
            cross_section=NodeFunction.with_linear_interpolator(cross_section),
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
    def test_ConstantSectionAndYoung_ReturnCorrectDisplacement(self):
        domain = _create_domain(length=1., node_number=6)
        ro = 2.

        result = _solve_for_classical('eigenproblem', domain, 'fixed_fixed', load_function_coefficients=(-ro, 0.))

        expected = np.array([0., -0.3717, -0.6015, -0.6015, -0.3717, 0.],)

        np.testing.assert_allclose(expected, result, atol=1e-4)


