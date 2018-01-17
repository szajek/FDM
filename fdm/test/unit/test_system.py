import unittest

import numpy as np
from mock import MagicMock

from fdm import Mesh1DBuilder, Mesh
from fdm.equation import Scheme, LinearEquationTemplate
from fdm.geometry import Point
from fdm.system import (LinearEquation, model_to_equations, EquationWriter,
                        Output,
                        create_variables)


def create_mesh(node_number, virtual_nodes=(), delta=2.):
    length = delta*node_number
    return Mesh(
        [Point(i*length/(node_number - 1)) for i in range(node_number)],
        virtual_nodes,
    )


class ModelToEquationsTest(unittest.TestCase):
    def test_Call_Always_ReturnEquationsCreatedBasedOnGivenTemplateAndMesh(self):
        def get_scheme(i):
            return Scheme({i: i.x})

        def get_free_value(i):
            return i.x

        length = 2.
        mesh_builder = Mesh1DBuilder(length=length)
        mesh_builder.add_uniformly_distributed_nodes(3)
        mesh = mesh_builder.create()

        equation = LinearEquationTemplate(get_scheme, get_free_value)

        model = MagicMock(
            mesh=mesh,
            bcs={},
            equation=equation,
        )

        equations = model_to_equations(model)

        expected_coefficients = [
            Scheme({Point(0): 0}),
            Scheme({Point(1): 1}),
            Scheme({Point(2): 2}),
        ]

        for equation_number in range(0, 3):
            equation = equations[equation_number]

            expected_free_value = get_free_value(Point(equation_number))

            self.assertEqual(expected_coefficients[equation_number], equation.scheme)
            self.assertEqual(expected_free_value, equation.free_value)


class EquationWriterTest(unittest.TestCase):
    def test_ToCoefficientsArray_PointNotFound_Raise(self):
        eq = LinearEquation(Scheme({0: 2.}), 1.)
        writer = EquationWriter(eq, {})

        with self.assertRaises(AttributeError):
            writer.to_coefficients_array(3)

    def test_ToCoefficientsArray_RenumeratorProvided_ReturnArrayWithWeightsInRenumberedPosition(self):
        eq = LinearEquation({0: 2.}, 1.)
        writer = EquationWriter(eq, {0: 1})

        result = writer.to_coefficients_array(3)

        expected = np.array([0., 2., 0.],)

        np.testing.assert_allclose(expected, result)

    def test_ToFreeValue_Always_ReturnFreeValue(self):
        eq = LinearEquation({0: 2.}, 1.)
        writer = EquationWriter(eq, {0: 1})

        result = writer.to_free_value()

        expected = np.array(1., )

        np.testing.assert_allclose(expected, result)


class CreateVariablesTest(unittest.TestCase):
    def test_Call_OnlyRealNodes_ReturnNodeToVariableNumberMapper(self):
        real_nodes = n1, n2 = Point(1.), Point(2.)

        grid = self._create_grid(real_nodes)

        result = create_variables(grid)

        expected = {n1: 0, n2: 1}

        self.assertEqual(expected, result)

    def test_Call_RealAndVirtualNodes_ReturnNodeToVariableNumberMapper(self):
        real_nodes = n1, n2 = Point(1.), Point(2.)
        virtual_nodes = v1, v2 = Point(-1.), Point(3.)

        grid = self._create_grid(real_nodes, virtual_nodes)

        result = create_variables(grid)

        expected = {n1: 0, n2: 1, v1: 2, v2: 3}

        self.assertEqual(expected, result)

    def _create_grid(self, real_nodes, virtual_nodes=()):
        return Mesh(
            real_nodes,
            virtual_nodes=virtual_nodes
        )


class OutputTest(unittest.TestCase):
    def test_GetItem_IndexInMesh_ReturnValueInRealNode(self):
        value = 2
        o = Output([1, value, 3], 2, {})

        result = o[1]

        expected = value

        self.assertEquals(expected, result)

    def test_GetItem_NegativeIndex_ReturnValueInVirtualNode(self):
        value = 3.
        virtual_node_address = -1
        address_forwarder = {virtual_node_address: 2}
        o = Output([1, 2, value, 4], 2, address_forwarder)

        result = o[virtual_node_address]

        expected = value

        self.assertEquals(expected, result)

    def test_GetItem_PositiveIndex_ReturnValueInVirtualNode(self):
        value = 2
        virtual_node_address = 3
        address_forwarder = {virtual_node_address: 2}
        o = Output([1, 2, value], 2, address_forwarder)

        result = o[virtual_node_address]

        expected = value

        self.assertEquals(expected, result)
