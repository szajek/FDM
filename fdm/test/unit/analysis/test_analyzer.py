import unittest
from mock import MagicMock

import fdm.analysis.analyzer

from fdm import Mesh
from fdm.analysis.analyzer import (Output, create_variables, OrderedNodes)
from fdm.geometry import Point


def create_mesh(node_number, virtual_nodes=(), delta=2.):
    length = delta*node_number
    return Mesh(
        [Point(i*length/(node_number - 1)) for i in range(node_number)],
        virtual_nodes,
    )


class CreateVariablesTest(unittest.TestCase):
    def test_Call_OnlyRealNodes_ReturnNodeToVariableNumberMapper(self):
        real_nodes = n1, n2 = Point(1.), Point(2.)

        nodes = self._create_ordered_nodes(real_nodes)

        result = self._create(nodes)

        expected = {n1: 0, n2: 1}

        self.assertEqual(expected, result)

    def test_Call_RealAndVirtualNodes_ReturnNodeToVariableNumberMapper(self):
        real_nodes = n1, n2 = Point(1.), Point(2.)
        virtual_nodes = v1, v2 = Point(-1.), Point(3.)

        nodes = self._create_ordered_nodes(real_nodes, virtual_nodes)

        result = self._create(nodes)

        expected = {n1: 0, n2: 1, v1: 2, v2: 3}

        self.assertEqual(expected, result)

    def _create_ordered_nodes(self, real_nodes, virtual_nodes=()):
        return real_nodes + virtual_nodes

    @staticmethod
    def _create(nodes):
        return fdm.analysis.analyzer.create_variables(nodes)


class ExtendVariablesTest(unittest.TestCase):
    def test_Call_NoExtensionPoints_ReturnTheSame(self):
        variables = self._create_variables(Point(1.), Point(2.))

        nodes = ()

        result = self._extend(variables, nodes)

        expected = variables

        self.assertEqual(expected, result)

    def test_Call_OneNode_ReturnExtendedVariables(self):
        variables = self._create_variables(Point(1.), Point(2.))

        nodes = [Point(1.5)]

        result = self._extend(variables, nodes)

        expected = self._create_variables(Point(1.), Point(2.), Point(1.5))

        self.assertEqual(expected, result)

    @staticmethod
    def _create_variables(*nodes):
        return create_variables(nodes)

    @staticmethod
    def _extend(variables, nodes):
        return fdm.analysis.analyzer.extend_variables(variables, nodes)


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


class OrderedNodesTest(unittest.TestCase):
    def test_IndicesForReal_Always_ReturnRealNodesIndices(self):

        rn = [Point(1.)]
        vn = [Point(-1.)]

        mesh = MagicMock(
            real_nodes=rn,
            virtual_nodes=vn,
        )

        o = OrderedNodes(mesh)

        result = o.indices_for_real

        expected = [0]

        self.assertEquals(expected, result)