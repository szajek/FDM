import unittest

import numpy as np
from mock import MagicMock

from fdm import Mesh
from fdm.analysis import (SchemeWriter, Output, create_variables, FreeValueWriter, OrderedNodes)
from fdm.equation import Scheme
from fdm.geometry import Point


def create_mesh(node_number, virtual_nodes=(), delta=2.):
    length = delta*node_number
    return Mesh(
        [Point(i*length/(node_number - 1)) for i in range(node_number)],
        virtual_nodes,
    )


class SchemeWriterTest(unittest.TestCase):
    def test_Write_PointNotFound_Raise(self):
        scheme = Scheme({Point(0): 2.})
        writer = SchemeWriter({Point(1): 1})

        with self.assertRaisesRegex(AttributeError, "No point in mapper found"):
            writer.write(scheme)

    def test_Write_MapperProvided_ReturnArrayWithWeightsInRenumberedPosition(self):

        writer = SchemeWriter({Point(0): 1, Point(1): 2, Point(2): 3})

        result = writer.write(Scheme({Point(0): 2.}))

        expected = np.array(
            [
                [0., 2., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
            ]
        )

        np.testing.assert_allclose(expected, result)

    def test_Write_ManySchemes_ReturnArrayWithTwoRowsFilled(self):

        writer = SchemeWriter({Point(0): 1, Point(1): 2, Point(2): 0})

        result = writer.write(
            Scheme({Point(0): 2.}),
            Scheme({Point(1): 3.}),
        )

        expected = np.array(
            [
                [0., 2., 0.],
                [0., 0., 3.],
                [0., 0., 0.],
            ]
        )

        np.testing.assert_allclose(expected, result)


class FreeValueWriterTest(unittest.TestCase):
    def test_Write_ManyItems_ReturnVectorWithManyRowsFilled(self):

        writer = FreeValueWriter({'dummy_1': 1, 'dummy_2': 1, 'dummy_3': 1, })

        result = writer.write(0., 1., 1.2)

        expected = np.array(
            [0, 1., 1.2]
        )

        np.testing.assert_array_equal(expected, result)


class CreateVariablesTest(unittest.TestCase):
    def test_Call_OnlyRealNodes_ReturnNodeToVariableNumberMapper(self):
        real_nodes = n1, n2 = Point(1.), Point(2.)

        nodes = self._create_ordered_nodes(real_nodes)

        result = create_variables(nodes)

        expected = {n1: 0, n2: 1}

        self.assertEqual(expected, result)

    def test_Call_RealAndVirtualNodes_ReturnNodeToVariableNumberMapper(self):
        real_nodes = n1, n2 = Point(1.), Point(2.)
        virtual_nodes = v1, v2 = Point(-1.), Point(3.)

        nodes = self._create_ordered_nodes(real_nodes, virtual_nodes)

        result = create_variables(nodes)

        expected = {n1: 0, n2: 1, v1: 2, v2: 3}

        self.assertEqual(expected, result)

    def _create_ordered_nodes(self, real_nodes, virtual_nodes=()):
        return real_nodes + virtual_nodes


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