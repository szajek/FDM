import unittest
from numpy.testing import assert_allclose

import fdm.mesh
import fdm.geometry


class Mesh1DBuilderTest(unittest.TestCase):
    def test_AddUniformlyDistributedNodes_NumberIsZeroOrOne_Raise(self):
        b = self._create(4)

        with self.assertRaises(AttributeError):
            b.add_uniformly_distributed_nodes(1)

    def test_AddUniformlyDistributedNodes_MoreThanOne_CreateGivenNumberOfEquallySpacedNodes(self):
        node_number = 3

        mesh = (
            self._create(4)
            .add_uniformly_distributed_nodes(node_number)
        ).create()

        actual = mesh.real_nodes

        self.assertEqual(node_number, len(actual))
        self.assertEqual(1., len(set(self._calculate_spaces_between_nodes(actual))))

    def test_AddMidNodes_Always_CreateAdditionalNodesBetweenRealNodes(self):
        node_number = 4

        mesh = (
            self._create(3)
            .add_uniformly_distributed_nodes(node_number)
            .add_middle_nodes()
        ).create()

        actual = mesh.additional_nodes

        expected = (_p(0.5), _p(1.5), _p(2.5))

        self.assertPointsEqual(expected, actual)

    def assertPointsEqual(self, expected, actual):
        for e, a in zip(expected, actual):
            self.assertAlmostEqual(e, a)

    @staticmethod
    def _create(length):
        return fdm.mesh.Mesh1DBuilder(length)

    @staticmethod
    def _calculate_spaces_between_nodes(nodes):
        def calc_distance_to_predecessor(i):
            return nodes[i].x - nodes[i - 1].x

        return [calc_distance_to_predecessor(i) for i in range(1, len(nodes))]


def _p(x):
    return fdm.geometry.Point(x)
