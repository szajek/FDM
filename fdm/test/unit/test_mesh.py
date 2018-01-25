import unittest

from fdm.mesh import Mesh1DBuilder


class Mesh1DBuilderTest(unittest.TestCase):
    def test_AddUniformlyDistributedNodes_NumberIsZeroOrOne_Raise(self):

        b = Mesh1DBuilder(4)

        with self.assertRaises(AttributeError):
            b.add_uniformly_distributed_nodes(1)

    def test_AddUniformlyDistributedNodes_MoreThanOne_CreateGivenNumberOfEquallySpacedNodes(self):
        node_number = 3

        nodes = self._create_uniformly_distributed_nodes(node_number)

        self.assertEqual(node_number, len(nodes))

        self.assertEqual(1., len(set(self._calculate_spaces_between_nodes(nodes))))

    def _calculate_spaces_between_nodes(self, nodes):
        def calc_distance_to_predecessor(i):
            return nodes[i].x - nodes[i - 1].x

        return [calc_distance_to_predecessor(i) for i in range(1, len(nodes))]

    def _create_uniformly_distributed_nodes(self, node_number):
        mesh = (
            Mesh1DBuilder(6)
                .add_uniformly_distributed_nodes(node_number)
                .create()
        )
        return mesh.real_nodes

