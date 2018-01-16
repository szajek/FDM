import unittest

from fdm.mesh import Mesh1DBuilder, Mesh
from fdm.geometry import Point


class MeshTest(unittest.TestCase):
    def test_PositionByPoint_AddressesAgree_ReturnGridAddress(self):
        grid = self._create_3node_grid()

        result = grid._position_by_point(Point(1.))

        expected = 1

        self.assertEqual(expected, result)

    def test_PositionByPoint_AddressNotAgree_ReturnGridAddress(self):
        grid = self._create_3node_grid()

        result = grid._position_by_point(Point(2.))

        expected = 1.5

        self.assertAlmostEqual(expected, result)

    def test_DistributeToPoints_CoincidentWithNode_ReturnDictWithValues(self):

        points = p1, p2 = Point(1.), Point(2.)

        grid = self._create(points)

        result = grid.distribute_to_points(p1, 1.)

        expected = {p1: 1.}

        self.assertEqual(expected, result)

    def test_DistributeToPoints_BetweenNodes_ReturnDictWithValues(self):
        points = p1, p2 = Point(1.), Point(2.)

        grid = self._create(points)

        result = grid.distribute_to_points(Point(1.2), 1.)

        expected = {p1: .8, p2: 0.2}

        self.assertTrue(expected.keys() == result.keys())
        self.assertTrue([(expected[k] - result[k]) < 1e-6 for k in expected.keys()])

    def test_DistributeToPoints_LastNode_ReturnDictWithValues(self):
        points = p1, p2 = Point(1.), Point(2.)

        grid = self._create(points)

        result = grid.distribute_to_points(p2, 1.)

        expected = {p2: 1.}

        self.assertEqual(expected, result)

    def _create_3node_grid(self):
        return Mesh([
            Point(0.),
            Point(1.),
            Point(3.),
        ])

    def _create(self, real_nodes, virtual_nodes=()):
        return Mesh(real_nodes, virtual_nodes=virtual_nodes)


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
        return mesh.nodes

