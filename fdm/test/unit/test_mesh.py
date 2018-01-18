import unittest

from fdm.mesh import Mesh1DBuilder, create_weights_distributor, IndexedPoints
from fdm.geometry import Point


class MeshTest(unittest.TestCase):
    pass


class IndexedPointsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._3_node = [Point(0.), Point(1.), Point(3.),]

    def test_FindIndex_PointsAgree_ReturnGridIndex(self):
        point = Point(1.)
        indexed_points = self._create(self._3_node, [point, Point(2.5)])

        result = indexed_points.get_index(point)

        expected = 1

        self.assertEqual(expected, result)

    def test_Call_PointInHalfDistance_ReturnGridIndex(self):
        point = Point(2.)
        indexed_points = self._create(self._3_node, [point])

        result = indexed_points.get_index(point)

        expected = 1.5

        self.assertAlmostEqual(expected, result)

    def test_Call_PointCloserToTheLeftNode_ReturnGridIndex(self):
        point = Point(1.5)
        indexed_points = self._create(self._3_node, [point, Point(2.5)])

        result = indexed_points.get_index(point)

        expected = 1.25

        self.assertAlmostEqual(expected, result)

    def test_Call_PointCloserToTheRightNode_ReturnGridIndex(self):
        point = Point(2.5)
        indexed_points = self._create(self._3_node, [point])

        result = indexed_points.get_index(point)

        expected = 1.75

        self.assertAlmostEqual(expected, result)

    def _create(self, *args, **kwargs):
        return IndexedPoints(*args, **kwargs)


class WeightDistributorTest(unittest.TestCase):

    def test_Distribute_CoincidentWithNode_ReturnDictWithValues(self):
        points = p1, p2 = [Point(0.), Point(1.)]
        indexed_points = IndexedPoints(points, [p1])

        distributor = self._create(indexed_points)

        result = distributor(p1, 1.)

        expected = {p1: 1.}

        self.assertEqual(expected, result)

    def test_Distribute_BetweenNodes_ReturnDictWithValues(self):
        points = p1, p2 = Point(1.), Point(2.)
        middle_point = Point(1.2)
        indexed_points = IndexedPoints(points, [middle_point])

        distributor = self._create(indexed_points)

        result = distributor(middle_point, 1.)

        expected = {p1: .8, p2: 0.2}

        self.assertTrue(expected.keys() == result.keys())
        self.assertTrue([(expected[k] - result[k]) < 1e-6 for k in expected.keys()])

    def test_Distribute_LastNode_ReturnDictWithValues(self):
        points = p1, p2 = Point(1.), Point(2.)
        indexed_points = IndexedPoints(points, [p2])

        distributor = self._create(indexed_points)

        result = distributor(p2, 1.)

        expected = {p2: 1.}

        self.assertEqual(expected, result)

    def _create(self, *args, **kwargs):
        return create_weights_distributor(*args, **kwargs)


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

