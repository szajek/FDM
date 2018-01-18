import unittest

import numpy as np

from fdm.geometry import Point, Vector, calculate_extreme_coordinates, BoundaryBox, FreeVector, IndexedPoints


class PointTest(unittest.TestCase):

    def test_Create_OnlyX_ReturnPointWithYandZasZero(self):

        p = Point(2.)

        self.assertEqual(0., p.y)
        self.assertEqual(0., p.z)

    def test_Add_FreeVector_ReturnTranslatedPoint(self):

        result = Point(1., 2., 3.) + FreeVector(Point(3., 2., 1.))

        expected = Point(4., 4., 4.)

        self.assertEqual(expected, result)

    def test_Subtract_FreeVector_ReturnTranslatedPoint(self):

        result = Point(1., 2., 3.) - FreeVector(Point(3., 2., 1.))

        expected = Point(-2., 0., 2.)

        self.assertEqual(expected, result)

    def test_RightMultiply_IntOrFloat_ReturnPointWithMultipliedCoords(self):
        result = Point(1., 2., 3.) * 3.

        expected = Point(3., 6., 9.)

        self.assertEqual(expected, result)

    def test_LeftMultiply_IntOrFloat_ReturnPointWithMultipliedCoords(self):
        result = 3. * Point(1., 2., 3.)

        expected = Point(3., 6., 9.)

        self.assertEqual(expected, result)

    def test_Negative_Always_ReturnPointWithInvertedCoords(self):
        result = -Point(1., 2., 3.)

        expected = Point(-1., -2., -3.)

        self.assertEqual(expected, result)

    def test_Eq_PointsAreTheSame_MergeWithSomeTolerance(self):
        self.assertTrue(Point(1) == Point(1))

    def test_Eq_PointsInCloseDistance_MergeWithSomeTolerance(self):
        self.assertTrue(Point(1 + 1e-8) == Point(1))

    def test_Eq_PointsNotCloseEnough_MergeWithSomeTolerance(self):
        self.assertFalse(Point(1 + 1e-7) == Point(1))

    def test_Hash_PointsAreTheSame_HashIsTheSame(self):
        self.assertTrue(hash(Point(1)) == hash(Point(1)))

    def test_Hash_PointsInCloseDistance_HashIsTheSame(self):
        self.assertTrue(hash(Point(1 + 1e-8)) == hash(Point(1)))

    def test_Hash_PointsNotCloseEnough_HashIsTheSame(self):
        self.assertFalse(hash(Point(1 + 1e-7)) == hash(Point(1)))


class VectorTest(unittest.TestCase):

    def test_Length_OnlyX_ReturnDistanceBetweenNodes(self):
        p1 = Point(2.)
        p2 = Point(2.55)

        result = Vector(p1, p2).length

        expected = 2.55 - 2.

        self.assertAlmostEqual(expected, result)

    def test_Iterate_Always_IterateOverPoints(self):

        nodes = [Point(1.), Point(2.)]
        vector = Vector(*nodes)

        for i, point in enumerate(vector):
            self.assertEqual(nodes[i], point)

    def test_Negative_Always_ReturnVectorWithInvertedPoints(self):
        nodes = [Point(1.), Point(2.)]
        vector = Vector(*nodes)

        result = -vector

        expected = Vector(vector.end, vector.start)

        self.assertEqual(expected, result)


class CalculateExtremeCoordinatesTest(unittest.TestCase):
    def test_Call_OnlyXCoordinate_ReturnListsOfExtremes(self):
        nodes = [Point(1., -2., 4.), Point(2., 5, -5), Point(3., 9., 4.)]

        result = calculate_extreme_coordinates(nodes)

        expected = (
           (1., -2., -5.),
           (3., 9., 4.)
        )

        self.assertEqual(expected, result)


class BoundaryBoxTest(unittest.TestCase):
    def test_Dimensions_OnlyXCoordinate_ReturnXRangeAndZeroForYAndZ(self):
        points = [Point(-1.), Point(5.)]
        bbox = BoundaryBox.from_points(points)

        result = bbox.dimensions

        expected = (6., 0., 0.)

        self.assertEqual(expected, result)


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