import unittest

import numpy as np

from fdm.geometry import Point, Vector, calculate_extreme_coordinates, BoundaryBox, FreeVector, \
    ClosePointsFinder, detect_dimension


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


class DetectDimensionTest(unittest.TestCase):
    def test_OnlyXNotZero_Always_ReturnOne(self):
        points_array = np.array(
            [
                [1., 0., 0.],
                [-1., 0., 0.],
            ]
        )

        result = detect_dimension(points_array)

        expected = 1

        self.assertEqual(expected, result)

    def test_OnlyXYNotZero_Always_ReturnTwo(self):
        points_array = np.array(
            [
                [1., 0., 0.],
                [-1., 1., 0.],
            ]
        )

        result = detect_dimension(points_array)

        expected = 2

        self.assertEqual(expected, result)

    def test_OnlyZ_Always_ReturnThree(self):
        points_array = np.array(
            [
                [0., 0., 0.],
                [0., 0., 1.],
            ]
        )

        result = detect_dimension(points_array)

        expected = 3

        self.assertEqual(expected, result)


class ClosePointsFinderForOneDimensionTest(unittest.TestCase):

    def test_Call_PointsAgree_ReturnTheClosestPointsAndDistances(self):
        points = p1, p2, p3 = [Point(0.), Point(1.), Point(3.), ]
        finder = self._create(points, [p2, Point(2.5)])

        result = finder(p2)

        expected = {p1: 1., p2: 0.}

        self.assertEqual(expected, result)

    def test_Call_PointInHalfDistance_ReturnTheClosestPointsAndDistances(self):
        points = p1, p2, p3 = [Point(0.), Point(1.), Point(3.), ]
        point = Point(2.)
        finder = self._create(points, [point])

        result = finder(point)

        expected = {p2: 1., p3: 1.}

        self.assertEqual(expected, result)

    def test_Call_PointCloserToTheLeftNode_ReturnTheClosestPointsAndDistances(self):
        points = p1, p2, p3 = [Point(0.), Point(1.), Point(3.5), ]
        point = Point(1.5)
        finder = self._create(points, [point, Point(2.5)])

        result = finder(point)

        expected = {p2: .5, p3: 2.0}

        self.assertEqual(expected, result)

    def test_Call_PointCloserToTheRightNode_ReturnTheClosestPointsAndDistances(self):
        points = p1, p2, p3 = [Point(0.), Point(1.), Point(3.), ]
        point = Point(2.5)
        finder = self._create(points, [point])

        result = finder(point)

        expected = {p2: 1.5, p3: .5}

        self.assertEqual(expected, result)

    def test_Call_PointSlightlyBeyondDomainAndToleranceGiven_ReturnTheClosestPointsAndDistances(self):
        points = p1, p2 = [Point(0.), Point(3.), ]
        tol = 1e-6
        to_find = Point(3. + tol)
        finder = self._create(points, [to_find], tolerance=tol)

        result = finder(to_find)

        expected = {p1: 3., p2: 0.}

        for p in points:
            self.assertTrue(abs(expected[p] - result[p]) < tol*1.1)

    @staticmethod
    def _create(*args, **kwargs):
        return ClosePointsFinder(*args, **kwargs)