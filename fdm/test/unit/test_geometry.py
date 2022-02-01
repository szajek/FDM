import unittest
import numpy as np
import mock
from numpy.testing import assert_allclose

import fdm.geometry

from fdm.geometry import (
    Point, Vector, calculate_extreme_coordinates, BoundaryBox, FreeVector, detect_dimension
)


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

    def test_Hash_PointsAreDifferent_HashIsDifferent(self):
        self.assertFalse(hash(Point(-1., 0.0, 0.0)) == hash(Point(-2., 0.0, 0.0)))

    def test_Hash_PointsInCloseDistance_HashIsTheSame(self):
        self.assertTrue(hash(Point(1 + 1e-8)) == hash(Point(1)))

    def test_Hash_PointsNotCloseEnough_HashDifferent(self):
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


class CreateClosePointsFinderTest(unittest.TestCase):
    def test_Points1d_Always_Return1dFinder(self):
        points = [Point(0.,), Point(1.), Point(2.)]
        finder = self.create(points, [Point(2.)])

        self.assertIsInstance(finder._finder, fdm.geometry.ClosePointsFinder1d)

    def test_Points2d_Always_Return2dFinder(self):
        points = [Point(0., 1.), Point(1.), Point(2.)]
        finder = self.create(points, [Point(2.)])

        self.assertIsInstance(finder._finder, fdm.geometry.ClosePointsFinder2d)

    @classmethod
    def create(cls, finder_1d=None, finder_2d=None):
        return fdm.geometry.create_close_point_finder(
            finder_1d or cls.mock_finder(),
            finder_2d or cls.mock_finder(),
        )


class CachedClosePointFinderTest(unittest.TestCase):
    def test_Find_TwiceWithVariousPoints_CallFinderTwice(self):
        finder = self.mock_finder()

        cached = self.create(finder)

        cached(Point(1.))
        cached(Point(2.))

        self.assertEqual(2, finder.call_count)

    def test_Find_TwiceWithTheSamePoints_CallFinderOnce(self):
        finder = self.mock_finder()

        cached = self.create(finder)

        cached(Point(1.))
        cached(Point(1.))

        self.assertEqual(1, finder.call_count)

    @classmethod
    def create(cls, finder):
        return fdm.geometry.CachedClosePointFinder1d(
            finder or cls.mock_finder()
        )

    @staticmethod
    def mock_finder():
        f = mock.Mock()
        return f


class ClosePointsFinder1dTest(unittest.TestCase):
    def test_Call_PointsAgree_ReturnTheClosestPointsAndDistances(self):
        points = p1, p2, p3 = [Point(0.), Point(1.), Point(3.), ]
        finder = self._create(points)

        result = finder(p2)

        expected = {p2: 0., p3: 2.}

        self.assertEqual(expected, result)

    def test_Call_PointInHalfDistance_ReturnTheClosestPointsAndDistances(self):
        points = p1, p2, p3 = [Point(0.), Point(1.), Point(3.), ]
        point = Point(2.)
        finder = self._create(points)

        result = finder(point)

        expected = {p2: 1., p3: 1.}

        self.assertEqual(expected, result)

    def test_Call_PointInHalfDistanceBasePointNotSorted_ReturnTheClosestPointsAndDistances(self):
        points = p3, p1, p2 = [Point(3.), Point(0.), Point(1.), ]
        point = Point(2.)
        finder = self._create(points)

        result = finder(point)

        expected = {p2: 1., p3: 1.}

        self.assertEqual(expected, result)

    def test_Call_PointCloserToTheLeftNode_ReturnTheClosestPointsAndDistances(self):
        points = p1, p2, p3 = [Point(0.), Point(1.), Point(3.5), ]
        point = Point(1.5)
        finder = self._create(points)

        result = finder(point)

        expected = {p2: .5, p3: 2.0}

        self.assertEqual(expected, result)

    def test_Call_PointCloserToTheRightNode_ReturnTheClosestPointsAndDistances(self):
        points = p1, p2, p3 = [Point(0.), Point(1.), Point(3.), ]
        point = Point(2.5)
        finder = self._create(points)

        result = finder(point)

        expected = {p2: 1.5, p3: .5}

        self.assertEqual(expected, result)

    def test_Call_PointAfterLast_RaisePointBeyondDomainException(self):
        points = [Point(0.), Point(1.)]
        point = Point(1.1)
        finder = self._create(points, tolerance=1e-6)

        with self.assertRaises(fdm.geometry.PointBeyondDomainException):
            finder(point)

    def test_Call_PointSlightlyAfterLast_ReturnTheClosestPointsAndDistances(self):
        points = p1, p2 = [Point(0.), Point(3.), ]
        to_find = Point(3.0001)
        finder = self._create(points, tolerance=1e-3)

        actual = finder(to_find)

        expected = {p1: 3., p2: 0.}

        self.assertEqualWithTol(expected, actual, atol=1e-3)

    def test_Call_PointBeforeFirst_RaisePointBeyondDomainException(self):
        points = [Point(0.), Point(1.)]
        point = Point(-0.1)
        finder = self._create(points, tolerance=1e-6)

        with self.assertRaises(fdm.geometry.PointBeyondDomainException):
            finder(point)

    def test_Call_PointSlightlyBeforeFirst_ReturnTheClosestPointsAndDistances(self):
        points = p1, p2 = [Point(0.), Point(3.), ]
        to_find = Point(-0.0001)
        finder = self._create(points, tolerance=1e-3)

        actual = finder(to_find)

        expected = {p1: 3., p2: 0.}

        self.assertEqualWithTol(expected, actual, atol=1e-3)

    def assertEqualWithTol(self, expected, actual, atol):
        expected_xs = self._to_xs(expected)
        actual_xs = self._to_xs(actual)
        assert_allclose(expected_xs, actual_xs, atol=atol)

    @staticmethod
    def _to_xs(array):
        return [p.x for p in array]

    @staticmethod
    def _create(*args, **kwargs):
        return fdm.geometry.ClosePointsFinder1d(*args, **kwargs)


class ClosePointsFinder2dTest(unittest.TestCase):
    def test_PointAgree_Always_ReturnTheClosestPointsAndDistances(self):
        points = p1, p2, p3 = [Point(0., 0.), Point(1., 0), Point(0., 1.), ]
        finder = self._create(points)

        result = finder(p2)

        expected = {p1: 1., p2: 0., p3: 1.4142}

        self.assertEqualWithTol(expected, result)

    def test_Call_PointInHalfDistance_ReturnTheClosestPointsAndDistances(self):
        points = p1, p2, p3 = [Point(0., 0.), Point(1., 0), Point(0., 1.), ]
        to_find = Point(0.5, 0.5)
        finder = self._create(points)

        result = finder(to_find)

        expected = {p1: 0.7071, p2: 0.7071, p3: 0.7071}

        self.assertEqualWithTol(expected, result)

    def test_Call_PointCloserToOneNode_ReturnTheClosestPointsAndDistances(self):
        points = p1, p2, p3 = [Point(0., 0.), Point(1., 0), Point(0., 1.), ]
        to_find = Point(0.1, 0.1)
        finder = self._create(points)

        result = finder(to_find)

        expected = {p1: 0.14142, p2: 0.9055, p3: 0.9055}

        self.assertEqualWithTol(expected, result)

    def assertEqualWithTol(self, expected, actual, atol=1e-3):
        self.assertEqual(len(expected), len(actual))

        def find_point_in_actual(p):
            for act in actual:
                if abs(p.x - act.x) < atol and abs(p.y - act.y) < atol:
                    return act

        for exp_p, exp_distance in expected.items():
            act_p = find_point_in_actual(exp_p)
            assert act_p is not None, 'Point {} not found in actual'.format(exp_p)
            act_distance = actual[act_p]
            assert_allclose(exp_distance, act_distance, atol=atol)

    @staticmethod
    def _create(*args, **kwargs):
        return fdm.geometry.ClosePointsFinder2d(*args, **kwargs)