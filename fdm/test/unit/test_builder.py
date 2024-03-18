import unittest
from numpy.testing import assert_allclose

import fdm.builder

from fdm.builder import SplineExtrapolation
from fdm.geometry import Point


class SplineValueControllerTest(unittest.TestCase):
    def test_Get_PointOn_ReturnInterpolatedValue(self):
        c = self._create([1., 2., 3., 4.])
        c.update_by_control_points([10., 20., 30., 40.])

        actual = c.get(Point(2.))

        expected = 20.

        self.assertAlmostEqual(expected, actual)

    def test_Get_PointBetween_ReturnInterpolatedValue(self):
        c = self._create([1., 2., 3., 4.])
        c.update_by_control_points([10., 20., 30., 40.])

        actual = c.get(Point(1.5))

        expected = 15.

        self.assertAlmostEqual(expected, actual)

    def test_Get_PointOutsideExtrapolationNone_ReturnZero(self):
        c = self._create([1., 2., 3., 4.], extrapolation=SplineExtrapolation.NONE)
        c.update_by_control_points([10., 20., 30., 40.])

        actual = c.get(Point(45.))

        expected = 0.

        self.assertAlmostEqual(expected, actual)

    def test_Get_PointOutsideExtrapolationAsForEdge_ReturnAsForTheLast(self):
        c = self._create([1., 2., 3., 4.], extrapolation=SplineExtrapolation.AS_FOR_EDGE)
        c.update_by_control_points([10., 20., 30., 40.])

        actual = c.get(Point(45.))

        expected = 40.

        self.assertAlmostEqual(expected, actual)

    @staticmethod
    def _create(points, extrapolation=SplineExtrapolation.NONE):
        return fdm.builder.SplineValueController(
            points, extrapolation=extrapolation
        )


class SegmentValueControllerTest(unittest.TestCase):
    def test_Get_SingleSegmentModule_ReturnUniform(self):
        c = self._create(length=1., segments_number=5, segments_in_module=1)
        c.update((2.,))

        actual = self.expand(c, length=1., points_number=6)

        expected = [2. for _ in range(6)]

        assert_allclose(expected, actual)

    def test_Get_TwoSegmentModule_ValueForGivenSegment(self):
        c = self._create(length=1., segments_number=6, segments_in_module=2)
        c.update((2., 1.))

        actual = [
            c.get(Point(1/12)),
            c.get(Point(3/12)),
            c.get(Point(5/12)),
            c.get(Point(7/12)),
            c.get(Point(9/12)),
            c.get(Point(11/12)),
            ]

        expected = [2., 1., 2., 1., 2., 1.]

        assert_allclose(expected, actual, atol=1e-2)

    def test_Get_TwoSegmentModule_AverageValuesBetweenSegments(self):
        c = self._create(length=1., segments_number=6, segments_in_module=2)
        c.update((2., 1.))

        actual = self.expand(c, length=1., points_number=7)

        expected = [2., 1.5, 1.5, 1.5, 1.5, 1.5, 1.]

        assert_allclose(expected, actual, atol=1e-2)

    @staticmethod
    def _create(length=1., segments_number=5, segments_in_module=1.):
        return fdm.builder.SegmentsValueController(
            length, segments_number=segments_number, segments_in_module=segments_in_module
        )

    @staticmethod
    def expand(controller, length=1., points_number=6):
        dx = length / (points_number - 1)
        return [controller.get(Point(dx * i)) for i in range(points_number)]


if __name__ == '__main__':
    unittest.main()
