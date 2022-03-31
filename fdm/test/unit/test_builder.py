import unittest

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


if __name__ == '__main__':
    unittest.main()
