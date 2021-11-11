import unittest

from fdm import Point, ClosePointsFinder
from fdm.analysis.utils import create_weights_distributor


class WeightDistributorTest(unittest.TestCase):
    def test_Distribute_CoincidentWithNode_ReturnDictWithValues(self):
        points = p1, p2 = [Point(0.), Point(1.)]
        close_point_finder = ClosePointsFinder(points, [p1])

        distributor = self._create(close_point_finder)

        result = distributor(p1, 1.)

        expected = {p1: 1., p2: 0.}

        self.assertEqual(expected, result)

    def test_Distribute_BetweenNodes_ReturnDictWithValues(self):
        points = p1, p2 = Point(1.), Point(2.)
        middle_point = Point(1.2)
        indexed_points = ClosePointsFinder(points, [middle_point])

        distributor = self._create(indexed_points)

        result = distributor(middle_point, 1.)

        expected = {p1: .8, p2: 0.2}

        self.assertTrue(expected.keys() == result.keys())
        self.assertTrue([(expected[k] - result[k]) < 1e-6 for k in expected.keys()])

    def test_Distribute_LastNode_ReturnDictWithValues(self):
        points = p1, p2 = Point(1.), Point(2.)
        indexed_points = ClosePointsFinder(points, [p2])

        distributor = self._create(indexed_points)

        result = distributor(p2, 1.)

        expected = {p1: 0., p2: 1.}

        self.assertEqual(expected, result)

    def _create(self, *args, **kwargs):
        return create_weights_distributor(*args, **kwargs)