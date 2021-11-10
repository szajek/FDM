import unittest

import numpy as np

from fdm import Scheme
from fdm.analysis.up_to_down import SchemeWriter, FreeValueWriter, create_weights_distributor
from fdm.geometry import Point, ClosePointsFinder


class SchemeWriterTest(unittest.TestCase):
    def test_Write_PointNotFound_Raise(self):
        scheme = Scheme({Point(0): 2.})
        writer = SchemeWriter({Point(1): 1})

        with self.assertRaisesRegex(AttributeError, "No point in mapper found"):
            writer.write(scheme)

    def test_Write_MapperProvided_ReturnArrayWithWeightsInRenumberedPosition(self):

        writer = SchemeWriter({Point(0): 1, Point(1): 2, Point(2): 3})

        result = writer.write(Scheme({Point(0): 2.}))

        expected = np.array(
            [
                [0., 2., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
            ]
        )

        np.testing.assert_allclose(expected, result)

    def test_Write_ManySchemes_ReturnArrayWithTwoRowsFilled(self):

        writer = SchemeWriter({Point(0): 1, Point(1): 2, Point(2): 0})

        result = writer.write(
            Scheme({Point(0): 2.}),
            Scheme({Point(1): 3.}),
        )

        expected = np.array(
            [
                [0., 2., 0.],
                [0., 0., 3.],
                [0., 0., 0.],
            ]
        )

        np.testing.assert_allclose(expected, result)


class FreeValueWriterTest(unittest.TestCase):
    def test_Write_ManyItems_ReturnVectorWithManyRowsFilled(self):

        writer = FreeValueWriter({'dummy_1': 1, 'dummy_2': 1, 'dummy_3': 1, })

        result = writer.write(0., 1., 1.2)

        expected = np.array(
            [0, 1., 1.2]
        )

        np.testing.assert_array_equal(expected, result)


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