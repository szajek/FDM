import unittest

import numpy as np

from fdm import Scheme
from fdm.analysis.up_to_down import SchemeWriter, FreeValueWriter
from fdm.geometry import Point


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


