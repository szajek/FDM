import unittest

from fdm.equation import Operator, Stencil, Number, Scheme, LazyOperation
from fdm.geometry import Point


class OperatorTest(unittest.TestCase):
    def test_FirstOrder_CentralDiffForNodeZero_GenerateProperCoefficients(self):
        value = 3.

        linear_operator = Operator(
            Stencil.central(),
            Number(value)
        )

        result = linear_operator.expand(Point(0))

        expected = Scheme({Point(-1): -0.5, Point(1): 0.5}) * value

        self.assertEqual(expected, result)

    def test_FirstOrder_CentralDiffForNodeThree_GenerateProperCoefficients(self):
        value = 2.

        linear_operator = Operator(
            Stencil.central(),
            Number(value)
        )

        result = linear_operator.expand(Point(3))

        expected = Scheme({Point(2): -0.5, Point(4): 0.5}) * value

        self.assertEqual(expected, result)

    def test_FirstOrder_FunctionMultiplication_GenerateProperCoefficients(self):

        point = Point(3.)

        def g(point):
            return point.x

        def f(point):
            return 100 * point.x

        linear_operator = Operator(
            Stencil.central(),
            LazyOperation.multiplication(
                Number(g),
                Number(f)
            )
        )

        result = sum(linear_operator.expand(point)._weights.values())

        expected = 200*point.x  # (g*f)' = g'*f + g*f' = 1*100x + x*100 = 200x

        self.assertEqual(expected, result)

    def test_SecondOrder_CentralDiffForNodeThree_GenerateProperCoefficients(self):
        value = 3.

        linear_operator = Operator(
            Stencil.central(1.),
            Operator(
                Stencil.central(1.),
                Number(value)
            )
        )

        result = linear_operator.expand(Point(3))

        expected = Scheme({Point(2): 1., Point(3): -2., Point(4): 1.}) * value

        self.assertEqual(expected, result)


class LazyOperationTest(unittest.TestCase):
    def test_Summation_Schemes_ReturnSchemeWithProperWeights(self):

        w1 = 9.99995772128789e-06
        w2 = 0.99999577213334

        s1 = Stencil({Point(-0.5): w1, Point(0.0): w2})
        s2 = Stencil({Point(-0.5): w1, Point(0.0): w2})

        s = LazyOperation.summation(s1, s2)

        result = s.expand(Point(0))

        expected = Scheme({Point(-0.5): w1*2, Point(0.0): w2*2})

        self.assertEqual(expected, result)
