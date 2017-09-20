import unittest

from fdm.equation import Operator, Stencil, Number, Scheme, LazyOperation, NodeFunction


class OperatorTest(unittest.TestCase):
    def test_FirstOrder_CentralDiffForNodeZero_GenerateProperCoefficients(self):
        value = 3.

        linear_operator = Operator(
            Stencil.central(),
            Number(value)
        )

        result = linear_operator.expand(0)

        expected = Scheme({-1: -0.5, 1: 0.5}) * value

        self.assertEqual(expected, result)

    def test_FirstOrder_CentralDiffForNodeThree_GenerateProperCoefficients(self):
        value = 2.

        linear_operator = Operator(
            Stencil.central(),
            Number(value)
        )

        result = linear_operator.expand(3)

        expected = Scheme({2: -0.5, 4: 0.5}) * value

        self.assertEqual(expected, result)

    def test_FirstOrder_FunctionMultiplication_GenerateProperCoefficients(self):

        node_address = 3.

        def g(address):
            return address

        def f(address):
            return 100*address

        linear_operator = Operator(
            Stencil.central(),
            LazyOperation.multiplication(
                Number(NodeFunction(g)),
                Number(NodeFunction(f))
            )
        )

        result = sum(linear_operator.expand(node_address)._weights.values())

        expected = 200*node_address  # (g*f)' = g'*f + g*f' = 1*100x + x*100 = 200x

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

        result = linear_operator.expand(3)

        expected = Scheme({2: 1., 3: -2., 4: 1.}, order=2.) * value

        self.assertEqual(expected, result)


class LazyOperationTest(unittest.TestCase):
    def test_Summation_Schemes_ReturnSchemeWithProperWeights(self):

        w1 = 9.99995772128789e-06
        w2 = 0.99999577213334

        s1 = Stencil({-0.5: w1, 0.0: w2})
        s2 = Stencil({-0.5: w1, 0.0: w2})

        s = LazyOperation.summation(s1, s2)

        result = s.expand(0)

        expected = Scheme({-0.5: w1*2, 0.0: w2*2})

        self.assertEqual(expected, result)
