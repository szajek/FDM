import unittest
import numpy
import mock
from numpy.testing import assert_allclose

from fdm.geometry import Point
from fdm.equation import (
    Stencil, DynamicElement, Scheme, Operator, Number
)
import fdm.analysis.down_to_up
import fdm.analysis.analyzer


@mock.patch('fdm.analysis.down_to_up.build_array')
class ComposeArrayTest(unittest.TestCase):
    def test_SingleElementTemplate_Always_CallArrayBuilderWithNodesAndElement(self, mock_array_builder):
        nodes = [Point(0.), Point(4.)]
        n = len(nodes)

        builder = self.mock_builder(size=(n, n))
        mock_array_builder.return_value = numpy.zeros((n, n))

        elements = [Stencil.central()]
        template = [(nodes, elements)]

        self._build(builder, template)

        mock_array_builder.assert_called_with(builder, nodes, elements)

    def test_TwoElementsTemplate_Always_CallArrayBuilderForEachElement(self, mock_array_builder):
        nodes_1 = [Point(0.), Point(4.)]
        nodes_2 = [Point(1.), Point(3.)]

        builder = self.mock_builder(size=(5, 5))
        mock_array_builder.return_value = numpy.zeros((5, 5))

        elements_1 = [Stencil.central()]
        elements_2 = [Stencil.backward()]
        template = [(nodes_1, elements_1), (nodes_2, elements_2)]

        self._build(builder, template)

        args_1, args_2 = mock_array_builder.call_args_list
        self.assertEqual(mock.call(builder, nodes_1, elements_1), args_1)
        self.assertEqual(mock.call(builder, nodes_2, elements_2), args_2)

    @staticmethod
    def mock_builder(size=None):
        builder = mock.Mock()
        builder.size = size
        return builder

    @staticmethod
    def _build(builder, elements):
        return fdm.analysis.down_to_up.compose_array(builder, elements)


class BuildArrayTest(unittest.TestCase):
    def test_SingleElement_Always_CallBuilderApplyWithNodesAndElement(self):
        nodes = [Point(0.), Point(4.)]
        builder = self.mock_builder()
        stencil = Stencil.central()

        self._build(builder, nodes, [stencil])

        builder.apply.assert_called_with(stencil, nodes)

    def test_TwoElements_Always_CallBuilderApplyTwiceWithElementsAndNodesOnlyInLastCall(self):
        nodes = [Point(0.), Point(4.)]
        builder = self.mock_builder()
        stencil_1 = Stencil.central()
        stencil_2 = Stencil.backward()

        self._build(builder, nodes, [stencil_1, stencil_2])

        self.assertEqual(2, builder.apply.call_count)

        args_1, args_2 = builder.apply.call_args_list
        self.assertEqual(mock.call(stencil_1, None), args_1)
        self.assertEqual(mock.call(stencil_2, nodes), args_2)

    @staticmethod
    def mock_builder():
        builder = mock.Mock()
        return builder

    @staticmethod
    def _build(builder, nodes, elements):
        return fdm.analysis.down_to_up.build_array(builder, nodes, elements)


class FlattenEquationTest(unittest.TestCase):
    def test_Flatten_Stencil_ReturnListWithStencil(self):
        stencil = Stencil.central()

        actual = self._flatten(stencil)

        expected = [stencil]

        self.assertEqualElement(expected, actual)

    def test_Flatten_StencilMultiplied_ReturnListWithTheSame(self):
        stencil = Number(4.)*Stencil.central()

        actual = self._flatten(stencil)

        expected = [stencil]

        self.assertEqualElement(expected, actual)

    def test_Flatten_EmptyOperator_ReturnListWithOperator(self):
        operator = Operator(Stencil.central())
        equation = operator

        actual = self._flatten(equation)

        expected = [operator]

        self.assertEqualElement(expected, actual)

    def test_Flatten_OperatorWithElement_ReturnListWithTheSame(self):
        operator = Operator(Stencil.central(), Number(4.))
        equation = operator

        actual = self._flatten(equation)

        expected = [operator]

        self.assertEqualElement(expected, actual)

    @unittest.skip('under development')
    def test_Flatten_TwoLevelOperator_ReturnListWithTwoOperatorsInCorrectOrder(self):
        operator_1 = Operator(Stencil.backward())
        operator_2 = Operator(Stencil.central(), operator_1)
        equation = operator_2

        actual = self._flatten(equation)

        expected = [operator_1, operator_2]

        self.assertEqualElement(expected, actual)

    @staticmethod
    def _flatten(equation):
        return fdm.analysis.down_to_up.flatten_equation(equation)

    def assertEqualElement(self, expected, actual):
        for exp, act in zip(expected, actual):
            exp_scheme = exp.expand(Point(0.))
            act_scheme = act.expand(Point(0.))
            self.assertEqual(exp_scheme, act_scheme)


class MatrixBuilderTest(unittest.TestCase):
    def test_Get_DoNothing_ReturnIdentityArrayOfVariableLengthSize(self):
        points = Point(0.), Point(1.), Point(2.)
        builder = self.create(*points)

        actual = builder.get()

        expected = numpy.identity(3)

        assert_allclose(expected, actual)

    def test_ApplyStencil_SimpleMultiplication_ReturnArrayComputedBasedOnElement(self):
        points = Point(0.), Point(1.), Point(2.)
        builder = self.create(*points)

        builder.apply(Stencil({Point(0.): 5}))
        actual = builder.get()

        expected = numpy.array(
            [
                [5., 0., 0.],
                [0., 5., 0.],
                [0., 0., 5.],
            ]
        )

        assert_allclose(expected, actual)

    def test_Restore_Always_RestoreIdentityMatrix(self):
        points = Point(0.), Point(1.), Point(2.)
        builder = self.create(*points)

        builder.apply(Stencil({Point(0.): 5}))
        builder.restore()

        actual = builder.get()

        expected = numpy.array(
            [
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
            ]
        )

        assert_allclose(expected, actual)

    def test_ApplyStencil_NodesGiven_ApplyOnlyForGivenNodes(self):
        points = p1, p2, p3 = Point(0.), Point(1.), Point(2.)
        builder = self.create(*points)

        builder.apply(Stencil({Point(0.): 5}), nodes=(p2, p3))
        actual = builder.get()

        expected = numpy.array(
            [
                [0., 0., 0.],
                [0., 5., 0.],
                [0., 0., 5.],
            ]
        )

        assert_allclose(expected, actual)

    def test_ApplyStencil_TwoPoints_ReturnArrayComputedBasedOnElement(self):
        p1, p2, p3 = Point(0.), Point(1.), Point(2.)
        builder = self.create(p1, p2, p3)

        element = self._build_dynamic_element({
                p1: Stencil({Point(0.): 2.}),
                p2: Stencil({Point(-1.): -1., Point(1.): 4.}),
                p3: Stencil({Point(0.): 3.}),
            })

        builder.apply(element)
        actual = builder.get()

        expected = numpy.array(
            [
                [2., 0., 0.],
                [-1., 0., 4.],
                [0., 0., 3.],
            ]
        )

        assert_allclose(expected, actual)

    def test_ApplyStencilTwice_SimpleMultiplicationAndTwoPoints_ReturnArrayComputedForTwoElements(self):
        p1, p2, p3 = Point(0.), Point(1.), Point(2.)
        builder = self.create(p1, p2, p3)

        element_1 = Stencil({Point(0.): 5})
        element_2 = self._build_dynamic_element({
                p1: Stencil({Point(0.): 2.}),
                p2: Stencil({Point(-1.): -1., Point(1.): 4.}),
                p3: Stencil({Point(0.): 3.}),
            })

        builder.apply(element_1)
        builder.apply(element_2)
        actual = builder.get()

        expected = numpy.array(
            [
                [10., 0., 0.],
                [-5., 0., 20.],
                [0., 0., 15.],
            ]
        )

        assert_allclose(expected, actual)

    def test_ApplyStencilTwice_TwoTimesMultipointStencils_ReturnArrayComputedForTwoElements(self):
        p1, p2, p3, p4, p5 = Point(0.), Point(1.), Point(2.), Point(3.), Point(4.)
        builder = self.create(p1, p2, p3, p4, p5)

        element_1 = self._build_dynamic_element({
            p2: Stencil({Point(-1.): -1., Point(1.): 2.}),
            p4: Stencil({Point(-1.): -1., Point(1.): 2.}),
        })
        element_2 = self._build_dynamic_element({
            p3: Stencil({Point(-1.): -3., Point(1.): 4.}),
        })

        builder.apply(element_1)
        builder.apply(element_2)
        actual = builder.get()

        expected = numpy.array(
            [
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [3., 0., -10., 0, 8],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
            ]
        )

        assert_allclose(expected, actual)

    def test_ApplyStencil_MiddlePoints_ReturnArrayBasedOnDistributedValues(self):
        p1, p2, p3 = Point(0.), Point(1.), Point(2.)
        builder = self.create(p1, p2, p3)

        element = self._build_dynamic_element({
                p1: Stencil({Point(0.5): -1.}),
                p2: Stencil({Point(-0.5): -1., Point(0.25): 4.}),
                p3: Stencil({Point(-0.25): 1.}),
            })

        builder.apply(element)
        actual = builder.get()

        expected = numpy.array(
            [
                [-0.5, -0.5, 0.],
                [-0.5, 2.5, 1.],
                [0., 0.25, 0.75],
            ]
        )

        assert_allclose(expected, actual)

    def test_Apply_SecondDerivativeTwice_ReturnCorrectArray(self):
        pm1, p0, p1, p2, p3, p4, p5 = Point(-1.), Point(0.), Point(1.), Point(2.), Point(3.), Point(4.), Point(5.)
        builder = self.create(pm1, p0, p1, p2, p3, p4, p5)

        operator = fdm.Operator(fdm.Stencil.central(span=1))
        second_operator = fdm.Operator(fdm.Stencil.central(span=1), operator)

        element_1 = self._build_dynamic_element({
                p0: second_operator,
                p1: second_operator,
                p2: second_operator,
                p3: second_operator,
                p4: second_operator,
            })
        element_2 = self._build_dynamic_element({
                p1: second_operator,
                p2: second_operator,
                p3: second_operator,
            })

        builder.apply(element_1)
        builder.apply(element_2)
        actual = builder.get()

        expected = numpy.array(
            [
                [0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.],
                [1., -4., 6., -4., 1., 0., 0.],
                [0., 1., -4., 6., -4., 1., 0.],
                [0., 0., 1., -4., 6., -4., 1.],
                [0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.],
            ]
        )

        assert_allclose(expected, actual)

    @staticmethod
    def _build_dynamic_element(data):
        null_stencil = Stencil({})

        def build_stencil(p):
            return data.get(p, null_stencil)
        return DynamicElement(build_stencil)

    @staticmethod
    def create(*points):
        variables = create_variables(*points)
        return fdm.analysis.down_to_up.MatrixBuilder(variables or create_variables())


class VectorBuilderTest(unittest.TestCase):
    def test_Get_DoNothing_ReturnZeroVectorOfVariableLengthSize(self):
        points = Point(0.), Point(1.), Point(2.)
        builder = self.create(*points)

        actual = builder.get()

        expected = numpy.zeros((3, 1))

        assert_allclose(expected, actual)

    def test_Apply_Function_ReturnArrayWithValuesComputedWithFunction(self):
        points = Point(4.), Point(1.), Point(2.)
        builder = self.create(*points)

        def calculator(p):
            return p.x

        builder.apply(calculator)
        actual = builder.get()

        expected = numpy.array(
            [
                [4.],
                [1.],
                [2.],
            ]
        )

        assert_allclose(expected, actual)

    def test_Apply_NodesGiven_ReturnArrayWithValuesComputedForGivenPoints(self):
        points = p1, p2, p3 = Point(4.), Point(1.), Point(2.)
        builder = self.create(*points)

        def calculator(p):
            return p.x

        builder.apply(calculator, nodes=[p2])
        actual = builder.get()

        expected = numpy.array(
            [
                [0.],
                [1.],
                [0.],
            ]
        )

        assert_allclose(expected, actual)

    @staticmethod
    def create(*points):
        variables = create_variables(*points)
        return fdm.analysis.down_to_up.VectorBuilder(variables or create_variables())


class DistributeSchemeToNodesTest(unittest.TestCase):
    def test_AllInNodes_Always_ReturnTheSame(self):
        nodes = p1, p2, p3 = (Point(0.), Point(1.), Point(2.))
        scheme = Scheme({p1: 1., p2: 2., p3: 3.})

        actual = self._distribute(nodes, scheme)

        expected = Scheme({p1: 1., p2: 2., p3: 3.})

        self.assertEqual(expected, actual)

    def test_MiddleNode_AtCenter_SplitWeightToAdjacentNodes(self):
        nodes = p1, p2 = (Point(0.), Point(1.))
        scheme = Scheme({Point(0.5): 2.})

        actual = self._distribute(nodes, scheme)

        expected = Scheme({p1: 1., p2: 1.})

        self.assertEqual(expected, actual)

    def test_MiddleNode_CloseToTheNode_SplitWeightProportionally(self):
        nodes = p1, p2 = (Point(0.), Point(1.))
        scheme = Scheme({Point(0.75): 4.})

        actual = self._distribute(nodes, scheme)

        expected = Scheme({p1: 1., p2: 3.})

        self.assertEqual(expected, actual)

    @staticmethod
    def _distribute(nodes, scheme):
        return fdm.analysis.down_to_up.distribute_scheme_to_nodes(nodes, scheme)


def create_variables(*points):
    return fdm.analysis.analyzer.create_variables(points)