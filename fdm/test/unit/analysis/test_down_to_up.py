import unittest
import numpy
from numpy.testing import assert_allclose

from fdm.geometry import Point
from fdm.equation import Stencil, DynamicElement, Scheme
import fdm.analysis.down_to_up
import fdm.analysis.analyzer


class ArrayBuilderTest(unittest.TestCase):
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

    @staticmethod
    def _build_dynamic_element(data):
        null_stencil = Stencil({})

        def build_stencil(p):
            return data.get(p, null_stencil)
        return DynamicElement(build_stencil)

    @staticmethod
    def create(*points):
        variables = create_variables(*points)
        return fdm.analysis.down_to_up.ArrayBuilder(variables or create_variables())


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