import math
import unittest

import numpy as np
from mock import MagicMock

from fdm.equation import (
    LazyOperation, Operator, Stencil, DynamicElement, Scheme, Number, Element, operate, merge_weights, MutateMixin
)
from fdm.geometry import FreeVector, Point, Vector


def _are_the_same_objects(obj, mutated):
    return obj is mutated


class MutateMixinTest(unittest.TestCase):

    class PublicFieldsExample(MutateMixin):
        def __init__(self, field_1, field_2):
            self.field_1 = field_1
            self.field_2 = field_2

            MutateMixin.__init__(self, 'field_1', 'field_2')

    class PrivateFieldsExample(MutateMixin):
        def __init__(self, field_1, field_2):

            self._field_1 = field_1
            self._field_2 = field_2

            MutateMixin.__init__(self, ('field_1', '_field_1'), ('field_2', '_field_2'))

    def test_Mutate_PublicField_ReturnNewObjectWithMutatedField(self):
        obj = self.PublicFieldsExample('field_1_value', 'field_2_value')

        mutated = obj.mutate(field_1='new')

        self.assertFalse(_are_the_same_objects(obj, mutated))
        self.assertEqual('new', mutated.field_1)
        self.assertEqual('field_2_value', mutated.field_2)

    def test_Mutate_PrivateField_ReturnNewObjectWithMutatedField(self):
        obj = self.PrivateFieldsExample('field_1_value', 'field_2_value')

        mutated = obj.mutate(field_1='new')

        self.assertFalse(_are_the_same_objects(obj, mutated))
        self.assertEqual('new', mutated._field_1)
        self.assertEqual('field_2_value', mutated._field_2)


class ToCoefficientsTest(unittest.TestCase):  # todo:
    def _test_ToCoefficients_Always_ReturnWeightsConsideringDeltaAndOrder(self):

        order = 2.
        weight = 1.
        delta = 2.
        scheme = Scheme({1: weight}, order)

        result = scheme.to_coefficients(delta)

        expected = {1: weight/delta**order}

        self.assertEquals(expected, result)

    def _test_ToCoefficients_PositiveMidNode_SpreadEquallyWeightsToAdjacentNodes(self):

        scheme = Scheme({0.5: 1})

        result = scheme.to_coefficients(1.)

        expected = {0: 0.5, 1: 0.5}

        self.assertEquals(expected, result)

    def _test_ToCoefficients_NegativeMidNode_SpreadEquallyWeightsToAdjacentNodes(self):

        scheme = Scheme({-0.5: 1})

        result = scheme.to_coefficients(1.)

        expected = {0: 0.5, -1: 0.5}

        self.assertEquals(expected, result)

    def _test_ToCoefficients_NegativeFloatNodeAddress_SpreadProportionallyWeightsToAdjacentNodes(self):

        scheme = Scheme({-0.25: 1.})

        result = scheme.to_coefficients(1.)

        expected = {0: 0.75, -1: 0.25}

        self.assertEquals(expected, result)

    def _test_ToCoefficients_PositiveFloatNodeAddress_SpreadProportionallyWeightsToAdjacentNodes(self):
        scheme = Scheme({0.75: 1})

        result = scheme.to_coefficients(1.)

        expected = {0: 0.25, 1: 0.75}

        self.assertEquals(expected, result)


class SchemeTest(unittest.TestCase):

    def test_Create_PointsAsIntegersOfFloat_ConvertToPoints(self):

        scheme = Scheme({Point(1): 1., Point(2.): 1.})

        result = set(scheme._weights.keys())

        expected = {Point(1), Point(2)}

        self.assertEqual(expected, result)

    def test_Equal_Always_CompareDataAndOrder(self):
        self.assertEqual(
            self._build_scheme((1., 2., 3.)),
            self._build_scheme((1., 2., 3.))
        )
        self.assertNotEqual(
            self._build_scheme((1., 2., 3.)),
            self._build_scheme((1., 2., 4.))
        )
        self.assertNotEqual(
            self._build_scheme((1., 2., 3.)),
            self._build_scheme((1., 2.))
        )

    def test_Duplicate_Always_ReturnCopiedObject(self):
        s1 = Scheme({1: 2, 3: 1.})

        result = s1.duplicate()

        self.assertEqual(s1, result)
        self.assertFalse(_are_the_same_objects(s1, result))

    def test_Add_None_ReturnUnchangedScheme(self):
        s1 = Scheme({1: 2, 3: 1.})
        s2 = None

        result = s1 + s2

        self.assertEqual(s1, result)

    def test_RightAdd_None_ReturnUnchangedScheme(self):
        s1 = None
        s2 = Scheme({1: 2, 3: 1.})

        result = s1 + s2

        self.assertEqual(s2, result)

    def test_Add_NoNodesSetsIntersection_MergeData(self):
        s1 = Scheme({1: 2})
        s2 = Scheme({2: 3})

        result = s1 + s2

        expected = Scheme({1: 2, 2: 3})

        self.assertEqual(expected, result)

    def test_Add_NodesSetsIntersection_MergeForUniqueAndAddForIntersection(self):
        s1 = Scheme({1: 2, 3: 1.})
        s2 = Scheme({2: 3, 3: 2.})

        result = s1 + s2

        expected = Scheme({1: 2, 2: 3, 3: 3})

        self.assertEqual(expected, result)

    def test_Shift_Always_TranslatePoints(self):

        scheme = Scheme({Point(0): 1., Point(3.): -3})

        result = scheme.shift(FreeVector(Point(-1.)))

        expected = Scheme({Point(-1): 1., Point(2.): -3})

        self.assertEquals(result, expected)

    def test_LeftMultiplication_IntegerOrFloat_MultiplyWeights(self):

        scheme = self._build_scheme(weights=(0., 2., 3.))

        result = 2.*scheme

        expected = self._build_scheme(weights=(0., 4., 6.))

        self.assertEqual(expected, result)

    def test_LeftMultiplication_NumpyFloat64_MultiplyWeightsAndReturnArray(self):
        """
        Left multiplication by numpy float should be prevent cause leads to an array
        """

        scheme = Scheme({Point(1): 1, Point(2): 1})

        result = np.float64(2.)*scheme

        expected = np.array([Point(2.), Point(4.)])

        np.testing.assert_array_equal(expected, result)

    def test_RightMultiplication_IntegerOrFloat_MultiplyWeights(self):

        scheme = self._build_scheme(weights=(0., 2., 3.))

        result = scheme*2.

        expected = self._build_scheme(weights=(0., 4., 6.))

        self.assertEqual(expected, result)

    def test_Power_IntegerOrFloat_RaiseWeightsToGivenPower(self):

        scheme = self._build_scheme(weights=(0., 2., 3.))

        result = scheme**2

        expected = self._build_scheme(weights=(0., 4., 9.))

        self.assertEqual(expected, result)

    def test_Start_Always_ReturnPointTheClosestToNegativeUtopia(self):

        scheme = Scheme({Point(-3.): 1., Point(-5.): 0.})

        result = scheme.start

        expected = Point(-5)

        self.assertEqual(expected, result)

    def test_End_Always_ReturnPointTheFarthestPointFromNegativeUtopia(self):
        scheme = Scheme({Point(-3.): 1., Point(-2.): 0.})

        result = scheme.end

        expected = Point(-2)

        self.assertEqual(expected, result)

    def test_ToValue_Always_ReturnValueCalculatedBasedOnOutput(self):

        scheme = Scheme({Point(1): 1.3, Point(2): 2.3})
        output = {Point(1): 2.1, Point(2): 1.1}

        result = scheme.to_value(output)

        expected = 1.3 * 2.1 + 2.3 * 1.1

        self.assertEqual(expected, result)

    def test_Distribute_Always_ReturnSchemeWithPointsInNodesAndDistributedWeights(self):
        scheme = Scheme({Point(.5): 1., Point(1.5): 2.})

        def distributor(point, value):
            if point == Point(.5):
                return {Point(0.): 0.5 * value, Point(1.): 0.5 * value}
            elif point == Point(1.5):
                return {Point(1.): 0.5 * value, Point(2.): 0.5 * value}
            else:
                raise NotImplementedError

        result = scheme.distribute(distributor)

        expected = Scheme({
            Point(0.): 0.5,
            Point(1.): 0.5 + 1.0,
            Point(2.): 1.,
        })

        self.assertEqual(expected, result)

    def test_Drop_AllElementsBiggerThanTol_ReturnTheSame(self):
        s = Scheme({Point(1.): 2.})

        result = s.drop(0.1)

        expected = Scheme({Point(1.): 2.})

        self.assertEqual(expected, result)

    def test_Drop_OneElementsSmallerThanTol_ReturnWithoutSmallWeight(self):
        s = Scheme({Point(1.): 2., Point(2.): 0.05})

        result = s.drop(0.1)

        expected = Scheme({Point(1.): 2.})

        self.assertEqual(expected, result)

    def _build_scheme(self, weights):
        return Scheme(
            {i: d for i, d in enumerate(weights)},
        )


class MergeWeightsTest(unittest.TestCase):
    def test_Call_NoIntersection_ReturnDictWithElementsFromBoth(self):

        weights_1 = {Point(1): 1}
        weights_2 = {Point(2): 2}

        result = merge_weights(weights_1, weights_2)

        expected = {Point(1): 1, Point(2): 2}

        self.assertEqual(expected, result)

    def test_Call_TheSameNodeAddresses_ReturnDictWitSummedValues(self):
        weights_1 = {Point(1): 1}
        weights_2 = {Point(1): 2}

        result = merge_weights(weights_1, weights_2)

        expected = {Point(1): 3}

        self.assertEqual(expected, result)

    def test_Call_Intersection_ReturnDictWithElementsFromBothAndSummedWeightsForSharedAddresses(self):
        weights_1 = {Point(1): 1, Point(3): 1}
        weights_2 = {Point(2): 2, Point(3): 2}

        result = merge_weights(weights_1, weights_2)

        expected = {Point(1): 1, Point(2): 2, Point(3): 3}

        self.assertEqual(expected, result)

    def test_Call_ManyGiven_ReturnMerged(self):
        weights_1 = {Point(1): 1}
        weights_2 = {Point(2): 2}
        weights_3 = {Point(3): 3}

        result = merge_weights(weights_1, weights_2, weights_3)

        expected = {Point(1): 1, Point(2): 2, Point(3): 3}

        self.assertEqual(expected, result)

    def test_Call_ClosePoints_ReturnMerged(self):
        weights_1 = {Point(1 + 1e-9): 1}
        weights_2 = {Point(1): 2}

        result = merge_weights(weights_1, weights_2)

        expected = {Point(1): 3}

        self.assertEqual(expected, result)


class OperateTest(unittest.TestCase):
    def test_Call_WithNone_ReturnTheSameScheme(self):
        scheme = Scheme({1: 2})

        result = operate(scheme, None)

        self.assertEqual(scheme, result)

    def test_Call_WithEmptySchemeOrElement_RaiseAttributeError(self):
        scheme = Scheme({Point(1): 1})
        element = Stencil({Point(1): 1})

        with self.assertRaises(AttributeError):
            operate(Scheme({}), element)

        with self.assertRaises(AttributeError):
            operate(scheme, Stencil({}))

    def test_Call_OneNodeScheme_ReturnRevolvedScheme(self):
        scheme = Scheme({1: 2})

        element = MagicMock(
            expand=lambda node_idx: Scheme({0: 1., 5.: 2.})
        )

        result = operate(scheme, element)

        expected = Scheme({0: 1. * 2., 5.: 2. * 2.})

        self.assertEqual(expected, result)

    def test_Call_WithSchemeVariedByNodeAddress_ReturnRevolvedScheme(self):
        scheme = Scheme({Point(1): 2, Point(2): 4})

        element = MagicMock(
            expand=lambda point: {
                Point(1): Scheme({Point(0): 1., Point(6.): 3.}),
                Point(2): Scheme({Point(0): 1., Point(5.): 2.}),
            }[point]
        )

        result = operate(scheme, element)

        expected = Scheme({Point(0): 1. * 2 + 1. * 4., Point(5.): 2. * 4., Point(6): 3. * 2.})

        self.assertEqual(expected, result)


#


class ElementTest(unittest.TestCase):
    class ConcreteElement(Element):
        def expand(self):
            return Scheme({1: 1})

    def test_Add_TwoElements_ReturnLazySummation(self):
        e1 = self.ConcreteElement()
        e2 = self.ConcreteElement()

        result = e1 + e2

        expected = LazyOperation.summation(e1, e2)

        self.assertEquals(expected, result)

    def test_Multiplication_TwoElements_ReturnLazyMultiplication(self):
        e1 = self.ConcreteElement()
        e2 = self.ConcreteElement()

        result = e1 * e2

        expected = LazyOperation.multiplication(e1, e2)

        self.assertEquals(expected, result)

    def test_Subtraction_TwoElements_ReturnLazySubtraction(self):
        e1 = self.ConcreteElement()
        e2 = self.ConcreteElement()

        result = e1 - e2

        expected = LazyOperation.subtraction(e1, e2)

        self.assertEquals(expected, result)

    def test_Division_TwoElements_ReturnLazyDivision(self):
        e1 = self.ConcreteElement()
        e2 = self.ConcreteElement()

        result = e1 / e2

        expected = LazyOperation.division(e1, e2)

        self.assertEquals(expected, result)

    def test_Division_TwoElements_ReturnLazyDivision(self):
        e1 = self.ConcreteElement()
        e2 = self.ConcreteElement()

        result = e1 ** e2

        expected = LazyOperation.power(e1, e2)

        self.assertEquals(expected, result)

    def test_ToStencil_Origin_CreateStencilByExpansionForOrigin(self):
        stencil = Stencil.central(1.)
        operator = Operator(stencil,
                            Operator(stencil)
                            )

        result = operator.to_stencil(Point(0.))

        expected = Stencil({Point(-1.): 1., Point(0.): -2., Point(1.): 1.})

        self.assertEqual(expected, result)

    def test_ToStencil_NotOrigin_CreateStencilByExpansionForAddressAndMoveToOrigin(self):
        stencil = Stencil.central(1.)
        operator = Operator(stencil,
                            Operator(stencil)
                            )

        result = operator.to_stencil(Point(2.))

        expected = Stencil({Point(-1.): 1., Point(0.): -2., Point(1.): 1.})

        self.assertEqual(expected, result)


class StencilTest(unittest.TestCase):
    def test_FromScheme_Always_CreateUsingWeightsAndOrderFromScheme(self):

        weights, order = {1: -2}, 3
        scheme = Scheme(weights)

        result = Stencil.from_scheme(scheme)

        expected = Stencil(weights)

        self.assertEqual(expected, result)

    def test_Uniform_Always_GenerateUniformlyDistributedNodesInGivenLimits(self):
        point_1, point_2 = Point(-1.), Point(2.)
        resolution = 3.

        _stencil = Stencil.uniform(point_1, point_2, resolution, lambda *args: None)
        result = list(sorted([point for point in _stencil._weights], key=lambda item: item.x))

        _delta = Vector(point_1, point_2).length / resolution
        expected_node_addresses = [point_1 + FreeVector(Point(_delta * i)) for i in range(4)]

        self.assertEquals(expected_node_addresses, result)

    def test_Central_RangeOne_GenerateWeightsEqualOneAssignedToAdjacentMidnodes(self):

        _scheme = Stencil.central(1.)
        result = _scheme._weights

        _scheme = Stencil({Point(-0.5): -1., Point(0.5): 1.})
        expected = _scheme._weights

        self.assertTrue(self._compare_dict(expected, result))

    def test_Central_RangeTwo_GenerateWeightsEqualHalfAssignedToAdjacentNodes(self):

        _scheme = Stencil.central(2.)
        result = _scheme._weights

        _scheme = Stencil({Point(-1.): -0.5, Point(1.): 0.5})
        expected = _scheme._weights

        self.assertTrue(self._compare_dict(expected, result,))

    def test_Eq_Always_CheckWeights(self):
        self.assertEqual(
            Stencil({-3.: -6.5, 3.: 2.5}),
            Stencil({-3.: -6.5, 3.: 2.5})
        )
        self.assertNotEqual(
            Stencil({-1.: -6.5, 3.: 2.5}),
            Stencil({-3.: -6.5, 3.: 2.5})
        )

    def test_Scale_Always_ScaleWeightsAddresses(self):

        stencil = Stencil({-3.: -6.5, 3.: 2.5})

        result = stencil.scale(2.)

        expected = Stencil({-6.: -6.5, 6.: 2.5})

        self.assertEqual(expected, result)

    def test_Symmetry_Always_ReturnMirroredStencil(self):
        s = Stencil({Point(1.): 2.})

        result = s.symmetry(Point(2.))

        expected = Stencil({Point(3.): 2.})

        self.assertEqual(expected, result)

    def test_Multiply_Always_ReturnStencilWithMultipliedWeights(self):
        s = Stencil({Point(1.): 2.})

        result = s.multiply(3.)

        expected = Stencil({Point(1.): 6.})

        self.assertEqual(expected, result)

    def test_Drop_AllElementsBiggerThanTol_ReturnTheSame(self):
        s = Stencil({Point(1.): 2.})

        result = s.drop(0.1)

        expected = Stencil({Point(1.): 2.})

        self.assertEqual(expected, result)

    def test_Drop_OneElementsSmallerThanTol_ReturnWithoutSmallWeight(self):
        s = Stencil({Point(1.): 2., Point(2.): 0.05})

        result = s.drop(0.1)

        expected = Stencil({Point(1.): 2.})

        self.assertEqual(expected, result)

    def _compare_dict(self, d1, d2, tol=1e-4):
        return len(d1) == len(d2) and all(math.fabs(d1[k] - d2[k]) < tol for k in d1.keys())


class DynamicElementTest(unittest.TestCase):
    def _test_Expand_Always_BuildStencilForGivenAddress(self):
        expected_weights = {4.: 9.}
        expected_order = 4.

        def builder(node_address):
            return Stencil(expected_weights, order=expected_order)

        dynamic_stencil = DynamicElement(builder)

        result = dynamic_stencil.expand(0.)

        expected = Scheme(expected_weights, expected_order)

        self.assertEqual(expected, result)


class LazyOperationTest(unittest.TestCase):
    def test_Summation_Always_CallAddMagicForLeftAddendScheme(self):
        scheme_1 = MagicMock()
        scheme_2 = MagicMock()

        addend_1 = MagicMock(
            expand=MagicMock(return_value=scheme_1)
        )
        addend_2 = MagicMock(
            expand=MagicMock(return_value=scheme_2)
        )

        op = LazyOperation.summation(addend_1, addend_2)
        op.expand()

        scheme_1.__add__.assert_called_once()
        scheme_2.__add__.assert_not_called()

    def test_Subtraction_Always_CallSubMagicForMinuendScheme(self):
        scheme_minuend = MagicMock()
        scheme_subtrahend = MagicMock()

        minuend = MagicMock(
            expand=MagicMock(return_value=scheme_minuend)
        )
        subtrahend = MagicMock(
            expand=MagicMock(return_value=scheme_subtrahend)
        )

        op = LazyOperation.subtraction(minuend, subtrahend)
        op.expand()

        scheme_minuend.__sub__.assert_called_once()
        scheme_subtrahend.__sub__.assert_not_called()

    def test_Multiplication_Always_CallMultMagicForLeftFactorScheme(self):
        scheme_factor_1 = MagicMock()
        scheme_factor_2 = MagicMock()

        factor_1 = MagicMock(
            expand=MagicMock(return_value=scheme_factor_1)
        )
        factor_2 = MagicMock(
            expand=MagicMock(return_value=scheme_factor_2)
        )

        op = LazyOperation.multiplication(factor_1, factor_2)
        op.expand()

        scheme_factor_1.__mul__.assert_called_once()
        scheme_factor_2.__mul__.assert_not_called()

    def test_Division_Always_CallDivMagicForDividendScheme(self):
        scheme_dividend = MagicMock()
        scheme_divisor = MagicMock()

        dividend = MagicMock(
            expand=MagicMock(return_value=scheme_dividend)
        )
        divisor = MagicMock(
            expand=MagicMock(return_value=scheme_divisor)
        )

        op = LazyOperation.division(dividend, divisor)
        op.expand()

        scheme_dividend.__truediv__.assert_called_once()
        scheme_divisor.__truediv__.assert_not_called()

    def test_Power_Always_CallPowMagicForBaseScheme(self):
        scheme_base = MagicMock()
        scheme_exponent = MagicMock()

        base = MagicMock(
            expand=MagicMock(return_value=scheme_base)
        )
        exponent = MagicMock(
            expand=MagicMock(return_value=scheme_exponent)
        )

        op = LazyOperation.power(base, exponent)
        op.expand()

        scheme_base.__pow__.assert_called_once()
        scheme_exponent.__pow__.assert_not_called()

    def test_Operation_NumpyFloat64ForFirstElement_RaiseAttributeError(self):

        scheme_factor_2 = MagicMock()

        factor_1 = MagicMock(
            expand=MagicMock(return_value=np.float64(2.))
        )
        factor_2 = MagicMock(
            expand=MagicMock(return_value=scheme_factor_2)
        )

        op = LazyOperation.multiplication(factor_1, factor_2)

        with self.assertRaises(AttributeError):
            op.expand()


class OperatorTest(unittest.TestCase):

    def test_Expand_NoElement_ReturnStencilSchemeShiftedToNodeAddress(self):

        stencil_weights = {Point(-1): 1., Point(1): 1.}
        stencil = Stencil(stencil_weights)
        operator = Operator(stencil, element=None)

        result = operator.expand(Point(-2))

        expected = Scheme(stencil_weights).shift(FreeVector(Point(-2)))

        self.assertEqual(expected, result)

    def test_Expand_ElementAsDispatcher_UseDynamicElementByLocal(self):
        stencil = Stencil({Point(-2): 1, Point(-1): 1, Point(1): 3, Point(2): 1})

        element_start = MagicMock(
            expand=MagicMock(return_value=Scheme({Point(-10): 1.}))
        )
        element_center = MagicMock(
            expand=MagicMock(return_value=Scheme({Point(0): 2.}))
        )
        element_end = MagicMock(
            expand=MagicMock(return_value=Scheme({Point(10): 3.}))
        )

        def dispatcher(start, end, position):
            return {
                start: element_start,
                end: element_end,
            }.get(position, element_center)

        dispatcher = Operator(stencil, dispatcher)

        result = dispatcher.expand(Point(3))

        expected = Scheme({Point(-10): 1., Point(0): 2.*1. + 2.*3., Point(10): 3.})

        self.assertEqual(expected, result)

    def _build_operator(self, *data):
        return Operator(
            {i: d for i, d in enumerate(data)}
        )


class NumberTest(unittest.TestCase):
    def test_Expand_Float_ReturnFloat(self):

        value = 3.
        number = Number(value)

        result = number.expand(1)

        expected = value

        self.assertEquals(expected, result)

    def test_Expand_Callable_ReturnValueComputedByCallable(self):

        value = 999.
        number = Number(lambda node_address: value)

        result = number.expand(1)

        expected = value

        self.assertEquals(expected, result)


#


class TemplateTest(unittest.TestCase):
    pass