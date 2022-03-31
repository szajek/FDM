import itertools
import unittest

import numpy
from numpy.testing import assert_allclose

import fdm.analysis
from fdm import Point, create_close_point_finder, Scheme
from fdm.analysis.tools import create_weights_distributor
from fdm.builder import static_boundary, dynamic_boundary


class WeightDistributorTest(unittest.TestCase):
    def test_Distribute_CoincidentWithNode_ReturnDictWithValues(self):
        points = p1, p2 = [Point(0.), Point(1.)]
        close_point_finder = create_close_point_finder(points)

        distributor = self._create(close_point_finder)

        result = distributor(p1, 1.)

        expected = {p1: 1., p2: 0.}

        self.assertEqual(expected, result)

    def test_Distribute_BetweenNodes_ReturnDictWithValues(self):
        points = p1, p2 = Point(1.), Point(2.)
        middle_point = Point(1.2)
        indexed_points = create_close_point_finder(points)

        distributor = self._create(indexed_points)

        result = distributor(middle_point, 1.)

        expected = {p1: .8, p2: 0.2}

        self.assertTrue(expected.keys() == result.keys())
        self.assertTrue([(expected[k] - result[k]) < 1e-6 for k in expected.keys()])

    def test_Distribute_LastNode_ReturnDictWithValues(self):
        points = p1, p2 = Point(1.), Point(2.)
        indexed_points = create_close_point_finder(points)

        distributor = self._create(indexed_points)

        result = distributor(p2, 1.)

        expected = {p1: 0., p2: 1.}

        self.assertEqual(expected, result)

    def _create(self, *args, **kwargs):
        return create_weights_distributor(*args, **kwargs)


class ApplyStaticsBCTest(unittest.TestCase):
    def test_TwoBCs_OneReplaced_ReturnMatrixOfExtendedRowsNumberByOne(self):
        points = p1, p2, p3, p4 = Point(0.), Point(1.), Point(2.), Point(3.)
        base = numpy.zeros((4, 4))
        bcs = _bc_stat(Scheme({})), _bc_stat(Scheme({}), replace=p1)

        result, _ = self.create(points, matrix=base, bcs=bcs)
        actual = result.shape[0]

        self.assertEqual(4, actual)

    def test_NoneBC_Matrix_ReturnBaseMatrix(self):
        points = Point(0.), Point(1.), Point(2.)
        base = numpy.array(
            [
                [3., 2., 0.],
                [0., 0., 3.],
                [5., 7., 0.],
            ]
        )

        actual, _ = self.create(points, matrix=base, bcs=())

        expected = base

        assert_allclose(expected, actual)

    def test_NoneBC_Vector_ReturnBaseVector(self):
        points = Point(0.), Point(1.), Point(2.)
        base = numpy.array(
            [3., 0., 5.]
        )

        _, actual = self.create(points, virtual_points=[], vector=base, bcs=())

        expected = base

        assert_allclose(expected, actual)

    def test_OneBC_Matrix_SetExpandedAtTheLastRow(self):
        points = p1, p2, p3, p4 = Point(0.), Point(1.), Point(2.), Point(3.)
        vp1 = Point(3.)
        base = numpy.array(
            [
                [3., 2., 0., 1.],
                [0., 0., 3., 2.],
                [5., 7., 0., 3.],
                [1., 1., 1., 1.],
            ]
        )
        bcs = [_bc_stat(Scheme({p1: -1., p3: 1.}))]

        actual, _ = self.create(points, virtual_points=[vp1], matrix=base, bcs=bcs)

        expected = numpy.array(
            [
                [3., 2., 0., 1.],
                [0., 0., 3., 2.],
                [5., 7., 0., 3.],
                [-1., 0., 1., 0.],
            ]
        )

        assert_allclose(expected, actual)

    def test_OneBC_MatrixAndReplaceAndPointSlightlyOutside_ReplaceRow(self):
        points = p1, p2, p3, p4 = Point(0.), Point(1.), Point(2.), Point(3.)
        slightly_outside_point = Point(2.00000001)
        base = numpy.array(
            [
                [3., 2., 0.],
                [0., 0., 3.],
                [5., 7., 0.],
            ]
        )
        bcs = [_bc_stat(Scheme({p1: -99., slightly_outside_point: 99.}), replace=p3)]

        actual, _ = self.create(points, matrix=base, bcs=bcs)

        expected = numpy.array(
            [
                [3., 2., 0.],
                [0., 0., 3.],
                [-99., 0., 99.],
            ]
        )

        assert_allclose(expected, actual)

    def test_OneBC_Vector_AddExpandedAtTheLastRow(self):
        points = Point(0.), Point(1.), Point(2.), Point(3.)
        base = numpy.zeros(4)
        bcs = [_bc_stat(value=4.)]

        _, actual = self.create(points, vector=base, bcs=bcs)

        expected = numpy.array(
            [0., 0., 0., 4.]
        )

        assert_allclose(expected, actual)

    def test_OneBC_VectorAndReplace_ReplaceGivenEquation(self):
        points = _, p2, _ = Point(0.), Point(1.), Point(2.)
        base = numpy.zeros(3)
        bcs = [_bc_stat(value=4., replace=p2)]

        _, actual = self.create(points, vector=base, bcs=bcs)

        expected = numpy.array(
            [0., 4., 0.]
        )

        assert_allclose(expected, actual)

    def test_TwoBCs_Matrix_AddExpandedAtTheLastRows(self):
        points = p1, p2, p3, p4, p5 = Point(0.), Point(1.), Point(2.), Point(3.), Point(4.)
        base = numpy.zeros((5, 5))
        bcs = [_bc_stat(Scheme({p1: -1., p3: 1.})), _bc_stat(Scheme({p2: 2.}))]

        actual, _ = self.create(points, matrix=base, bcs=bcs)

        expected = numpy.array(
            [
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [-1., 0., 1., 0., 0.],
                [0., 2., 0., 0., 0.],
            ]
        )

        assert_allclose(expected, actual)

    def test_TwoBCs_OneReplace_SetOneAtLastRowAndOneReplace(self):
        points = p1, p2, p3, p4 = Point(0.), Point(1.), Point(2.), Point(3.)
        base = numpy.array(
            [
                [0., 0., 0., 0.],
                [9., 9., 9., 9.],
                [0., 0., 0., 0.],
                [1., 1., 1., 1.],
            ]
        )
        bcs = [_bc_stat(Scheme({p1: -1., p3: 1.})), _bc_stat(Scheme({p2: 2.}), replace=p2)]

        actual, _ = self.create(points, matrix=base, bcs=bcs)

        expected = numpy.array(
            [
                [0., 0., 0., 0.],
                [0., 2., 0., 0.],
                [0., 0., 0., 0.],
                [-1., 0., 1., 0.],
            ]
        )

        assert_allclose(expected, actual)

    def test_TwoBCs_Vector_AddExpandedAtTheLastRows(self):
        points = Point(0.), Point(1.), Point(2.), Point(3.), Point(4.)
        base = numpy.zeros(5)
        bcs = [_bc_stat(value=3.), _bc_stat(value=4.)]

        _, actual = self.create(points, vector=base, bcs=bcs)

        expected = numpy.array(
            [0., 0., 0., 3., 4.]
        )

        assert_allclose(expected, actual)

    def test_TwoBCs_VectorOneReplace_AddOneAtLastRowAndOneReplace(self):
        points = _, p2, _, _ = Point(0.), Point(1.), Point(2.), Point(3.)
        base = numpy.array([1., 2., 3., 0])
        bcs = [_bc_stat(value=3.), _bc_stat(value=4., replace=p2)]

        _, actual = self.create(points, vector=base, bcs=bcs)

        expected = numpy.array(
            [1., 4., 3., 3.]
        )

        assert_allclose(expected, actual)

    def test_OneBC_Midpoint_AddDistributedWeights(self):
        points = Point(0.), Point(1.), Point(2.), Point(3.)
        base = numpy.zeros((4, 4))
        bcs = [_bc_stat(Scheme({Point(0.5): -1., Point(1.5): 1.}))]

        actual, _ = self.create(points, matrix=base, bcs=bcs)

        expected = numpy.array(
            [
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [-0.5, 0., 0.5, 0.],
            ]
        )

        assert_allclose(expected, actual)

    @staticmethod
    def create(points, virtual_points=None, matrix=None, vector=None, bcs=()):
        extra_bcs = [bc for bc in bcs if bc.replace is None]
        all_points = tuple(itertools.chain(points, virtual_points or ()))
        variables = create_variables(*all_points)
        matrix = numpy.zeros((len(points), len(variables) + len(extra_bcs))) if matrix is None else matrix
        vector = numpy.zeros(len(points)) if vector is None else vector
        return fdm.analysis.tools.apply_statics_bc(variables, matrix, vector, bcs)


class ApplyDynamicsBCTest(unittest.TestCase):
    def test_NoneBCs_Always_ReturnTheSameMatrices(self):
        points = Point(0.), Point(1.), Point(2.)
        matrix_a = numpy.zeros((3, 3))
        matrix_b = numpy.zeros((3, 3))
        bcs = []

        A, B = self.apply(points, matrix_a=matrix_a, matrix_b=matrix_b, bcs=bcs)

        assert_allclose(matrix_a, A)
        assert_allclose(matrix_b, B)

    def test_SingleBCs_Always_AddBCAsLastRow(self):
        points = p1, p2, p3 = Point(0.), Point(1.), Point(2.)
        matrix_a = numpy.zeros((3, 3))
        matrix_b = numpy.zeros((3, 3))
        bcs = [_bc_dyn(Scheme({p1: 1.}), Scheme({p3: 2.}))]

        actual_A, actual_B = self.apply(points, matrix_a=matrix_a, matrix_b=matrix_b, bcs=bcs)
        expected_A = [
            [0., 0., 0.],
            [0., 0., 0.],
            [1., 0., 0.],
        ]
        expected_B = [
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 2.],
        ]

        assert_allclose(expected_A, actual_A)
        assert_allclose(expected_B, actual_B)

    @staticmethod
    def apply(points, virtual_points=None, matrix_a=None, matrix_b=None, bcs=()):
        all_points = tuple(itertools.chain(points, virtual_points or ()))
        variables = create_variables(*all_points)
        A = numpy.zeros((len(points), len(variables))) if matrix_a is None else matrix_a
        B = numpy.zeros((len(points), len(variables))) if matrix_b is None else matrix_b
        return fdm.analysis.tools.apply_dynamics_bc(variables, A, B, bcs)


def _bc_stat(scheme=None, value=0., replace=None):
    scheme = scheme or Scheme({})
    return static_boundary(scheme, value, replace)


def _bc_dyn(scheme_a=None, scheme_b=None):
    return static_boundary(scheme_a or Scheme({}), scheme_b or Scheme({}))


def create_variables(*points):
    return fdm.analysis.analyzer.create_variables(points)


class SchemeToNodesDistributorTest(unittest.TestCase):
    def test_AllInNodes_Always_ReturnTheSame(self):
        nodes = p1, p2, p3 = (Point(0.), Point(1.), Point(2.))
        scheme = Scheme({p1: 1., p2: 2., p3: 3.})

        distributor = self.create(nodes)
        actual = distributor(scheme)

        expected = Scheme({p1: 1., p2: 2., p3: 3.})

        self.assertEqual(expected, actual)

    def test_MiddleNode_AtCenter_SplitWeightToAdjacentNodes(self):
        nodes = p1, p2 = (Point(0.), Point(1.))
        scheme = Scheme({Point(0.5): 2.})

        distributor = self.create(nodes)
        actual = distributor(scheme)

        expected = Scheme({p1: 1., p2: 1.})

        self.assertEqual(expected, actual)

    def test_MiddleNode_CloseToTheNode_SplitWeightProportionally(self):
        nodes = p1, p2 = (Point(0.), Point(1.))
        scheme = Scheme({Point(0.75): 4.})

        distributor = self.create(nodes)
        actual = distributor(scheme)

        expected = Scheme({p1: 1., p2: 3.})

        self.assertEqual(expected, actual)

    @staticmethod
    def create(nodes):
        return fdm.analysis.tools.SchemeToNodesDistributor(nodes)
