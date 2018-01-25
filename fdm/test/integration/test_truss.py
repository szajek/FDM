import unittest

import numpy as np

import fdm.builder as builder
from fdm.analysis import solve, AnalysisType


class TrussStaticEquationFiniteDifferencesTest(unittest.TestCase):
    def setUp(self):
        self._length = 1.
        self._node_number = 6

    def test_ConstantSectionAndYoungModulus_ReturnCorrectDisplacement(self):
        model = (
            self._create_predefined_builder()
                .set_field(builder.FieldType.LINEAR, a=1.)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [
                [0.],
                [0.08],
                [0.152],
                [0.208],
                [0.24],
                [0.24],
            ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-6)

    def test_VariedSection_ReturnCorrectDisplacement(self):
        def young_modulus(point):
            return 2. - (point.x / self._length) * 1.

        model = (
            self._create_predefined_builder()
                .set_field(builder.FieldType.CONSTANT, m=1.)
                .set_young_modulus(young_modulus)
        ).create()

        result = self._solve(model)

        expected = np.array(
            [[-3.92668354e-16],
             [8.42105263e-02],
             [1.54798762e-01],
             [2.08132095e-01],
             [2.38901326e-01],
             [2.38901326e-01],
             ]
        )

        np.testing.assert_allclose(expected, result, atol=1e-6)

    def _create_predefined_builder(self):
        return (
            builder.create(self._length, self._node_number)
                .set_analysis_type('SYSTEM_OF_LINEAR_EQUATIONS')
                .set_boundary(builder.Side.LEFT, builder.BoundaryType.FIXED)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FREE)
                .set_load(builder.LoadType.MASS)
        )

    def _solve(self, model):
        return solve(AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, model).displacement


class TrussDynamicEigenproblemEquationFiniteDifferencesTest(unittest.TestCase):
    def setUp(self):
        self._length = 1.
        self._node_number = 31

    def test_ConstantSectionAndYoung_ReturnCorrectDisplacement(self):

        model = (
            self._create_predefined_builder()
        ).create()

        result = self._solve(model).eigenvectors[0]

        # expected = np.array([0., -0.3717, -0.6015, -0.6015, -0.3717, 0.], )
        expected = np.array([0., -0.026989, -0.053683, -0.079788, -0.105019, -0.129099, -0.151765, -0.172769, -0.191879, -0.208887,
                  -0.223607, -0.235876, -0.245562, -0.252557, -0.256784, -0.258199, -0.256784, -0.252557, -0.245562,
                  -0.235876, -0.223607, -0.208887, -0.191879, -0.172769, -0.151765, -0.129099, -0.105019, -0.079788,
                  -0.053683, -0.026989, 0.])

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def _create_predefined_builder(self):
        return (
            builder.create(self._length, self._node_number)
                .set_analysis_type('EIGENPROBLEM')
                .set_boundary(builder.Side.LEFT, builder.BoundaryType.FIXED)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FIXED)
                .set_load(builder.LoadType.MASS)
                # .add_virtual_nodes(1, 1)
                # .set_virtual_boundary_strategy(model.VirtualBoundaryStrategy.SYMMETRY)
                # .set_virtual_boundary_strategy('based_on_second_derivative')

        )

    def _solve(self, model):
        return solve(AnalysisType.EIGENPROBLEM, model)