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
        self._node_number = 11

    def test_ConstantSectionAndYoung_ReturnCorrectEigenValuesAndVectors(self):

        model = (
            self._create_predefined_builder()
        ).create()

        result = self._solve(model)

        expected_eigenvectors = np.array(
            [
                [0., 0.309017, 0.587785, 0.809017, 0.951057, 1., 0.951057, 0.809017, 0.587785, 0.309017, 0.],
                [0., 0.618034, 1., 1., 0.618034, 0., -0.618034, -1., -1., -0.618034, 0.],
                [0., 0.809017, 0.951057, 0.309017, -0.587785, -1., -0.587785, 0.309017, 0.951057, 0.809017, 0.]
            ]
        )
        expected_eigenvalues = [
            9.7887,
            38.197,
            82.443,
        ]  # rad/s

        for i, (expected_value, expected_vector) in enumerate(zip(expected_eigenvalues, expected_eigenvectors)):
            self.assertAlmostEqual(expected_value, result.eigenvalues[i], places=3)
            np.testing.assert_allclose(expected_vector, result.eigenvectors[i], atol=1e-6)

    def _create_predefined_builder(self):
        return (
            builder.create(self._length, self._node_number)
                .set_analysis_type('EIGENPROBLEM')
                .set_young_modulus(1.)
                .set_boundary(builder.Side.LEFT, builder.BoundaryType.FIXED)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FIXED)
                .set_load(builder.LoadType.MASS)
                .set_virtual_boundary_strategy(builder.VirtualBoundaryStrategy.SYMMETRY)
        )

    def _solve(self, model):
        return solve(AnalysisType.EIGENPROBLEM, model)