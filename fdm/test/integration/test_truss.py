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

    def test_LocalSectionReduction_ReturnCorrectEigenValuesAndVectors(self):

        builder = self._create_predefined_builder()
        data = [
            [0., 1.],
            [0.20, 1.],
            [0.21, 0.2],
            # [0.21, 1.],
            [0.39, 0.2],
            # [0.39, 1.],
            [0.40, 1.],
            [1., 1.]
        ]
        xs, values = zip(*data)
        builder.set_density_controller('spline', controllers_number=6, controllers_coordinates=xs, order=1)
        builder.density_controller.update_by_control_points(values)
        builder.set_stiffness_to_density_relation('exponential', c_1=1., c_2=1.)

        model = builder.create()

        result = self._solve(model)

        expected_eigenvectors = np.array(
            [
                [-0.0, 0.109264, 0.209878, 0.629871, 1.0, 0.994859, 0.910958, 0.75494, 0.539156, 0.280689, -0.0],
                [-0.0, 0.629629, 1.0, 0.793039, 0.259534, 0.0459666, -0.186528, -0.342218, -0.356995, -0.224774, -0.0],
                [-0.0, 0.529211, 0.767237, -0.153396, -0.989627, -0.6123558, 0.101848, 0.760013, 1.0, 0.689762, -0.0],
            ]
        )

        expected_eigenvalues = [
            7.9167,
            41.1763,
            55.0225,
        ]  # rad/s

        for i, (expected_value, expected_vector) in enumerate(zip(expected_eigenvalues, expected_eigenvectors)):

            np.testing.assert_allclose(expected_vector, result.eigenvectors[i], atol=1e-6)
            self.assertAlmostEqual(expected_value, result.eigenvalues[i], places=3)

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