import unittest

import numpy as np

import fdm.builder as builder
from fdm.system import solve


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
            builder.Truss1d(self._length, self._node_number)
                .set_boundary(builder.Side.LEFT, builder.BoundaryType.FIXED)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FREE)
                .set_load(builder.LoadType.MASS)
        )

    def _solve(self, model):
        return solve('linear_system_of_equations', model).displacement


class TrussDynamicEigenproblemEquationFiniteDifferencesTest(unittest.TestCase):
    def setUp(self):
        self._length = 1.
        self._node_number = 6

    @unittest.skip("Wrong BC applied")
    def test_ConstantSectionAndYoung_ReturnCorrectDisplacement(self):

        model = (
            self._create_predefined_builder()
                .set_field(builder.FieldType.CONSTANT, m=2.)
        ).create()

        result = self._solve(model)

        expected = np.array([0., -0.3717, -0.6015, -0.6015, -0.3717, 0.], )

        np.testing.assert_allclose(expected, result, atol=1e-4)

    def _create_predefined_builder(self):
        return (
            builder.Truss1d(self._length, self._node_number)
                .set_boundary(builder.Side.LEFT, builder.BoundaryType.FIXED)
                .set_boundary(builder.Side.RIGHT, builder.BoundaryType.FIXED)
                .set_load(builder.LoadType.MASS)
        )

    def _solve(self, model):
        return solve('eigenproblem', model).displacement