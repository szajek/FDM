import unittest
import numpy as np

import fdm.analysis
import fdm.analysis.analyzer
import fdm.builder


class Truss1dCaseStudy(unittest.TestCase):
    def setUp(self):
        self._length = 1
        self._node_number = 31

    def test_Call_DensityVaried_ReturnCorrectDisplacements(self):
        model = self._create_predefined_builder().create()

        result = self._solve(model).displacement

        expected = np.array([
            [0.],
            [0.00380564],
            [0.00808468],
            [0.01279212],
            [0.01787146],
            [0.02324998],
            [0.02883322],
            [0.03450025],
            [0.04010146],
            [0.04546146],
            [0.05038895],
            [0.05469291],
            [0.05820123],
            [0.06077555],
            [0.06231697],
            [0.06276744],
            [0.06211032],
            [0.0603667],
            [0.05758823],
            [0.05385212],
            [0.04927591],
            [0.04403642],
            [0.03836253],
            [0.03250429],
            [0.02669429],
            [0.02111944],
            [0.01591346],
            [0.01116183],
            [0.00691243],
            [0.00318848],
            [0.],
                ])

        np.testing.assert_array_almost_equal(expected, result)

    def _create_predefined_builder(self):
        builder = (
            fdm.builder.create('truss1d', self._length, self._node_number)
                .set_analysis_type('SYSTEM_OF_LINEAR_EQUATIONS')
                .set_density_controller('spline_interpolated_linearly', 6)
                .add_virtual_nodes(1, 1)
                .set_boundary(fdm.builder.Side.LEFT, fdm.builder.BoundaryType.FIXED)
                .set_boundary(fdm.builder.Side.RIGHT, fdm.builder.BoundaryType.FIXED)
                .set_load(fdm.builder.LoadType.MASS)
                .set_field(fdm.builder.FieldType.SINUSOIDAL, n=1.)
                .set_virtual_boundary_strategy('based_on_second_derivative')
                .set_stiffness_to_density_relation('exponential', c_1=1., c_2=1.)
        )

        builder.density_controller.update_by_control_points([0.8, 0.3385, 0.2, 0.2, 0.3351, 1.0])
        return builder

    @staticmethod
    def _solve(model):
        return fdm.analysis.solve(fdm.analysis.analyzer.AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, model)


class Beam1dCaseStudy(unittest.TestCase):
    def setUp(self):
        self._length = 1
        self._node_number = 101

    def test_Call_EJQisOne_ReturnCorrectDisplacements(self):
        model = self._create_predefined_builder().create()

        result = self._solve(model).displacement

        E = J = q = 1.
        expected_max = -1./384.*q*self._length**4/(E*J)

        np.testing.assert_allclose(min(result), [expected_max], rtol=5e-1)

    def _create_predefined_builder(self):
        builder = (
            fdm.builder.create('beam1d', self._length, self._node_number)
                .set_analysis_type('SYSTEM_OF_LINEAR_EQUATIONS')
                .set_density_controller('uniform', 1.)
                .add_virtual_nodes(1, 1)
                .set_boundary(fdm.builder.Side.LEFT, fdm.builder.BoundaryType.FIXED)
                .set_boundary(fdm.builder.Side.RIGHT, fdm.builder.BoundaryType.FIXED)
                .set_load(fdm.builder.LoadType.MASS)
                .set_field(fdm.builder.FieldType.CONSTANT, value=1.)
                .set_virtual_boundary_strategy('zero_value')
        )

        return builder

    @staticmethod
    def _solve(model):
        return fdm.analysis.solve(fdm.analysis.analyzer.AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, model)
