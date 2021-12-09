import unittest
import numpy as np
from numpy.testing import assert_allclose

import fdm.analysis
import fdm.analysis.analyzer
import fdm.builder

from fdm.analysis import AnalysisStrategy


class Truss1dCaseStudy(unittest.TestCase):
    def setUp(self):
        self._length = 1
        self._node_number = 31

    def test_UpToDown_DensityConstants_ReturnCorrectDisplacements(self):
        builder = self._create_predefined_builder()
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        model = builder.create()

        result = self._solve(model).displacement

        expected = np.array([
            [-0.],
            [0.01060063],
            [0.02108512],
            [0.0313386],
            [0.04124872],
            [0.05070691],
            [0.05960955],
            [0.0678591],
            [0.07536516],
            [0.08204551],
            [0.08782695],
            [0.09264614],
            [0.09645028],
            [0.09919769],
            [0.10085827],
            [0.10141383],
            [0.10085827],
            [0.09919769],
            [0.09645028],
            [0.09264614],
            [0.08782695],
            [0.08204551],
            [0.07536516],
            [0.0678591],
            [0.05960955],
            [0.05070691],
            [0.04124872],
            [0.0313386],
            [0.02108512],
            [0.01060063],
            [0.],
         ])

        assert_allclose(expected, result, atol=1e-8)

    def test_DownToUp_DensityConstants_ReturnCorrectDisplacements(self):
        builder = self._create_predefined_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        model = builder.create()

        result = self._solve(model).displacement

        expected = np.array([
            [-0.],
            [0.01060063],
            [0.02108512],
            [0.0313386],
            [0.04124872],
            [0.05070691],
            [0.05960955],
            [0.0678591],
            [0.07536516],
            [0.08204551],
            [0.08782695],
            [0.09264614],
            [0.09645028],
            [0.09919769],
            [0.10085827],
            [0.10141383],
            [0.10085827],
            [0.09919769],
            [0.09645028],
            [0.09264614],
            [0.08782695],
            [0.08204551],
            [0.07536516],
            [0.0678591],
            [0.05960955],
            [0.05070691],
            [0.04124872],
            [0.0313386],
            [0.02108512],
            [0.01060063],
            [0.],
         ])

        assert_allclose(expected, result, atol=1e-8)

    def test_UpToDown_DensityVaried_ReturnCorrectDisplacements(self):
        builder = self._create_predefined_builder()
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        builder.set_density_controller('spline_interpolated_linearly', 6)
        builder.density_controller.update_by_control_points([0.8, 0.3385, 0.2, 0.2, 0.3351, 1.0])
        model = builder.create()

        result = self._solve(model).displacement

        expected = np.array([
            [-0.],
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
            [-0.],
        ])

        np.testing.assert_array_almost_equal(expected, result)

    def test_DownToUp_DensityVaried_ReturnCorrectDisplacements(self):
        builder = self._create_predefined_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_density_controller('spline_interpolated_linearly', 6)
        builder.density_controller.update_by_control_points([0.8, 0.3385, 0.2, 0.2, 0.3351, 1.0])
        model = builder.create()

        result = self._solve(model).displacement

        expected = np.array([
            [-0.],
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
            [-0.],
        ])

        np.testing.assert_array_almost_equal(expected, result)

    def _create_predefined_builder(self):
        builder = (
            fdm.builder.create('truss1d', self._length, self._node_number)
                .set_analysis_type('SYSTEM_OF_LINEAR_EQUATIONS')
                .add_virtual_nodes(1, 1)
                .set_boundary(fdm.builder.Side.LEFT, fdm.builder.BoundaryType.FIXED)
                .set_boundary(fdm.builder.Side.RIGHT, fdm.builder.BoundaryType.FIXED)
                .set_load(fdm.builder.LoadType.MASS)
                .set_field(fdm.builder.FieldType.SINUSOIDAL, n=1.)
                .set_stiffness_to_density_relation('exponential', c_1=1., c_2=1.)
        )
        return builder

    @staticmethod
    def _solve(model):
        return fdm.analysis.solve(fdm.analysis.analyzer.AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, model)


class Truss1dCaseStudy11(unittest.TestCase):
    def setUp(self):
        self._length = 1
        self._node_number = 11

    def test_UpToDown_DensityConstants_ReturnCorrectDisplacements(self):
        builder = self._create_predefined_builder()
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        model = builder.create()

        result = self._solve(model).displacement

        expected = np.array([
            [0.],
            [0.03156876],
            [0.06004735],
            [0.08264808],
            [0.09715865],
            [0.10215865],
            [0.09715865],
            [0.08264808],
            [0.06004735],
            [0.03156876],
            [-0.],
         ])

        assert_allclose(expected, result, atol=1e-8)

    def test_DownToUp_DensityConstants_ReturnCorrectDisplacements(self):
        builder = self._create_predefined_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        model = builder.create()

        result = self._solve(model).displacement

        expected = np.array([
            [0.],
            [0.03156876],
            [0.06004735],
            [0.08264808],
            [0.09715865],
            [0.10215865],
            [0.09715865],
            [0.08264808],
            [0.06004735],
            [0.03156876],
            [-0.],
         ])

        assert_allclose(expected, result, atol=1e-8)

    def test_UpToDown_DensityVaried_ReturnCorrectDisplacements(self):
        builder = self._create_predefined_builder()
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        builder.set_density_controller('spline_interpolated_linearly', 6)
        builder.density_controller.update_by_control_points([0.8, 0.3385, 0.2, 0.2, 0.3351, 1.0])
        model = builder.create()

        result = self._solve(model).displacement

        expected = np.array([
            [-0.],
            [0.01263637],
            [0.02836704],
            [0.04466711],
            [0.0572195],
            [0.06168855],
            [0.05641765],
            [0.04288248],
            [0.02593254],
            [0.01089392],
            [0.],
         ])

        np.testing.assert_array_almost_equal(expected, result)

    def test_DownToUp_DensityVaried_ReturnCorrectDisplacements(self):
        builder = self._create_predefined_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_density_controller('spline_interpolated_linearly', 6)
        builder.density_controller.update_by_control_points([0.8, 0.3385, 0.2, 0.2, 0.3351, 1.0])
        model = builder.create()

        result = self._solve(model).displacement

        expected = np.array([
            [-0.],
            [0.01263637],
            [0.02836704],
            [0.04466711],
            [0.0572195],
            [0.06168855],
            [0.05641765],
            [0.04288248],
            [0.02593254],
            [0.01089392],
            [0.],
         ])

        np.testing.assert_array_almost_equal(expected, result)

    def _create_predefined_builder(self):
        builder = (
            fdm.builder.create('truss1d', self._length, self._node_number)
                .set_analysis_type('SYSTEM_OF_LINEAR_EQUATIONS')
                .add_virtual_nodes(1, 1)
                .set_boundary(fdm.builder.Side.LEFT, fdm.builder.BoundaryType.FIXED)
                .set_boundary(fdm.builder.Side.RIGHT, fdm.builder.BoundaryType.FIXED)
                .set_load(fdm.builder.LoadType.MASS)
                .set_field(fdm.builder.FieldType.SINUSOIDAL, n=1.)
                .set_stiffness_to_density_relation('exponential', c_1=1., c_2=1.)
        )
        return builder

    @staticmethod
    def _solve(model):
        return fdm.analysis.solve(fdm.analysis.analyzer.AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, model)


class Beam1dCaseStudy(unittest.TestCase):
    def setUp(self):
        self._length = 1
        self._node_number = 101

    def test_UpToDown_EJQisOne_ReturnCorrectDisplacements(self):
        builder = self._create_predefined_builder()
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        model = builder.create()

        result = self._solve(model).displacement

        E = J = q = 1.
        expected_max = -1./384.*q*self._length**4/(E*J)  # -0.00260416

        np.testing.assert_allclose(min(result), [expected_max], rtol=5e-1)

    def test_DownToUp_EJQisOne_ReturnCorrectDisplacements(self):
        builder = self._create_predefined_builder()
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        model = builder.create()

        result = self._solve(model).displacement

        E = J = q = 1.
        expected_max = -1./384.*q*self._length**4/(E*J)

        np.testing.assert_allclose(min(result), [expected_max], rtol=5e-1)

    def _create_predefined_builder(self):
        builder = (
            fdm.builder.create('beam1d', self._length, self._node_number)
                .set_analysis_type('SYSTEM_OF_LINEAR_EQUATIONS')
                .set_density_controller('uniform', 1.)
                .add_virtual_nodes(8, 8)
                .set_boundary(fdm.builder.Side.LEFT, fdm.builder.BoundaryType.FIXED)
                .set_boundary(fdm.builder.Side.RIGHT, fdm.builder.BoundaryType.FIXED)
                .set_load(fdm.builder.LoadType.MASS)
                .set_field(fdm.builder.FieldType.CONSTANT, value=1.)
        )

        return builder

    @staticmethod
    def _solve(model):
        return fdm.analysis.solve(fdm.analysis.analyzer.AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, model)

