import unittest
import numpy
import numpy as np
from numpy.testing import assert_allclose

import fdm.analysis
import fdm.analysis.analyzer
import fdm.builder

from fdm.analysis import (AnalysisStrategy, AnalysisType)


class EigensolverTest(unittest.TestCase):
    def test_Eig_3x3_NonSym_ReturnCorrectEigenvalues(self):
        A = numpy.array([
            [1., 2., 9.],
            [12., 11., 2.],
            [0., 0., 4.]
        ])

        B = numpy.identity(3)

        expected = 13., 4, -1

        actual, _, = self.calc_eig(A, B)

        assert_allclose(expected, actual, atol=1e-6)

    def test_Eig_3x3_Sym_ReturnCorrectEigenvalues(self):
        A = numpy.array([
            [1., 12., 9.],
            [12., 11., 2.],
            [9., 2., 4.]
        ])

        B = numpy.identity(3)

        expected = 21.718077, 4.488646, -10.206723

        actual, _, = self.calc_eig(A, B)

        assert_allclose(expected, actual, atol=1e-6)

    def test_Lobpcg_3x3_ReturnCorrectEigenvalues(self):
        A = numpy.array([  # mast be symmetric
            [1., 12., 9.],
            [12., 11., 2.],
            [9., 2., 4.]
        ])

        B = numpy.identity(3)

        expected = 21.718077, 4.488646, -10.206723

        actual, _, = self.calc_lobpcg(A, B)

        assert_allclose(expected, actual, atol=1e-6)

    def test_Eig_Generalized_Sym3x3_ReturnCorrectValues(self):
        A = numpy.array([
            [1., 12., 9.],
            [12., 11., 2.],
            [9., 2., 4.]
        ])

        B = numpy.array([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3],
        ])

        actual, _ = self.calc_eig(A, B)

        expected = 13.31819,  1.726638, -7.211495

        assert_allclose(expected, actual, atol=1e-6)

    def test_Lobpcg_Generalized_Sym3x3_ReturnCorrectValues(self):
        A = numpy.array([
            [1., 12., 9.],
            [12., 11., 2.],
            [9., 2., 4.]
        ])

        B = numpy.array([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3],
        ])

        actual, _ = self.calc_lobpcg(A, B)

        expected = 13.31819,  1.726638, -7.211495

        assert_allclose(expected, actual, atol=1e-6)

    @classmethod
    def calc_lobpcg(cls, A, B):
        return fdm.analysis.analyzer.eigenproblem_lobpcg_solver(A, B)

    @classmethod
    def calc_eig(cls, A, B):
        return fdm.analysis.analyzer.eigenproblem_eig_solver(A, B)


class Truss1dTest(unittest.TestCase):
    @staticmethod
    def solve(model, analysis_type):
        return fdm.analysis.solve(analysis_type, model)

    @staticmethod
    def _create_predefined_builder(analysis_type, length, node_number):
        builder = (
            fdm.builder.create('truss1d', length, node_number)
            .set_analysis_type(analysis_type)
            .add_virtual_nodes(1, 1)
            .set_boundary(fdm.builder.Side.LEFT, fdm.builder.BoundaryType.FIXED)
            .set_boundary(fdm.builder.Side.RIGHT, fdm.builder.BoundaryType.FIXED)
            .set_load(fdm.builder.LoadType.MASS)
            .set_field(fdm.builder.FieldType.SINUSOIDAL, n=1.)
            .set_stiffness_to_density_relation('exponential', c_1=1., c_2=1.)
        )
        return builder


class Truss1dCaseStudy(Truss1dTest):
    def test_UpToDown_DensityConstants_ReturnCorrectDisplacements(self):
        builder = self._create_builder('SYSTEM_OF_LINEAR_EQUATIONS')
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        model = builder.create()

        result = self.solve(model, AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS).displacement

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
        builder = self._create_builder('SYSTEM_OF_LINEAR_EQUATIONS')
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        model = builder.create()

        result = self.solve(model, AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS).displacement

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
        builder = self._create_builder('SYSTEM_OF_LINEAR_EQUATIONS')
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        builder.set_density_controller('spline_interpolated_linearly', 6)
        builder.density_controller.update_by_control_points([0.8, 0.3385, 0.2, 0.2, 0.3351, 1.0])
        model = builder.create()

        result = self.solve(model, AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS).displacement

        expected = np.array([
            [-0.],
            [0.00380493],
            [0.00808263],
            [0.0127882],
            [0.01786536],
            [0.02324168],
            [0.02882309],
            [0.03448898],
            [0.04008986],
            [0.04545028],
            [0.05037859],
            [0.05468319],
            [0.05819134],
            [0.06076425],
            [0.06230312],
            [0.06275049],
            [0.06209021],
            [0.0603431],
            [0.0575593],
            [0.0538134],
            [0.04922091],
            [0.04395985],
            [0.03826431],
            [0.03239178],
            [0.02658125],
            [0.02102192],
            [0.01584295],
            [0.01112099],
            [0.00689589],
            [0.00318606],
            [0.],
        ])

        assert_allclose(expected, result, atol=1e-6)

    def test_DownToUp_DensityVaried_ReturnCorrectDisplacements(self):
        builder = self._create_builder('SYSTEM_OF_LINEAR_EQUATIONS')
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_density_controller('spline_interpolated_linearly', 6)
        builder.density_controller.update_by_control_points([0.8, 0.3385, 0.2, 0.2, 0.3351, 1.0])
        model = builder.create()

        result = self.solve(model, AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS).displacement

        expected = np.array([
            [-0.],
            [0.00380493],
            [0.00808263],
            [0.0127882],
            [0.01786536],
            [0.02324168],
            [0.02882309],
            [0.03448898],
            [0.04008986],
            [0.04545028],
            [0.05037859],
            [0.05468319],
            [0.05819134],
            [0.06076425],
            [0.06230312],
            [0.06275049],
            [0.06209021],
            [0.0603431],
            [0.0575593],
            [0.0538134],
            [0.04922091],
            [0.04395985],
            [0.03826431],
            [0.03239178],
            [0.02658125],
            [0.02102192],
            [0.01584295],
            [0.01112099],
            [0.00689589],
            [0.00318606],
            [0.],
        ])

        assert_allclose(expected, result, atol=1e-6)

    @staticmethod
    def _create_builder(analysis_type):
        return Truss1dTest._create_predefined_builder(analysis_type, 1., 31)


class Truss1dCaseStudy11(Truss1dTest):
    def setUp(self):
        self._length = 1
        self._node_number = 11

    def test_Statics_UpToDown_DensityConstants_ReturnCorrectDisplacements(self):
        builder = self._create_builder('SYSTEM_OF_LINEAR_EQUATIONS')
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        model = builder.create()

        result = self.solve(model, AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS).displacement

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

    def test_Statics_DownToUp_DensityConstants_ReturnCorrectDisplacements(self):
        builder = self._create_builder('SYSTEM_OF_LINEAR_EQUATIONS')
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        model = builder.create()

        result = self.solve(model, AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS).displacement

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

    def test_Statics_UpToDown_DensityVaried_ReturnCorrectDisplacements(self):
        builder = self._create_builder('SYSTEM_OF_LINEAR_EQUATIONS')
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        builder.set_density_controller('spline_interpolated_linearly', 6)
        builder.density_controller.update_by_control_points([0.8, 0.3385, 0.2, 0.2, 0.3351, 1.0])
        model = builder.create()

        result = self.solve(model, AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS).displacement

        expected = np.array([
            [-0.],
            [0.01263353],
            [0.02835833],
            [0.04465645],
            [0.05720914],
            [0.06167197],
            [0.05638752],
            [0.04280512],
            [0.02584205],
            [0.01087651],
            [-0.],
        ])

        assert_allclose(expected, result, atol=1e-6)

    def test_Statics_DownToUp_DensityVaried_ReturnCorrectDisplacements(self):
        builder = self._create_builder('SYSTEM_OF_LINEAR_EQUATIONS')
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_density_controller('spline_interpolated_linearly', 6)
        builder.density_controller.update_by_control_points([0.8, 0.3385, 0.2, 0.2, 0.3351, 1.0])
        model = builder.create()

        result = self.solve(model, AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS).displacement

        expected = np.array([
            [-0.],
            [0.01263353],
            [0.02835833],
            [0.04465645],
            [0.05720914],
            [0.06167197],
            [0.05638752],
            [0.04280512],
            [0.02584205],
            [0.01087651],
            [-0.],
        ])
        assert_allclose(expected, result, atol=1e-6)

    def test_Dynamics_UpToDown_DensityConstants_ReturnCorrectEigenvalues(self):
        builder = self._create_builder('EIGENPROBLEM')
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        model = builder.create()

        result = self.solve(model, AnalysisType.EIGENPROBLEM)

        actual = result.eigenvalues[:6]

        expected = np.array([
            3.1287,
            6.1803,
            9.0798,
            11.7557,
            14.1421,
            16.1803
        ])

        assert_allclose(expected, actual, atol=1e-4)

    def test_Dynamics_DownToUp_DensityConstants_ReturnCorrectEigenvalues(self):
        builder = self._create_builder('EIGENPROBLEM')
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        model = builder.create()

        result = self.solve(model, AnalysisType.EIGENPROBLEM)

        actual = result.eigenvalues[:6]

        expected = np.array([
            3.1287,
            6.1803,
            9.0798,
            11.7557,
            14.1421,
            16.1803
        ])

        assert_allclose(expected, actual, atol=1e-4)

    @staticmethod
    def _create_builder(analysis_type):
        return Truss1dTest._create_predefined_builder(analysis_type, 1., 11)


class Beam1dTest(unittest.TestCase):
    @staticmethod
    def solve(model, analysis_type):
        return fdm.analysis.solve(analysis_type, model)

    @staticmethod
    def _create_predefined_builder(analysis_type, length, node_number):
        builder = (
            fdm.builder.create('beam1d', length, node_number)
            .set_analysis_type(analysis_type)
            .set_density_controller('uniform', 1.)
            .set_young_modulus_controller('uniform', 1.)
            .set_moment_of_inertia_controller('uniform', 1.)
            .add_virtual_nodes(8, 8)
            .set_boundary(fdm.builder.Side.LEFT, fdm.builder.BoundaryType.FIXED)
            .set_boundary(fdm.builder.Side.RIGHT, fdm.builder.BoundaryType.FIXED)
            .set_load(fdm.builder.LoadType.MASS)
            .set_field(fdm.builder.FieldType.CONSTANT, value=1.)
        )

        return builder


class Beam1dCaseStudy(Beam1dTest):
    def test_Statics_UpToDown_EJQisOne_Fixed_ReturnCorrectDisplacements(self):
        builder = self._create_builder('SYSTEM_OF_LINEAR_EQUATIONS')
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        model = builder.create()

        result = self.solve(model, AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS).displacement

        E = J = q = L = 1.
        expected_max = -1. / 384. * q * L ** 4 / (E * J)  # -0.00260416

        np.testing.assert_allclose(min(result), [expected_max], rtol=1e-3)

    def test_Statics_DownToUp_EJQisOne_Fixed_ReturnCorrectDisplacements(self):
        builder = self._create_builder('SYSTEM_OF_LINEAR_EQUATIONS')
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        model = builder.create()

        result = self.solve(model, AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS).displacement

        E = J = q = L = 1.
        expected_max = -1. / 384. * q * L ** 4 / (E * J)

        np.testing.assert_allclose(min(result), [expected_max], rtol=1e-3)

    def test_Dynamics_UpToDown_EJRhoisOne_Fixed_ReturnCorrectEigenvalues(self):
        builder = self._create_builder('EIGENPROBLEM')
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        model = builder.create()

        result = self.solve(model, AnalysisType.EIGENPROBLEM).eigenvalues

        actual = result[:3]

        expected = [
            22.364,  # 22.373 -> 3.561Hz
            61.617,  # 61.688 -> 9.818Hz
            120.715,  # 121.020 -> 19.261Hz
        ]

        np.testing.assert_allclose(actual, expected, rtol=1e-3)

    def test_Dynamics_DownToUp_EJQisOne_Fixed_ReturnCorrectEigenvalues(self):
        builder = self._create_builder('EIGENPROBLEM')
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        model = builder.create()

        result = self.solve(model, AnalysisType.EIGENPROBLEM).eigenvalues

        actual = result[:3]

        expected = [
            22.364,  # 22.373 -> 3.561Hz
            61.617,  # 61.688 -> 9.818Hz
            120.715,  # 121.020 -> 19.261Hz
        ]

        np.testing.assert_allclose(actual, expected, atol=1e-1)

    def test_Dynamics_DownToUp_EJQisOne_Hinge_ReturnCorrectEigenvalues(self):
        builder = self._create_builder('EIGENPROBLEM')
        builder.set_analysis_strategy(AnalysisStrategy.DOWN_TO_UP)
        builder.set_boundary(fdm.builder.Side.LEFT, fdm.builder.BoundaryType.HINGE)
        builder.set_boundary(fdm.builder.Side.RIGHT, fdm.builder.BoundaryType.HINGE)
        model = builder.create()

        result = self.solve(model, AnalysisType.EIGENPROBLEM).eigenvalues

        actual = result[:3]

        expected = [
            9.87,  # 9.871 -> 1.571Hz
            39.47,  # 39.484 -> 6.284Hz
            88.76,  # 88.876 -> 14.145Hz
        ]

        np.testing.assert_allclose(actual, expected, atol=1e-1)

    @staticmethod
    def _create_builder(analysis_type):
        return Beam1dTest._create_predefined_builder(analysis_type, 1., 101)


class Beam1d31CaseStudy(Beam1dTest):
    def test_Dynamics_UpToDown_EJRhoisOne_Fixed_ReturnCorrectEigenvalues(self):
        builder = self._create_builder('EIGENPROBLEM')
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        builder.set_boundary(fdm.builder.Side.LEFT, fdm.builder.BoundaryType.FIXED)
        builder.set_boundary(fdm.builder.Side.RIGHT, fdm.builder.BoundaryType.FIXED)
        model = builder.create()

        result = self.solve(model, AnalysisType.EIGENPROBLEM).eigenvalues

        actual = result[:3]

        expected = [
            22.27,  # 22.373 -> 3.561Hz
            61.06,  # 61.688 -> 9.818Hz
            118.84,  # 121.020 -> 19.261Hz
        ]

        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_Dynamics_UpToDown_EJRhoisOne_Hinge_ReturnCorrectEigenvalues(self):
        builder = self._create_builder('EIGENPROBLEM')
        builder.set_analysis_strategy(AnalysisStrategy.UP_TO_DOWN)
        builder.set_boundary(fdm.builder.Side.LEFT, fdm.builder.BoundaryType.HINGE)
        builder.set_boundary(fdm.builder.Side.RIGHT, fdm.builder.BoundaryType.HINGE)
        model = builder.create()

        result = self.solve(model, AnalysisType.EIGENPROBLEM).eigenvalues

        actual = result[:3]

        expected = [
            9.86,  # 9.871 -> 1.571Hz
            39.33,  # 39.484 -> 6.284Hz
            88.1,  # 88.876 -> 14.145Hz
        ]

        np.testing.assert_allclose(actual, expected, rtol=1e-3)

    @staticmethod
    def _create_builder(analysis_type):
        return Beam1dTest._create_predefined_builder(analysis_type, 1., 31)
