import numpy

from fdm.geometry import create_close_point_finder


def create_weights_distributor(close_point_finder):
    def distribute(point, value):
        close_points = close_point_finder(point)
        distance_sum = sum(close_points.values())
        return dict(
            {p: (1. - distance/distance_sum)*value for p, distance in close_points.items()},
        )
    return distribute


def apply_statics_bc(variables, matrix, vector, bcs):
    extra_bcs = extract_extra_bcs(bcs)
    replace_bcs = extract_replace_bcs(bcs)
    extra_bcs_number = len(extra_bcs)

    _matrix = numpy.copy(matrix)
    _vector = numpy.copy(vector)

    assert (_rows_number(_matrix) == len(variables), 'Number of BCs must be equal "vars_number" - "real_nodes_number"')

    points = list(variables)

    matrix_bc_applicator = create_matrix_bc_applicator(_matrix, points, variables)
    vector_bc_applicator = create_vector_bc_applicator(_vector)

    for i, (scheme, value, replace) in enumerate(replace_bcs):
        matrix_bc_applicator(variables[replace], scheme)
        vector_bc_applicator(variables[replace], value)

    initial_idx = _rows_number(matrix) - extra_bcs_number
    for i, (scheme, value, _) in enumerate(extra_bcs):
        matrix_bc_applicator(initial_idx + i, scheme)
        vector_bc_applicator(initial_idx + i, value)

    return _matrix, _vector


def apply_dynamics_bc(variables, matrix_a, matrix_b, bcs):
    extra_bcs = extract_extra_bcs(bcs)
    replace_bcs = extract_replace_bcs(bcs)
    extra_bcs_number = len(extra_bcs)

    _matrix_a = numpy.copy(matrix_a)
    _matrix_b = numpy.copy(matrix_b)

    assert _rows_number(_matrix_a) == len(variables), 'Number of BCs must be equal "vars_number" - "real_nodes_number"'

    points = list(variables)

    matrix_a_bc_applicator = create_matrix_bc_applicator(_matrix_a, points, variables)
    matrix_b_bc_applicator = create_matrix_bc_applicator(_matrix_b, points, variables)

    for i, (scheme_a, scheme_b, replace) in enumerate(replace_bcs):
        matrix_a_bc_applicator(variables[replace], scheme_a)
        matrix_b_bc_applicator(variables[replace], scheme_b)

    initial_idx = _rows_number(_matrix_a) - extra_bcs_number
    for i, (scheme_a, scheme_b, _) in enumerate(extra_bcs):
        matrix_a_bc_applicator(initial_idx + i, scheme_a)
        matrix_b_bc_applicator(initial_idx + i, scheme_b)

    return _matrix_a, _matrix_b


def extract_extra_bcs(bcs):
    return [bc for bc in bcs if bc.replace is None]


def extract_replace_bcs(bcs):
    return [bc for bc in bcs if bc.replace is not None]


def create_matrix_bc_applicator(matrix, points, variables, tol=1e-6):
    def apply(row_idx, scheme):
        matrix[row_idx, :] = 0.
        if len(scheme):
            distributor = SchemeToNodesDistributor(points)
            scheme = distributor(scheme)
            scheme = scheme.drop(tol)
        for p, weight in scheme.items():
            col_idx = variables[p]
            matrix[row_idx, col_idx] = weight
    return apply


def create_vector_bc_applicator(vector):
    def apply(row_idx, value):
        vector[row_idx] = value
    return apply


def _zero_vector_last_rows(vector, number):
    _vector = numpy.zeros(vector.shape)
    _vector[:-number] = vector[:-number]
    return _vector


def _zero_matrix_last_rows(matrix, number):
    _matrix = numpy.zeros(matrix.shape)
    _matrix[:-number, :] = matrix[:-number, :]
    return _matrix


def _rows_number(matrix):
    return matrix.shape[0]


def _cols_number(matrix):
    return matrix.shape[1]


class SchemeToNodesDistributor(object):
    def __init__(self, nodes):
        self._distributor = WeightsDistributor(nodes)

    def __call__(self, scheme):
        return scheme.distribute(self._distributor)


class WeightsDistributor(object):
    def __init__(self, nodes):
        self._distributor = create_weights_distributor(
            create_close_point_finder(nodes)
        )

    def __call__(self, point, weight):
        return self._distributor(point, weight)