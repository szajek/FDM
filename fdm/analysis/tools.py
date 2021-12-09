import numpy

from fdm.geometry import ClosePointsFinder


def create_weights_distributor(close_point_finder):
    def distribute(point, value):
        close_points = close_point_finder(point)
        distance_sum = sum(close_points.values())
        return dict(
            {p: (1. - distance/distance_sum)*value for p, distance in close_points.items()},
        )
    return distribute


def apply_statics_bc(variables, matrix, vector, bcs):
    extra_bcs = _extract_extra_bcs(bcs)
    _matrix = _extend_matrix(matrix, len(extra_bcs))
    _vector = _extend_vector(vector, len(extra_bcs))

    assert (_rows_number(_matrix) == len(variables), 'Number of BCs must be equal "vars_number" - "real_nodes_number"')

    initial_size = _rows_number(matrix)
    points = list(variables)

    matrix_bc_applicator = create_matrix_bc_applicator(_matrix, points, variables, initial_size)
    vector_bc_applicator = create_vector_bc_applicator(_vector, variables, initial_size)

    for i, (scheme, value, replace) in enumerate(bcs):
        matrix_bc_applicator(i, scheme, replace)
        vector_bc_applicator(i, value, replace)

    return _matrix, _vector


def apply_dynamics_bc(variables, matrix_a, matrix_b, bcs):
    extra_bcs = _extract_extra_bcs(bcs)
    _matrix_a = _extend_matrix(matrix_a, len(extra_bcs))
    _matrix_b = _extend_matrix(matrix_b, len(extra_bcs))

    assert (_rows_number(_matrix_a) == len(variables), 'Number of BCs must be equal "vars_number" - "real_nodes_number"')

    points = list(variables)
    initial_size = _rows_number(matrix_a)

    matrix_a_bc_applicator = create_matrix_bc_applicator(_matrix_a, points, variables, initial_size)
    matrix_b_bc_applicator = create_matrix_bc_applicator(_matrix_b, points, variables, initial_size)

    for i, (scheme_a, scheme_b, replace) in enumerate(bcs):
        matrix_a_bc_applicator(i, scheme_a, replace)
        matrix_b_bc_applicator(i, scheme_b, replace)

    return _matrix_a, _matrix_b


def _extract_extra_bcs(bcs):
    return [bc for bc in bcs if bc.replace is None]


def create_matrix_bc_applicator(matrix, points, variables, initial_size, tol=1e-6):
    def apply(bc_idx, scheme, replace):
        if replace:
            row_idx = variables[replace]
            matrix[row_idx, :] = 0.
        else:
            row_idx = initial_size + bc_idx
        if len(scheme):
            scheme = distribute_scheme_to_nodes(points, scheme)
            scheme = scheme.drop(tol)
        for p, weight in scheme.items():
            col_idx = variables[p]
            matrix[row_idx, col_idx] = weight
    return apply


def create_vector_bc_applicator(vector, variables, initial_size):
    def apply(bc_idx, value, replace):
        if replace:
            row_idx = variables[replace]
        else:
            row_idx = initial_size + bc_idx
        vector[row_idx] = value
    return apply


def _extend_vector(vector, bcs_number):
    current_rows_number = len(vector)
    size = current_rows_number + bcs_number
    _vector = numpy.zeros(size)
    vector_r = vector.shape[0]
    _vector[:vector_r] = vector[:vector_r]
    return _vector


def _extend_matrix(matrix, bcs_number):
    current_rows_number = matrix.shape[0]
    size = current_rows_number + bcs_number
    _matrix = numpy.zeros((size, size))
    rows_number = _rows_number(matrix)
    _matrix[:rows_number, :size] = matrix[:rows_number, :size]
    return _matrix


def _rows_number(matrix):
    return matrix.shape[0]


def distribute_scheme_to_nodes(nodes, scheme):
    free_points = tuple(scheme)
    distributor = create_weights_distributor(  # todo: optimize - it slows donw
        ClosePointsFinder(nodes, free_points)
    )
    return scheme.distribute(distributor)


