import abc
import collections
import enum
import math

import numpy as np
import dicttools
import scipy.interpolate

import fdm.builder
import fdm.model
import fdm
from fdm.geometry import Point

MIN_DENSITY = 1e-2


class BoundaryType(enum.Enum):
    NONE = 0
    FIXED = 1
    FREE = 2


class BoundaryScheme(enum.Enum):
    FIXED_FIXED = (BoundaryType.FIXED, BoundaryType.FIXED)
    FIXED_FREE = (BoundaryType.FIXED, BoundaryType.FREE)

    def __init__(self, left, right):
        self.left = left
        self.right = right


class Side(enum.Enum):
    LEFT = 0
    RIGHT = 1


class FieldType(enum.Enum):
    NONE = 0
    CONSTANT = 1
    SINUSOIDAL = 2
    LINEAR = 3


class LoadType(enum.Enum):
    NONE = 0
    MASS = 1


Field = collections.namedtuple('Field', ('type', 'properties'))
Boundary = collections.namedtuple('Boundary', ('type', 'properties'))


class VirtualBoundaryStrategy(enum.Enum):
    AS_AT_BORDER = 0
    SYMMETRY = 1


class Builder1d:
    def __init__(self, length, nodes_number):
        self._length = length
        self._nodes_number = nodes_number
        self._stiffness_factory = None

        self._span = length / (nodes_number - 1)

        self._context = {
            'analysis_type': None,
            'left_virtual_nodes_number': 0,
            'right_virtual_nodes_number': 0,
            'boundary': {
                Side.LEFT: BoundaryType.FIXED,
                Side.RIGHT: BoundaryType.FIXED,
            },
            'operator_dispatcher_strategy': 'standard',
            'virtual_boundary_strategy': 'null',
            'field': Field(FieldType.NONE, ()),
            'load': LoadType.NONE,
            'stiffness_to_density_relation': None,
        }

        self.density_controller = self._create_value_controller('uniform', value=1.)
        self.young_modulus_controller = self._create_value_controller('uniform', value=1.)

    def set_stiffness_factory(self, factory):
        self._stiffness_factory = factory

    def set_analysis_type(self, _type):
        self._context['analysis_type'] = fdm.analysis.AnalysisType[_type]
        return self

    def set_density_controller(self, _type, *args, **options):
        self.density_controller = self._create_value_controller(_type, *args, **options)
        return self

    def set_boundary(self, side, _type, **opts):
        self._context['boundary'][side] = Boundary(_type, opts)
        return self

    def set_operator_dispatcher_strategy(self, strategy):
        self._context['operator_dispatcher_strategy'] = strategy
        return self

    def set_virtual_boundary_strategy(self, _type):
        self._context['virtual_boundary_strategy'] = _type
        return self

    def set_field(self, _type, **properties):
        self._context['field'] = Field(_type, properties)
        return self

    def set_load(self, _type):
        self._context['load'] = _type
        return self

    def set_stiffness_to_density_relation(self, _type, **options):
        self._context['stiffness_to_density_relation'] = RELATIONS[_type](**options)
        return self

    def set_young_modulus_controller(self, _type, *args, **options):
        self.young_modulus_controller = self._create_value_controller(_type, *args, **options)
        return self

    def _create_value_controller(self, _type, *args, **options):
        return VALUE_CONTROLLERS[_type](self._length, self._nodes_number, *args, **options)

    def add_virtual_nodes(self, left, right):
        self._context['left_virtual_nodes_number'] = left
        self._context['right_virtual_nodes_number'] = right
        return self

    def create(self):
        assert self.density_controller is not None, "Define density controller first."
        assert self.young_modulus_controller is not None, "Define Young modulus controller first."
        assert self._context['analysis_type'] is not None, "Define analysis type first."
        mesh = self._create_mesh()

        return fdm.Model(
            self._create_template(mesh),
            mesh,
        )

    def _create_mesh(self):
        vn_left = [-self._span * (i + 1) for i in range(self._context['left_virtual_nodes_number'])]
        vn_right = [self._length + self._span * (i + 1) for i in range(self._context['right_virtual_nodes_number'])]
        return (
            fdm.Mesh1DBuilder(self._length)
                .add_uniformly_distributed_nodes(self._nodes_number)
                .add_virtual_nodes(*(vn_left + vn_right))
                .create())

    def _create_template(self, mesh):
        assert self._stiffness_factory is not None, 'Set stiffness factory first'
        _type = self._context['analysis_type']

        stiffness_operators_set = self._create_stiffness_operators_set()

        stiffness_stencils = OPERATOR_DISPATCHER[self._context['operator_dispatcher_strategy']](
            stiffness_operators_set, self._length, self._span)

        bcs = BC_BUILDERS[_type](
            self._length, self._span, mesh, self._context['boundary'], self._context['virtual_boundary_strategy']
        )

        if _type == fdm.AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS:
            load_vector = LoadVectorValueFactory(self._length, self._span, self.density_controller, self._context)
            items = stiffness_stencils, load_vector
        elif _type == fdm.AnalysisType.EIGENPROBLEM:
            mass_stencils = MassSchemeFactory(self._length, self._span, self.density_controller, self._context)
            items = stiffness_stencils, mass_stencils
        else:
            raise AttributeError

        def get_expander(point):
            lhs, rhs = items
            return bcs.get(point, [lhs, rhs(point)])

        return fdm.equation.Template(get_expander)

    def _create_stiffness_operators_set(self):
        return self._stiffness_factory(self._span, self._get_corrected_young_modulus)

    def _get_corrected_young_modulus(self, point):
        correction = self._context['stiffness_to_density_relation']
        scale = correction(max(self.density_controller.get(point), MIN_DENSITY)) if correction else 1.
        return self.young_modulus_controller(point) * scale

    @property
    def _last_node_index(self):
        return self._nodes_number - 1

    @property
    def density(self):
        return self._revolve_for_points(self.density_controller.get)

    @property
    def young_modulus(self):
        return self._revolve_for_points(self.young_modulus_controller.get)

    @property
    def points(self):
        return [Point(self._span * i) for i in range(self._nodes_number)]

    def _revolve_for_points(self, _callable):
        return list(map(_callable, self.points))

    def create_density_getter(self):
        return self.density.get


def create_for_truss_1d(length, nodes_number):
    builder = Builder1d(length, nodes_number)
    builder.set_stiffness_factory(create_truss1d_stiffness_operators_set)
    return builder


def create_for_beam_1d(length, nodes_number):
    builder = Builder1d(length, nodes_number)
    builder.set_stiffness_factory(create_beam1d_stiffness_operators_set)
    return builder


_builders = {
    'truss1d': create_for_truss_1d,
    'beam1d': create_for_beam_1d
}


def create(type_, *args, **kwargs):
    return _builders[type_](*args, **kwargs)


def create_truss1d_stiffness_operators_set(span, young_modulus_controller):
    deformation_operator = fdm.Operator(fdm.Stencil.central(span=span))
    classical_operator = fdm.Number(young_modulus_controller) * deformation_operator
    class_eq_central = fdm.Operator(fdm.Stencil.central(span=span), classical_operator)
    class_eq_backward = fdm.Operator(fdm.Stencil.backward(span=span), classical_operator)
    class_eq_forward = fdm.Operator(fdm.Stencil.forward(span=span), classical_operator)

    return {
        'central': class_eq_central,
        'forward': class_eq_forward,
        'backward': class_eq_backward,
        'forward_central': class_eq_central,
        'backward_central': class_eq_central,
    }


def create_beam1d_stiffness_operators_set(span, young_modulus_controller):
    central_operator = fdm.Operator(fdm.Stencil.central(span=span))
    central_stencil = fdm.Stencil.central(span=span)
    backward_stencil = fdm.Stencil.backward(span=span)
    forward_stencil = fdm.Stencil.forward(span=span)

    EI = fdm.Number(young_modulus_controller) * fdm.Number(1.)

    class_eq_central = fdm.Operator(
        central_stencil, fdm.Operator(
            central_stencil, fdm.Operator(
                central_stencil, fdm.Operator(
                    EI * central_operator
                )
            )
        )
    )

    class_eq_backward = fdm.Operator(
        backward_stencil, fdm.Operator(
            backward_stencil, fdm.Operator(
                backward_stencil, fdm.Operator(
                    EI * central_operator
                )
            )
        )
    )
    class_eq_forward = fdm.Operator(
        forward_stencil, fdm.Operator(
            forward_stencil, fdm.Operator(
                forward_stencil, fdm.Operator(
                    EI * central_operator
                )
            )
        )
    )

    return {
        'central': class_eq_central,
        'forward': class_eq_forward,
        'backward': class_eq_backward,
        'forward_central': class_eq_central,
        'backward_central': class_eq_central,
    }


def _create_statics_boundary(length, span, mesh, boundary, virtual_boundary_strategy):
    def _create_real_boundary():

        bcs = []
        if boundary[Side.LEFT].type != BoundaryType.NONE:
            bcs.append(
                (Point(0), statics_bc_equation_builder(Side.LEFT, boundary[Side.LEFT], span))
            )

        if boundary[Side.RIGHT].type != BoundaryType.NONE:
            bcs.append(
                (Point(length), statics_bc_equation_builder(Side.RIGHT, boundary[Side.RIGHT], span))
            )
        return dict(bcs)

    def _create_virtual_boundary(nodes):
        virtual_boundary_builder = VIRTUAL_BOUNDARY_PROVIDER[virtual_boundary_strategy] \
            (length, span)

        return {node: virtual_boundary_builder(node) for node in nodes}

    return dicttools.merge(
        _create_real_boundary(),
        _create_virtual_boundary(mesh.virtual_nodes)
    )


def _create_eigenproblem_boundary(length, span, mesh, boundary, virtual_boundary_strategy):
    statics_boundary = _create_statics_boundary(length, span, mesh, boundary, virtual_boundary_strategy)

    def create_mass_bc_stencil(_type):
        if _type == BoundaryType.FIXED:
            return fdm.Stencil({})
        else:
            raise NotImplementedError

    start, end = Point(0), Point(length)
    left_type, right_type = boundary[Side.LEFT].type, boundary[Side.RIGHT].type

    def _create_real_boundary():

        bcs = []
        left_type != BoundaryType.NONE and bcs.append(
            (start, (statics_boundary[start].operator, create_mass_bc_stencil(left_type)))
        )
        right_type != BoundaryType.NONE and bcs.append(
            (end, (statics_boundary[end].operator, create_mass_bc_stencil(right_type)))
        )
        return dict(bcs)

    def _create_virtual_boundary(nodes):
        return {node: (fdm.Stencil({}), fdm.Stencil({})) for node in nodes}

    return dicttools.merge(
        _create_real_boundary(),
        _create_virtual_boundary(mesh.virtual_nodes)
    )


BC_BUILDERS = {
    fdm.AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS: _create_statics_boundary,
    fdm.AnalysisType.EIGENPROBLEM: _create_eigenproblem_boundary,
}


def _create_static_dirichlet_bc(value=0.):
    return LinearEquationTemplate(
        fdm.Operator(fdm.Stencil({Point(0): 1.})),
        lambda node_address: value,
    )


def _create_static_neumann_bc(stencil, value=0.):
    return LinearEquationTemplate(
        fdm.Operator(stencil),
        lambda node_address: value,
    )


def _create_bc_by_equation(operator, free_value=0.):
    return LinearEquationTemplate(
        operator,
        lambda node_address: free_value,
    )


def _create_virtual_nodes_bc(x, strategy):
    m = {
        VirtualBoundaryStrategy.SYMMETRY: 2.,
        VirtualBoundaryStrategy.AS_AT_BORDER: 1.,
    }[strategy]
    return LinearEquationTemplate(
        fdm.Stencil(
            {
                Point(0.): 1.,
                Point(-np.sign(x) * m * abs(x)): -1.
            }
        ),
        lambda p: 0.
    )


_statics_bc_generators = {
    'dirichlet': _create_static_dirichlet_bc,
    'neumann': _create_static_neumann_bc,
    'equation': _create_bc_by_equation,
    'virtual_node': _create_virtual_nodes_bc,
}


class LinearEquationTemplate(fdm.Template):
    def __init__(self, operator, free_value):
        self.operator = operator
        self.free_value = free_value
        super().__init__((operator, free_value))


class EigenproblemEquationTemplate(fdm.Template):
    def __init__(self, operator_A, operator_B):
        self.operator_A = operator_A
        self.operator_B = operator_B
        super().__init__((operator_A, operator_B))


def create_statics_bc(_type, *args, **kwargs):
    return _statics_bc_generators[_type](*args, **kwargs)


def statics_bc_equation_builder(side, boundary, span):
    _type, opts = boundary
    if _type == BoundaryType.FIXED:
        return create_statics_bc('dirichlet', value=opts.get('value', 0.))
    elif _type == BoundaryType.FREE:
        stencil = fdm.Stencil.forward(span=span) if side == Side.LEFT else fdm.Stencil.backward(span=span)
        return create_statics_bc('neumann', stencil, value=opts.get('value', 0.))
    else:
        raise NotImplementedError


def create_virtual_boundary_based_on_second_derivative(length, span):
    def p(s):
        return Point(span * s)

    def equation(w):
        return fdm.Operator(fdm.Stencil(w)), lambda a: 0.

    left_weights = {p(0): -1., p(1): 3., p(2): -3., p(3): 1.}
    right_weights = {p(-3): 1., p(-2): -3., p(-1): 3., p(0): -1.}

    bcs = {
        Point(-span): equation(left_weights),
        Point(length + span): equation(right_weights)
    }

    def get(point):
        return bcs[point]

    return get


def create_virtual_boundary_based_on_fourth_derivative(length, span):
    def p(s):
        return Point(span * s)

    def equation(w):
        return fdm.Operator(fdm.Stencil(w)), lambda a: 0.

    left_weights = {p(-2): -1., p(-1): 4., p(0): -5., p(2): 5., p(3): -4., p(4): 1.}
    right_weights = {p(2): -1., p(1): 4., p(0): -5., p(-2): 5., p(-3): -4., p(-4): 1.}

    bcs = dicttools.merge(
        {Point(span): equation(left_weights) for span in range(-6, 0)},
        {Point(length + span): equation(right_weights) for span in range(1, 7)},
    )

    def get(point):
        return bcs[point]

    return get


def create_virtual_boundary_null_provider(*args, **kwargs):
    def get(point):
        return fdm.Stencil({}), lambda a: 0.

    return get


def create_virtual_boundary_zero_value_provider():
    def get(point):
        return create_statics_bc('dirichlet', value=0)

    return get


def create_standard_virtual_boundary_provider(strategy, length):
    def get(point):
        x = point.x - length if point.x > length else point.x
        return create_statics_bc('virtual_node', x, strategy)

    return get


VIRTUAL_BOUNDARY_PROVIDER = {
    'based_on_second_derivative': create_virtual_boundary_based_on_second_derivative,
    'based_on_fourth_derivative': create_virtual_boundary_based_on_fourth_derivative,
    'null': create_virtual_boundary_null_provider,
    'zero_value': lambda *args: create_virtual_boundary_zero_value_provider(),
    VirtualBoundaryStrategy.SYMMETRY:
        lambda length, *args: create_standard_virtual_boundary_provider(
            VirtualBoundaryStrategy.SYMMETRY, length),
    VirtualBoundaryStrategy.AS_AT_BORDER:
        lambda length, *args: create_standard_virtual_boundary_provider(
            VirtualBoundaryStrategy.AS_AT_BORDER, length),
}


#


def create_mass_load_provider(length, density, field):
    _type, props = field

    def to_ordinate(point):
        return point.x / length

    def sinus(point, factor):
        return math.sin(factor * to_ordinate(point) * math.pi)

    def linear(point, a, b):
        return a * to_ordinate(point) + b

    def get(point):

        if _type == FieldType.NONE:
            return 0.
        elif _type == FieldType.CONSTANT:
            m = props.get('m', 1.)
            return -m * density(point)
        elif _type == FieldType.SINUSOIDAL:
            m = props.get('m', 1.)
            factor_n = props.get('n', 1.)
            return -m * sinus(point, factor_n) * density(point)
        elif _type == FieldType.LINEAR:
            a = props.get('a', 1.)
            b = props.get('b', 0.)
            return -linear(point, a, b) * density(point)
        else:
            raise NotImplementedError

    return get


def create_free_vector_provider_factory(length):
    def zero(point):
        return 0.

    def build(load, density, field):
        if load == LoadType.MASS:
            return create_mass_load_provider(length, density, field)
        elif load == LoadType.NONE:
            return zero
        else:
            raise NotImplementedError

    return build


class DensityController(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get(self, point):
        raise NotImplementedError

    def __call__(self, point):
        return self.get(point)


def calculate_directly_for_point(point, calculator):
    return calculator(point)


class SplineValueController(DensityController):
    def __init__(self, points, computing_procedure=calculate_directly_for_point, order=3):
        self._points = points
        self._computing_procedure = computing_procedure
        self._order = order

        self._values = None
        self._cached_values = {}

    def update_by_control_points(self, controllers_values):
        assert len(self._points) == len(controllers_values), \
            "Wrong length of control points values. Should be %d" % len(self._points)

        self._values = controllers_values
        self._cached_values = {}

    def get(self, point):
        assert self._values is not None, "Initialize control points first."
        return self._cached_values.setdefault(point, self._computing_procedure(point, self._calculate))

    def _calculate(self, point):
        return float(scipy.interpolate.spline(
            self._points,
            self._values,
            [point.x],
            order=self._order
        )[0])

    def __len__(self):
        return len(self._cached_values)

    def __iter__(self):
        return iter(self._cached_values)


def create_x_data(length, nodes_number):
    dx = 1. / (nodes_number - 1)
    return [dx * i * length for i in range(0, nodes_number)]


def create_spline_value_controller(length, points_number, knots_number, **kwargs):
    return SplineValueController(create_x_data(length, knots_number), **kwargs)


def create_spline_value_controller_with_user_knots(length, points_number, coordinates, values=None, **kwargs):
    controller = SplineValueController(coordinates, **kwargs)
    if values is not None:
        controller.update_by_control_points(values)
    return controller


def create_linearly_interpolated_spline_value_controller(length, points_number, knots_number, **kwargs):
    span = length / (points_number - 1)

    def compute(point, calculator):
        n = int(point.x / span)
        x1 = n * span
        x2 = x1 + span
        v1 = calculator(Point(x1))
        v2 = calculator(Point(x2))

        return v1 + (v2 - v1) / (x2 - x1) * (point.x - x1)

    return SplineValueController(create_x_data(length, knots_number), computing_procedure=compute, **kwargs)


def create_segments_value_controller(length, points_number, number=10, values=(1.,)):
    segment_length = length / number
    segments_in_module = len(values)

    tol = 1e-3
    stiffness = []
    for i in range(number + 1):
        x = i * segment_length
        si = int(math.fmod(i, segments_in_module))
        s = values[si]
        if i in [0, number]:
            stiffness.append((x, s))
        else:
            s2 = values[int(math.fmod(i + 1, segments_in_module))]
            xp, xn = x - tol, x + tol
            stiffness.append((xp, s))
            stiffness.append((xn, s2))

    xs, values = zip(*stiffness)
    controller = SplineValueController(xs, order=1)
    controller.update_by_control_points(values)
    return controller


class UniformValueController(DensityController):
    def __init__(self, length, nodes_number, value):
        self._value = value

    def get(self, point):
        return self._value

    def update(self, value):
        self._value = value


class DirectValueController(DensityController):
    def __init__(self, length, points_number, default=1.):
        span = length / (points_number - 1)
        points = [Point(i * span) for i in range(points_number)]
        self._values = {p: default for p in points}

    def get(self, point):
        assert self._values is not None, "Update values first."
        return self._values[point]

    def update(self, values):
        self._values.update(values)


def create_segments_discrete_value_controller(length, points_number, number=10, values=(1.,), default=1.):
    dx = length / (points_number - 1)
    segment_length = length / number
    segments_in_module = len(values)

    def get_density(x):
        if math.fabs(x - round(x / segment_length) * segment_length) < 1e-6:
            n = round(x / segment_length)
            segment_in_module_idx = int(math.fmod(n, segments_in_module))
            return values[segment_in_module_idx] / dx
        else:
            return 1e-6

    controller = DirectValueController(length, points_number, default=default)
    controller.update({Point(dx * i): get_density(dx * i) for i in range(points_number)})
    return controller


class UserValueController(DensityController):
    def __init__(self, length, nodes_number, _callable):
        self._callable = _callable

    def get(self, point):
        return self._callable(point)


VALUE_CONTROLLERS = {
    'spline': create_spline_value_controller,
    'spline_with_user_knots': create_spline_value_controller_with_user_knots,
    'spline_interpolated_linearly': create_linearly_interpolated_spline_value_controller,
    'segments': create_segments_value_controller,
    'uniform': UniformValueController,
    'direct': DirectValueController,
    'segments_discrete': create_segments_discrete_value_controller,
    'user': UserValueController,
}


def create_standard_operator_dispatcher(operators, length, span):
    return operators['central']


def create_min_virtual_layer_operator_dispatcher(operators, length, span):

    def dispatch(point):
        return {
            Point(0. + span): operators['forward_central'],
            Point(length - span): operators['backward_central'],
        }.get(point, operators['central'])

    return fdm.DynamicElement(dispatch)


OPERATOR_DISPATCHER = {
    'standard': create_standard_operator_dispatcher,
    'minimize_virtual_layer': create_min_virtual_layer_operator_dispatcher,
}


def exponential_relation(c_1, c_2):
    def get(value):
        return c_1 * value ** c_2

    return get


def user_relation(_callable):
    def get(value):
        return _callable(value)

    return get


RELATIONS = {
    'exponential': exponential_relation,
    'user': user_relation,
}


class LoadVectorValueFactory:
    def __init__(self, length, span, density_controller, context):
        self._density_controller = density_controller
        self._context = context

        self._free_vector_provider = create_free_vector_provider_factory(length)

    def __call__(self, point):
        return self._free_vector_provider(
            self._context['load'],
            self._density_controller.get,
            self._context['field']
        )


class MassSchemeFactory:
    def __init__(self, length, span, density_controller, context):
        self._density_controller = density_controller

    def __call__(self, point):
        return fdm.Stencil(
            {Point(0.): self._density_controller.get(point)}
        )
