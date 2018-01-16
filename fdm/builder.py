import abc
import collections
import enum
import math

import dicttools
import scipy.interpolate

import fdm.builder
import fdm.model
from fdm.geometry import Point


MIN_DENSITY = 1e-2


class BoundaryType(enum.Enum):
    FIXED = 0
    FREE = 1


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
    CONSTANT = 0
    SINUSOIDAL = 1
    LINEAR = 2


class LoadType(enum.Enum):
    NONE = 0
    MASS = 1


Field = collections.namedtuple('Field', ('type', 'properties'))
Boundary = collections.namedtuple('Boundary', ('type', 'properties'))


def bc_equation_builder(side, boundary, span):
    _type, opts = boundary
    if _type == BoundaryType.FIXED:
        return fdm.create_bc('dirichlet', value=opts.get('value', 0.))
    elif _type == BoundaryType.FREE:
        stencil = fdm.Stencil.forward(span=span) if side == Side.LEFT else fdm.Stencil.backward(span=span)
        return fdm.create_bc('neumann', stencil, value=opts.get('value', 0.))
    else:
        raise NotImplementedError


def create_virtual_boundary_based_on_second_derivative(length, span, boundaries):
    def bc_virtual_nodes_builder(side, boundary):
        _type, opts = boundary
        if _type == BoundaryType.FIXED:
            weights = {0 * span: -1., 1 * span: 3., 2 * span: -3., 3 * span: 1.} if side == Side.LEFT \
                else {-3 * span: 1., -2 * span: -3., -1 * span: 3., 0 * span: -1.}
            return fdm.LinearEquationTemplate(
                operator=fdm.Operator(fdm.Stencil(weights)),
                free_value=lambda a: 0.
            )

        elif _type == BoundaryType.FREE:
            raise NotImplementedError
        else:
            raise NotImplementedError

    bcs = dict(
        (
            (Point(-span), bc_virtual_nodes_builder(Side.LEFT, boundaries[Side.LEFT])),
            (Point(length + span), bc_virtual_nodes_builder(Side.RIGHT, boundaries[Side.RIGHT])),
        )
    )

    return lambda point: bcs[point]


def create_virtual_boundary_null_provider(*args, **kwargs):
    def get(point):
        raise AttributeError("No bc provided for {}".format(point))
    return get


def create_standard_virtual_boundary_provider(strategy, length):
    def get(point):
        x = point.x - length if point.x > length else point.x
        return fdm.model.create_bc('virtual_node', x, strategy)
    return get


VIRTUAL_BOUNDARY_PROVIDER = {
    'based_on_second_derivative': create_virtual_boundary_based_on_second_derivative,
    'null': create_virtual_boundary_null_provider,
    fdm.model.VirtualBoundaryStrategy.SYMMETRY:
        lambda length, *args: create_standard_virtual_boundary_provider(
            fdm.model.VirtualBoundaryStrategy.SYMMETRY, length),
    fdm.model.VirtualBoundaryStrategy.AS_AT_BORDER:
        lambda length, *args: create_standard_virtual_boundary_provider(
            fdm.model.VirtualBoundaryStrategy.AS_AT_BORDER, length),
}


def create_mass_load_provider(length, density, field):
    _type, props = field

    def to_ordinate(point):
        return point.x / length

    def sinus(point, factor):
        return math.sin(factor * to_ordinate(point) * math.pi)

    def linear(point, a, b):
        return a * to_ordinate(point) + b

    def get(point):

        if _type == FieldType.CONSTANT:
            m = props.get('m', 1.)
            return -m * density(point)
        elif _type == FieldType.SINUSOIDAL:
            m = props.get('m', 1.)
            factor_n = props.get('n', 1.)
            return -m * sinus(point, factor_n) * density(point)
        elif _type == FieldType.LINEAR:
            a = props.get('a', 1.)
            b = props.get('b', 0.)
            return -linear(point, a, b)*density(point)
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


SIMPLE_COMPUTING_PROCEDURE = lambda point, calculator: calculator(point)


class SplineDensityController(collections.Mapping):
    def __init__(self, length, computing_procedure=SIMPLE_COMPUTING_PROCEDURE, controllers_number=6):
        self._length = length
        self._computing_procedure = computing_procedure
        self._controllers_number = controllers_number

        self._values = {}
        self._control_points = self._create_control_points()
        self._control_values = None

    def update_by_control_points(self, controllers_values):
        assert self._controllers_number == len(controllers_values), \
            "Wrong length of control points values. Should be %d" % len(self._control_points)

        self._control_values = controllers_values
        self._values = {}

    def _create_control_points(self):
        span = self._length / (self._controllers_number - 1)
        return list(i * span for i in range(self._controllers_number))

    def __getitem__(self, point):
        assert self._control_values is not None, "Initialize control points first."
        return self._values.setdefault(point, self._computing_procedure(point, self._calculate))

    def _calculate(self, point):
        order = 3
        return float(scipy.interpolate.spline(
            self._control_points,
            self._control_values,
            [point.x],
            order=order
        )[0])

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)


def create_spline_density_controller(length, points_number, **kwargs):
    return SplineDensityController(length, computing_procedure=SIMPLE_COMPUTING_PROCEDURE, **kwargs)


def create_linearly_interpolated_spline_density_controller(length, points_number, **kwargs):
    span = length / (points_number - 1)

    def compute(point, calculator):
        n = int(point.x / span)
        x1 = n * span
        x2 = x1 + span
        v1 = calculator(Point(x1))
        v2 = calculator(Point(x2))

        return v1 + (v2 - v1) / (x2 - x1) * (point.x - x1)

    return SplineDensityController(length, computing_procedure=compute, **kwargs)


class UniformDensityController(DensityController):
    def __init__(self, value):
        self._value = value

    def get(self, point):
        return self._value

    def update(self, value):
        self._value = value


DENSITY_CONTROLLERS = {
    'spline': create_spline_density_controller,
    'spline_interpolated_linearly': create_linearly_interpolated_spline_density_controller,
    'uniform': UniformDensityController,
}


def create_standard_operator_dispatcher(operators, length, span):
    return operators['central']


def create_min_virtual_layer_operator_dispatcher(operators, length, span):
    return fdm.DynamicElement(
        lambda point: {
            Point(0. + span): operators['forward_central'],
            Point(length - span): operators['backward_central'],
        }.get(point, operators['central'])
    )


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


class Truss1d:
    def __init__(self, length, nodes_number):
        self._length = length
        self._nodes_number = nodes_number

        self._span = length / (nodes_number - 1)
        self._left_virtual_nodes_number = 0
        self._right_virtual_nodes_number = 0

        self._boundary_builder = bc_equation_builder
        self._virtual_boundary_builder = None

        self._boundary = {}
        self._field = None
        self._load = LoadType.NONE
        self._young_modulus = 1.

        self.set_virtual_boundary_strategy('null')
        self.density_controller = DENSITY_CONTROLLERS['uniform'](1.)
        self._stiffness_to_density_relation = None
        self._operator_dispatcher_strategy = 'standard'
        self._free_vector_provider = create_free_vector_provider_factory(length)

    def set_density_controller(self, _type, **options):
        self.density_controller = DENSITY_CONTROLLERS[_type](self._length, self._nodes_number, **options)
        return self

    def set_boundary(self, side, _type, **opts):
        self._boundary[side] = Boundary(_type, opts)
        return self

    def set_operator_dispatcher_strategy(self, strategy):
        self._operator_dispatcher_strategy = strategy
        return self

    def set_virtual_boundary_strategy(self, _type):
        self._virtual_boundary_builder = VIRTUAL_BOUNDARY_PROVIDER[_type](self._length, self._span, self._boundary)
        return self

    def set_field(self, _type, **properties):
        self._field = Field(_type, properties)
        return self

    def set_load(self, _type):
        self._load = _type
        return self

    def set_stiffness_to_density_relation(self, _type, **options):
        self._stiffness_to_density_relation = RELATIONS[_type](**options)
        return self

    def set_young_modulus(self, value_or_callable):
        self._young_modulus = value_or_callable
        return self

    def add_virtual_nodes(self, left, right):
        self._left_virtual_nodes_number = left
        self._right_virtual_nodes_number = right
        return self

    def create(self):
        assert self.density_controller is not None, "Define density controller first."

        free_vector = self._free_vector_provider(self._load, self.density_controller.get, self._field)
        operators = self._create_operators()
        mesh = self._create_mesh()

        return fdm.Model(
            self._create_equation(operators, free_vector),
            mesh,
            bcs=dicttools.merge(
                self._create_boundary(operators['central_deformation_operator']),
                self._create_virtual_boundary(mesh.virtual_nodes)
            )
        )

    def _create_equation(self, operators, free_vector):
        return fdm.LinearEquationTemplate(
            OPERATOR_DISPATCHER[self._operator_dispatcher_strategy](operators, self._length, self._span),
            free_vector
        )

    def _create_operators(self):
        def get_stiffness_correction(point):
            return self._stiffness_to_density_relation(max(self.density_controller.get(point), MIN_DENSITY)) \
                if self._stiffness_to_density_relation else 1.

        return self._create_operators_set(fdm.Number(get_stiffness_correction))

    def _create_operators_set(self, stiffness_multiplier):

        deformation_operator = fdm.Operator(fdm.Stencil.central(span=self._span))
        classical_operator = stiffness_multiplier * fdm.Number(self._young_modulus) * deformation_operator
        class_eq_central = fdm.Operator(fdm.Stencil.central(span=self._span), classical_operator)
        class_eq_backward = fdm.Operator(fdm.Stencil.backward(span=self._span), classical_operator)
        class_eq_forward = fdm.Operator(fdm.Stencil.forward(span=self._span), classical_operator)

        return {
            'central': class_eq_central,
            'forward': class_eq_forward,
            'backward': class_eq_backward,
            'forward_central': class_eq_central,
            'backward_central': class_eq_central,
            'central_deformation_operator': deformation_operator,
        }

    def _create_mesh(self):
        vn_left = [-self._span * (i + 1) for i in range(self._left_virtual_nodes_number)]
        vn_right = [self._length + self._span * (i + 1) for i in range(self._right_virtual_nodes_number)]
        return (
            fdm.Mesh1DBuilder(self._length)
                .add_uniformly_distributed_nodes(self._nodes_number)
                .add_virtual_nodes(*(vn_left + vn_right))
                .create())

    def _create_boundary(self, fractional_deformation_operator):
        return dict(
            (
                (Point(0), self._boundary_builder(Side.LEFT, self._boundary[Side.LEFT], self._span)),
                (Point(self._length), self._boundary_builder(
                    Side.RIGHT,
                    Boundary(self._boundary[Side.RIGHT].type,
                             dicttools.merge(
                                 self._boundary[Side.RIGHT].properties,
                                 {'operator': fractional_deformation_operator}
                             )
                             ),
                    self._span
                ))
            )
        )

    def _create_virtual_boundary(self, nodes):
        return {node: self._virtual_boundary_builder(node) for node in nodes}

    @property
    def _last_node_index(self):
        return self._nodes_number - 1

    @property
    def density(self):
        points = [Point(self._span*i) for i in range(self._nodes_number)]
        return list(map(self.density_controller.get, points))

    def create_density_getter(self):
        return self.density.get
