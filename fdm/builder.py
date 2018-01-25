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


def create_virtual_boundary_based_on_second_derivative(length, span, boundaries):
    def bc_virtual_nodes_builder(side, boundary):
        _type, opts = boundary
        if _type == BoundaryType.FIXED:
            weights = {Point(0) * span: -1., Point(1 * span): 3., Point(2 * span): -3., Point(3 * span): 1.} if side == Side.LEFT \
                else {Point(-3 * span): 1., Point(-2 * span): -3., Point(-1 * span): 3., Point(0 * span): -1.}
            return fdm.Operator(fdm.Stencil(weights)), lambda a: 0.

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
        return fdm.Stencil({}), lambda a: 0.
    return get


def create_standard_virtual_boundary_provider(strategy, length):
    def get(point):
        x = point.x - length if point.x > length else point.x
        return create_statics_bc('virtual_node', x, strategy)
    return get


VIRTUAL_BOUNDARY_PROVIDER = {
    'based_on_second_derivative': create_virtual_boundary_based_on_second_derivative,
    'null': create_virtual_boundary_null_provider,
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


class StiffnessSchemeFactory:
    def __init__(self, length, span, density_controller, context):
        self._length = length
        self._span = span
        self._context = context

        self._density_controller = density_controller
        self._stiffness_stencils = self._create_stiffness_stencils()

    def __call__(self, mesh):

        return OPERATOR_DISPATCHER[self._context['operator_dispatcher_strategy']]\
            (self._stiffness_stencils, self._length, self._span)

    def _create_stiffness_stencils(self):
        def get_stiffness_correction(point):
            return self._context['stiffness_to_density_relation'](max(self._density_controller.get(point), MIN_DENSITY)) \
                if self._context['stiffness_to_density_relation'] else 1.

        return self._create_operators_set(fdm.Number(get_stiffness_correction))

    def _create_operators_set(self, stiffness_multiplier):

        deformation_operator = fdm.Operator(fdm.Stencil.central(span=self._span))
        classical_operator = stiffness_multiplier * fdm.Number(self._context['young_modulus']) * deformation_operator
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

    def __getitem__(self, item):
        return self._stiffness_stencils[item]


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
        self._context = context

    def __call__(self, point):
        return fdm.Stencil(
                    {Point(0.): 1.}
                )


def _create_statics_boundary(length, span, mesh, boundary, virtual_boundary_strategy, deformation_operator):

    def _create_real_boundary(_deformation_operator):

        bcs = []
        boundary[Side.LEFT].type != BoundaryType.NONE and bcs.append(
            (Point(0), statics_bc_equation_builder(Side.LEFT, boundary[Side.LEFT], span))
        )
        boundary[Side.RIGHT].type != BoundaryType.NONE and bcs.append(
            (Point(length), statics_bc_equation_builder(
                Side.RIGHT,
                Boundary(boundary[Side.RIGHT].type,
                         dicttools.merge(
                             boundary[Side.RIGHT].properties,
                             {'operator': _deformation_operator}
                         )
                         ),
                span
                )
            )
        )
        return dict(bcs)

    def _create_virtual_boundary(nodes):
        virtual_boundary_builder = VIRTUAL_BOUNDARY_PROVIDER[virtual_boundary_strategy]\
            (length, span, boundary)

        return {node: virtual_boundary_builder(node) for node in nodes}

    return dicttools.merge(
        _create_real_boundary(deformation_operator),
        _create_virtual_boundary(mesh.virtual_nodes)
    )


def _create_eigenproblem_boundary(length, span, mesh, boundary, virtual_boundary_strategy, deformation_operator):
    statics_boundary = _create_statics_boundary(length, span, mesh, boundary, virtual_boundary_strategy, deformation_operator)

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

    return dicttools.merge(
        _create_real_boundary(),
    )


BC_BUILDERS = {
    fdm.AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS: _create_statics_boundary,
    fdm.AnalysisType.EIGENPROBLEM: _create_eigenproblem_boundary,
}


def create_template(_type, stiffness_factory, length, span, mesh, density_controller, context):
    stiffness_stencils = stiffness_factory(length, span, density_controller, context)
    mass_stencils = MassSchemeFactory(length, span, density_controller, context)
    load_vector = LoadVectorValueFactory(length, span, density_controller, context)
    all_items = stiffness_stencils, mass_stencils, load_vector

    bcs = BC_BUILDERS[_type](
        length, span, mesh, context['boundary'], context['virtual_boundary_strategy'],
        stiffness_stencils['central_deformation_operator']
    )

    valid_items_numbers = {
        fdm.AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS: [0, 2],
        fdm.AnalysisType.EIGENPROBLEM: [0, 1],
    }

    items = [all_items[no] for no in valid_items_numbers[_type]]

    return fdm.equation.Template(
        lambda point: bcs.get(point, [item(point) for item in items])
    )


class Truss1d:
    def __init__(self, length, nodes_number, stiffness_factory_builder):
        self._length = length
        self._nodes_number = nodes_number
        self._stiffness_factory_builder = stiffness_factory_builder

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
            'young_modulus': 1.,
        }

        self.density_controller = DENSITY_CONTROLLERS['uniform'](1.)

    def set_analysis_type(self, _type):
        _type = self._context['analysis_type'] = fdm.analysis.AnalysisType[_type]
        return self

    def set_density_controller(self, _type, **options):
        self.density_controller = DENSITY_CONTROLLERS[_type](self._length, self._nodes_number, **options)
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

    def set_young_modulus(self, value_or_callable):
        self._context['young_modulus'] = value_or_callable
        return self

    def add_virtual_nodes(self, left, right):
        self._context['left_virtual_nodes_number'] = left
        self._context['right_virtual_nodes_number'] = right
        return self

    def create(self):
        assert self.density_controller is not None, "Define density controller first."
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
        return create_template(
            self._context['analysis_type'],
            self._stiffness_factory_builder,
            self._length, self._span, mesh, self.density_controller, self._context
        )

    @property
    def _last_node_index(self):
        return self._nodes_number - 1

    @property
    def density(self):
        points = [Point(self._span*i) for i in range(self._nodes_number)]
        return list(map(self.density_controller.get, points))

    def create_density_getter(self):
        return self.density.get


def create(length, nodes_number):
    return Truss1d(length, nodes_number, StiffnessSchemeFactory)