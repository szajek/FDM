import abc
import collections
import enum
import math

import numpy
import numpy as np
import dicttools
import scipy.interpolate

import fdm.analysis
import fdm.analysis.analyzer
import fdm.model
import fdm
from fdm.geometry import Point
from fdm.analysis import AnalysisStrategy


MIN_DENSITY = 1e-2


class BoundaryType(enum.Enum):
    NONE = 0
    FREE = 1
    FIXED = 2
    HINGE = 3


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
    POINT = 2


Field = collections.namedtuple('Field', ('type', 'properties'))
Load = collections.namedtuple('Load', ('type', 'properties'))
Boundary = collections.namedtuple('Boundary', ('type', 'properties'))
StaticsBoundary = collections.namedtuple('Boundary', ('scheme', 'value', 'replace'))
DynamicBoundary = collections.namedtuple('Boundary', ('scheme_1', 'scheme_2', 'replace'))


def static_boundary(scheme, value, replace=None):
    return StaticsBoundary(scheme, value, replace)


def dynamic_boundary(scheme_1, scheme_2, replace=None):
    return DynamicBoundary(scheme_1, scheme_2, replace)


class VirtualBoundaryStrategy(enum.Enum):
    AS_AT_BORDER = 0
    SYMMETRY = 1


def create(type_, *args, **kwargs):
    builder = create_builder()
    return builder(type_, *args, **kwargs)


def create_builder():
    builder = Strategy()
    builder.register('truss1d', create_for_truss_1d)
    builder.register('beam1d', create_for_beam_1d)
    return builder


def create_for_truss_1d(length, nodes_number):
    builder = Builder1d(length, nodes_number)
    builder.set_stiffness_factory({
        AnalysisStrategy.UP_TO_DOWN: create_truss1d_stiffness_operators_up_to_down,
        AnalysisStrategy.DOWN_TO_UP: create_truss1d_stiffness_operators_down_to_up,
    })
    builder.set_complex_boundary_factory(create_complex_truss_bcs())
    return builder


def create_for_beam_1d(length, nodes_number):
    builder = Builder1d(length, nodes_number)
    builder.set_stiffness_factory({
        AnalysisStrategy.UP_TO_DOWN: create_beam1d_stiffness_operators_up_to_down,
        AnalysisStrategy.DOWN_TO_UP: create_beam1d_stiffness_operators_down_to_up,
    })
    builder.set_complex_boundary_factory(create_complex_beam_bcs())
    return builder


StiffnessInput = collections.namedtuple('StiffnessInput', (
    'mesh', 'length', 'span', 'strategy', 'young_modulus_controller', 'moment_of_inertia_controller'
))


BCsInput = collections.namedtuple('BCsInput', (
    'mesh', 'length', 'span', 'virtual_nodes_strategy', 'moment_of_inertia_controller'
))


class Builder1d:
    def __init__(self, length, nodes_number):
        self._length = length
        self._nodes_number = nodes_number
        self._stiffness_factory = None
        self._stiffness_operator_strategy = None
        self._complex_boundary_factory = null_complex_boundary_factory
        self._analysis_strategy = fdm.analysis.AnalysisStrategy.UP_TO_DOWN

        self._span = length / (nodes_number - 1)

        self._context = {
            'analysis_type': None,
            'left_virtual_nodes_number': 0,
            'right_virtual_nodes_number': 0,
            'boundary': {
                Side.LEFT: BoundaryType.FIXED,
                Side.RIGHT: BoundaryType.FIXED,
            },
            'stiffness_operator_strategy': 'standard',
            'virtual_boundary_strategy': 'null',
            'field': Field(FieldType.NONE, ()),
            'load': LoadType.NONE,
            'stiffness_to_density_relation': None,
            'add_middle_nodes': False,
        }

        self.density_controller = self._create_value_controller('uniform', value=1.)
        self.young_modulus_controller = self._create_value_controller('uniform', value=1.)
        self.moment_of_inertia_controller = self._create_value_controller('uniform', value=1.)

    def set_analysis_strategy(self, strategy):
        self._analysis_strategy = strategy
        return self

    def set_stiffness_factory(self, factory):
        self._stiffness_factory = factory

    def set_complex_boundary_factory(self, factory):
        self._complex_boundary_factory = factory

    def set_analysis_type(self, _type):
        self._context['analysis_type'] = fdm.analysis.AnalysisType[_type]
        return self

    def set_density_controller(self, _type, *args, **options):
        self.density_controller = self._create_value_controller(_type, *args, **options)
        return self

    def set_boundary(self, side, _type, **opts):
        self._context['boundary'][side] = Boundary(_type, opts)
        return self

    def set_stiffness_operator_strategy(self, strategy):
        self._context['stiffness_operator_strategy'] = strategy
        return self

    def set_virtual_boundary_strategy(self, _type):
        self._context['virtual_boundary_strategy'] = _type
        return self

    def set_field(self, _type, **properties):
        self._context['field'] = Field(_type, properties)
        return self

    def set_load(self, _type, **properties):
        self._context['load'] = Load(_type, properties)
        return self

    def set_stiffness_to_density_relation(self, _type, **options):
        self._context['stiffness_to_density_relation'] = RELATIONS[_type](**options)
        return self

    def set_young_modulus_controller(self, _type, *args, **options):
        self.young_modulus_controller = self._create_value_controller(_type, *args, **options)
        return self

    def set_moment_of_inertia_controller(self, _type, *args, **options):
        self.moment_of_inertia_controller = self._create_value_controller(_type, *args, **options)
        return self

    def _create_value_controller(self, _type, *args, **options):
        return VALUE_CONTROLLERS[_type](self._length, self._nodes_number, *args, **options)

    def add_virtual_nodes(self, left, right):
        self._context['left_virtual_nodes_number'] = left
        self._context['right_virtual_nodes_number'] = right
        return self

    def add_middle_nodes(self):
        self._context['add_middle_nodes'] = True
        return self

    def create(self):
        assert self.density_controller is not None, "Define density controller first."
        assert self.young_modulus_controller is not None, "Define Young modulus controller first."
        assert self._context['analysis_type'] is not None, "Define analysis type first."
        mesh = self._create_mesh()

        return fdm.Model(
            self._create_mesh(), self._create_template(mesh),
            self._create_complex_bcs(mesh), self._analysis_strategy
        )

    def _create_mesh(self):
        vn_left = [-self._span * (i + 1) for i in range(self._context['left_virtual_nodes_number'])]
        vn_right = [self._length + self._span * (i + 1) for i in range(self._context['right_virtual_nodes_number'])]

        builder = (
            fdm.Mesh1DBuilder(self._length)
            .add_uniformly_distributed_nodes(self._nodes_number)
            .add_virtual_nodes(*(vn_left + vn_right))

        )
        if self._context['add_middle_nodes']:
            builder.add_middle_nodes()
        return builder.create()

    def _create_template(self, mesh):
        assert self._stiffness_factory is not None, 'Set stiffness factory first'

        _type = self._context['analysis_type']

        stiffness_stencils = self._create_stiffness_stencils(mesh)

        if self._analysis_strategy == fdm.analysis.AnalysisStrategy.UP_TO_DOWN:
            creator = self._create_template_for_up_to_down
        elif self._analysis_strategy == fdm.analysis.AnalysisStrategy.DOWN_TO_UP:
            creator = self._create_template_for_down_to_up
        else:
            raise NotImplementedError

        if _type == fdm.AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS:
            load_vector = LoadVectorValueFactory(self._length, self._span, self.density_controller, self._context)
            rhs = load_vector
        elif _type == fdm.AnalysisType.EIGENPROBLEM:
            mass_stencils = MassSchemeFactory(self._length, self._span, self.density_controller, self._context)
            rhs = mass_stencils
        else:
            raise AttributeError

        return creator(mesh, stiffness_stencils, rhs)

    def _create_template_for_up_to_down(self, mesh, stiffness_stencils, rhs):
        def get_expander(point):
            return stiffness_stencils, rhs(point)

        return fdm.equation.Template(get_expander)

    def _create_template_for_down_to_up(self, mesh, stiffness_stencils, rhs):
        def rhs_caller(function):
            def call(p):
                return function(p)(p)
            return call

        return [(mesh.real_nodes, (stiffness_stencils, rhs_caller(rhs)))]

    def _create_stiffness_stencils(self, mesh):
        data = StiffnessInput(
            mesh, self._length, self._span,
            self._context['stiffness_operator_strategy'],
            self._get_corrected_young_modulus,
            self.moment_of_inertia_controller,
        )

        return self._stiffness_factory[self._analysis_strategy](data)

    def _get_corrected_young_modulus(self, point):
        correction = self._context['stiffness_to_density_relation']
        scale = correction(max(self.density_controller.get(point), MIN_DENSITY)) if correction else 1.
        return self.young_modulus_controller(point) * scale

    def _create_complex_bcs(self, mesh):
        data = BCsInput(
            mesh, self._length, self._span,
            self._context['virtual_boundary_strategy'],
            self.moment_of_inertia_controller
        )

        return self._complex_boundary_factory(
            self._context['analysis_type'], self._context['boundary'], data)

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
    def moment_of_inertia(self):
        return self._revolve_for_points(self.moment_of_inertia_controller.get)

    @property
    def points(self):
        return [Point(self._span * i) for i in range(self._nodes_number)]

    def _revolve_for_points(self, _callable):
        return list(map(_callable, self.points))

    def create_density_getter(self):
        return self.density.get


class Strategy(object):
    def __init__(self):
        self._registry = {}

    def register(self, name, strategy):
        self._registry[name] = strategy

    def __call__(self, name, *args, **kwargs):
        return self._registry[name](*args, **kwargs)


def create_truss1d_stiffness_operators_up_to_down(data):
    span = data.span
    mesh = data.mesh

    first_node, last_node = mesh.real_nodes[0], mesh.real_nodes[-1]
    strains = fdm.Operator(fdm.Stencil.central(span=span))
    stresses = fdm.Number(data.young_modulus_controller) * strains
    stresses_derivative_central = fdm.Operator(fdm.Stencil.central(span=span), stresses)
    stresses_derivative_forward = fdm.Operator(fdm.Stencil.forward(span=span), stresses)
    stresses_derivative_backward = fdm.Operator(fdm.Stencil.backward(span=span), stresses)

    def dispatcher(point):
        if point == first_node:
            return stresses_derivative_forward
        elif point == last_node:
            return stresses_derivative_backward
        else:
            return stresses_derivative_central
    return fdm.DynamicElement(dispatcher)


def create_truss1d_stiffness_operators_down_to_up(data):
    span = data.span

    strains = fdm.Operator(fdm.Stencil.central(span=span))
    stresses = fdm.Number(data.young_modulus_controller) * strains
    stresses_derivative = fdm.Operator(fdm.Stencil.central(span=span), stresses)

    real_nodes = data.mesh.real_nodes
    element_1 = limit_element(stresses_derivative, real_nodes)

    return [element_1]


def create_beam1d_stiffness_operators_up_to_down(data):
    span = data.span
    central_stencil = fdm.Stencil.central(span=span)

    EI = fdm.Number(data.young_modulus_controller) * fdm.Number(1.)

    class_eq_central = fdm.Operator(
        central_stencil, fdm.Operator(
            central_stencil, fdm.Operator(
                central_stencil, fdm.Operator(
                    central_stencil, EI
                )
            )
        )
    )

    return class_eq_central


def create_beam1d_stiffness_operators_down_to_up(data):
    span = data.span

    central_stencil = fdm.Stencil.central(span=span)

    EI = fdm.Number(data.young_modulus_controller) * fdm.Number(1.)

    second_derivative = fdm.Operator(
        central_stencil, fdm.Operator(
            central_stencil
        )
    )

    second_derivative_EI = fdm.Operator(
        central_stencil, fdm.Operator(
            central_stencil, EI
        )
    )

    real_nodes = data.mesh.real_nodes
    virtual_nodes = data.mesh.virtual_nodes
    hvn = int(len(virtual_nodes)/2.)
    left_virtual_nodes = virtual_nodes[:hvn]
    right_virtual_nodes = virtual_nodes[hvn:]
    element_1 = limit_element(second_derivative_EI, real_nodes + left_virtual_nodes[:-2] + right_virtual_nodes[:-2])
    element_2 = limit_element(second_derivative, real_nodes)
    return [element_1, element_2]


def limit_element(element, points):
    def get(p):
        return element if p in points else fdm.Stencil.null()
    return fdm.DynamicElement(get)


def null_complex_boundary_factory(*args, **kwargs):
    return ()


def create_complex_beam_bcs():
    strategy = Strategy()
    strategy.register(fdm.AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, create_beam_statics_bcs)
    strategy.register(fdm.AnalysisType.EIGENPROBLEM, create_beam_eigenproblem_bc)
    return strategy


def create_beam_statics_bcs(boundary, data):
    span = data.span
    mesh = data.mesh

    begin_node, end_node = mesh.real_nodes[0], mesh.real_nodes[-1]
    begin_displacement_fixed = static_boundary(fdm.Scheme({begin_node: 1.}), 0.)
    end_displacement_fixed = static_boundary(fdm.Scheme({end_node: 1.}), 0.)

    derivative = fdm.Operator(fdm.Stencil.central(span))

    begin_rotation_fixed = static_boundary(derivative.expand(begin_node), 0.)
    end_rotation_fixed = static_boundary(derivative.expand(end_node), 0.)

    bcs = []
    if boundary[Side.LEFT].type == BoundaryType.FIXED:
        bcs += [
            begin_displacement_fixed,
            begin_rotation_fixed,
        ]

    if boundary[Side.RIGHT].type == BoundaryType.FIXED:
        bcs += [
            end_displacement_fixed,
            end_rotation_fixed,
        ]

    def p(s, base=0.):
        return Point(base + span * s)

    left_vbc_stencil = fdm.Stencil({p(-2): -1., p(-1): 4., p(0): -5., p(2): 5., p(3): -4., p(4): 1.})
    right_vbc_stencil = fdm.Stencil({p(2): -1., p(1): 4., p(0): -5., p(-2): 5., p(-3): -4., p(-4): 1.})

    virtual_nodes = mesh.virtual_nodes
    vn = len(virtual_nodes)
    hvn = int(vn / 2.)
    left_virtual_nodes = virtual_nodes[:hvn]
    right_virtual_nodes = virtual_nodes[hvn:]

    bcs += [static_boundary(left_vbc_stencil.expand(node), 0.) for node in left_virtual_nodes[:-2]]
    bcs += [static_boundary(right_vbc_stencil.expand(node), 0.) for node in right_virtual_nodes[:-2]]

    return bcs


def create_beam_eigenproblem_bc(boundary, data):  # todo:
    return []


def create_complex_truss_bcs():
    strategy = Strategy()
    strategy.register(fdm.AnalysisType.SYSTEM_OF_LINEAR_EQUATIONS, create_truss_statics_bcs)
    strategy.register(fdm.AnalysisType.EIGENPROBLEM, create_truss_eigenproblem_bc)
    return strategy


def create_truss_statics_bcs(boundary, data):
    mesh = data.mesh

    begin_node, end_node = mesh.real_nodes[0], mesh.real_nodes[-1]
    begin_displacement_fixed = static_boundary(fdm.Scheme({begin_node: 1.}), 0.)
    end_displacement_fixed = static_boundary(fdm.Scheme({end_node: 1.}), 0.)

    bcs = []
    if boundary[Side.LEFT].type == BoundaryType.FIXED:
        bcs += [
            begin_displacement_fixed,
        ]

    if boundary[Side.RIGHT].type == BoundaryType.FIXED:
        bcs += [
            end_displacement_fixed,
        ]

    return bcs


def create_truss_eigenproblem_bc(boundary, data):
    mesh = data.mesh

    begin_node, end_node = mesh.real_nodes[0], mesh.real_nodes[-1]
    begin_displacement_fixed = dynamic_boundary(fdm.Scheme({begin_node: 1.}), fdm.Scheme({}), replace=begin_node)
    end_displacement_fixed = dynamic_boundary(fdm.Scheme({end_node: 1.}), fdm.Scheme({}), replace=end_node)

    bcs = []
    if boundary[Side.LEFT].type == BoundaryType.FIXED:
        bcs += [
            begin_displacement_fixed,
        ]

    if boundary[Side.RIGHT].type == BoundaryType.FIXED:
        bcs += [
            end_displacement_fixed,
        ]

    return bcs


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

#


def create_free_vector_provider_factory(length):
    def zero(point):
        return 0.

    def build(load, density, field):
        if load.type == LoadType.MASS:
            return create_mass_load_provider(length, density, field)
        elif load.type == LoadType.POINT:
            return create_point_load_provider(load, length)
        elif load.type == LoadType.NONE:
            return zero
        else:
            raise NotImplementedError

    return build


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


def create_point_load_provider(load, length):
    props = load.properties
    L = length
    L1 = props['ordinate']
    P = props['magnitude']
    k1 = props.get('k1', 100.)
    if L1 <= 0.5:
        k2 = (L1 + 1)**(-200.) + 1.
    else:
        k2 = (-L1 + 2)**(-200.) + 1.

    def get(point):
        ksi = point.x/L
        a = k1*k2/(2.*math.tanh(k1/2.))
        b = 1./(math.cosh(k1*(ksi - L1))**2)
        value = a*b*P/L
        return value

    return get


class DensityController(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get(self, point):
        raise NotImplementedError

    def __call__(self, point):
        return self.get(point)


def calculate_directly_for_point(point, calculator):
    return calculator(point)


class SplineExtrapolation(enum.Enum):
    NONE = 0
    AS_FOR_EDGE = 1


class SplineValueController(DensityController):
    def __init__(self, points, computing_procedure=calculate_directly_for_point, order=3,
                 extrapolation=SplineExtrapolation.AS_FOR_EDGE, min_value=None, max_value=None):
        self._points = points
        self._computing_procedure = computing_procedure
        self._order = order
        self._extrapolation = extrapolation
        self._min_value = min_value
        self._max_value = max_value

        self._min_x, self._max_x = min(points), max(points)

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
        x = point.x

        if x < self._min_x and self._extrapolation == SplineExtrapolation.AS_FOR_EDGE:
            x = self._min_x
        elif x > self._max_x and self._extrapolation == SplineExtrapolation.AS_FOR_EDGE:
            x = self._max_x

        interpolator = scipy.interpolate.UnivariateSpline(
            self._points, self._values, k=self._order, ext=1, s=0.00000000001
        )
        value = float(interpolator(x))

        if self._min_value is not None:
            value = max(self._min_value, value)

        if self._max_value is not None:
            value = min(self._max_value, value)

        return value

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


class SegmentsValueController(SplineValueController):
    def __init__(self, length, segments_number=10, segments_in_module=1):
        self._length = length
        self._segments_number = segments_number
        self._segments_in_module = segments_in_module

        points, _ = self._compute_knot_points([1. for _ in range(segments_in_module)])
        SplineValueController.__init__(self, points, order=1)

    def update(self, values):
        assert len(values) == self._segments_in_module, "'Values' length must be equal segments number in module"
        xs, values = self._compute_knot_points(values)

        self.update_by_control_points(values)

    def _compute_knot_points(self, values=(1.,)):
        segment_length = self._length / self._segments_number
        segments_in_module = len(values)

        tol = 1e-3
        stiffness = []
        for i in range(self._segments_number + 1):
            x = i * segment_length
            si = int(math.fmod(i, segments_in_module))
            s = values[si]
            s2 = values[int(math.fmod(i - 1, segments_in_module))]
            if i in [0]:
                stiffness.append((x, s))
            elif i in [self._segments_number]:
                stiffness.append((x, s2))
            else:
                xp, xn = x - tol, x + tol
                stiffness.append((xp, s2))
                stiffness.append((xn, s))

        xs, values = zip(*stiffness)

        return xs, values


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


