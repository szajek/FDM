import abc
import collections
import enum
import functools
import math

import dicttools

__all__ = ['Scheme', 'Element', 'Stencil', 'DynamicElement', 'LazyOperation', 'Operator', 'Number',
           'NodeFunction', 'LinearEquationTemplate', 'Delta']


class Immutable(type):
    def __new__(msc, name, bases, nmspc):
        new_class = type(name, bases, nmspc)

        original_initialization = new_class.__init__

        def set_attribute(self, key, value):
            raise AttributeError("Attributes are immutable")

        def wrapper(self, *args, **kwargs):
            new_class.__setattr__ = object.__setattr__
            original_initialization(self, *args, **kwargs)
            new_class.__setattr__ = set_attribute

        new_class.__init__ = wrapper

        return new_class


class MutateMixin:
    def __init__(self, *fields):
        self._fields = fields

    def mutate(self, **kw):
        def get_keyword(definition):
            return definition if isinstance(definition, str) else definition[0]

        def get_value(definition):
            return kw.get(
                get_keyword(definition),
                getattr(self, definition if isinstance(definition, str) else definition[1])
            )

        return self.__class__(**{get_keyword(definition): get_value(definition) for definition in self._fields})


class FloatWrapper(metaclass=abc.ABCMeta):
    def __init__(self, value, ignore_for_math_operations=()):
        self._value = value
        self._ignore_for_math_operations = ignore_for_math_operations

    def __math__(self, _type, other):
        if isinstance(other, self._ignore_for_math_operations):
            return self._value

        if isinstance(other, (int, float)):
            return self._operators[_type](self._value, other)
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self.__math__('add', other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return self.__math__('mul', other)

    def __imul__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__math__('sub', other)

    def __truediv__(self, other):
        return self.__math__('div', other)

    def __pow__(self, other):
        return self.__math__('pow', other)

    _operators = {
        'mul': lambda a, b: a * b,
        'add': lambda a, b: a + b,
        'div': lambda a, b: a / b,
        'sub': lambda a, b: a - b,
        'pow': lambda a, b: a ** b,
    }


class Delta(FloatWrapper):
    def __init__(self, *values):
        self._values = values
        avg = sum(values) / len(values)
        FloatWrapper.__init__(self, avg)

    @property
    def average(self):
        return self._value

    @classmethod
    def from_connections(cls, *connections):
        return cls(*[c.length for c in connections])


def merge_weights(*weights):
    def merge(weights_1, weights_2):
        return {node_address: weights_1.get(node_address, 0.) + weights_2.get(node_address, 0.)
                for node_address in set(weights_1.keys()) | set(weights_2.keys())}

    return functools.reduce(merge, weights)


NODE_TOLERANCE = 1e-4


class Address:
    __slots__ = '_value', 'axis'

    def __init__(self, value, axes=None):
        self._value = value
        self.axes = axes if axes else (1,)


class Scheme(MutateMixin, metaclass=Immutable):
    __slots__ = 'weights', 'order'

    def __init__(self, weights, order=1):
        self.weights = dicttools.FrozenDict(weights)
        self.order = order

        MutateMixin.__init__(self, 'weights', 'order')

    def __iter__(self):
        return sorted(self.weights.items()).__iter__()

    def __len__(self):
        return len(self.weights)

    def __add__(self, other):
        if other is None:
            return self.duplicate()
        elif isinstance(other, Scheme):
            if not self._check_order_consistency(other):
                raise AttributeError("All schemes in addition operation must have the same order")
            return self.mutate(weights=merge_weights(self.weights, other.weights))
        elif isinstance(other, (int, float)):
            return self.shift(other)
        else:
            raise NotImplementedError

    def _check_order_consistency(self, other):
        return other.order == self.order

    def __iadd__(self, other):
        return self.__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def shift(self, number):
        return self.mutate(
            weights={node + number: weight for node, weight in self.weights.items()}
        )

    def to_coefficients(self, delta):
        delta = pow(delta, self.order)

        return Coefficients(merge_weights(*[
            self._distribute_to_closest_nodes(node_address, coeff / delta)
            for node_address, coeff in self.weights.items()
            ]))

    @staticmethod
    def _distribute_to_closest_nodes(address, value):
        modulo = math.fmod(address, 1.)
        abs_modulo = math.fabs(modulo)

        _w1, _w2 = (1. - abs_modulo), abs_modulo
        w1, w2 = (_w1, _w2) if modulo > 0. else (_w2, _w1)

        return {math.floor(address): w1 * value, math.ceil(address): w2 * value} if math.fabs(
            modulo) > NODE_TOLERANCE else \
            {int(address): value}

    def __mul__(self, other):

        if isinstance(other, (int, float)):
            return self.mutate(weights={idx: e * other for idx, e in self.weights.items()})
        else:
            raise NotImplementedError

    def __imul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):

        if isinstance(other, (int, float)):
            return self.mutate(weights={idx: e ** other for idx, e in self.weights.items()})
        else:
            raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, Scheme):
            return other.weights == self.weights and self.order == other.order
        else:
            raise NotImplementedError

    def __repr__(self):
        return "{name}: {data} of order {order}".format(
            name=self.__class__.__name__, data=self.weights, order=self.order)

    def duplicate(self):
        return self.mutate()

    @classmethod
    def from_number(cls, node_address, value):
        return Scheme({node_address: value}, order=0.)


class Coefficients(collections.Mapping):
    def __init__(self, coefficients):
        self._coefficients = coefficients

    def __getitem__(self, key):
        return self._coefficients[key]

    def __iter__(self):
        return self._coefficients.__iter__()

    def __len__(self):
        return len(self._coefficients)

    def to_value(self, output):
        return functools.reduce(lambda _sum, address: _sum + self[address] * output[address], self.keys(), 0.)

    def __repr__(self):
        return "{name}: {data}".format(name=self.__class__.__name__, data=self._coefficients)


def operate(scheme, element):
    if element is None:
        return scheme

    if not len(scheme):
        raise AttributeError("Empty scheme can not operate on anything")

    def to_scheme(_element, _address):
        expanded = _element.expand(_address)
        return Scheme.from_number(_address, expanded) if isinstance(expanded, (int, float)) else expanded

    addends = []
    for address, weight in scheme:
        element_scheme = to_scheme(element, address) * weight
        if not len(element_scheme):
            raise AttributeError("Empty scheme can not be operated by scheme")
        element_scheme = element_scheme.mutate(order=element_scheme.order + scheme.order)
        addends.append(element_scheme)

    return sum(addends, None)


class Element(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def expand(self, address, *args, **kw):
        raise NotImplementedError

    def __call__(self, node_address, *args, **kw):
        return self.expand(node_address, *args, **kw)

    def __add__(self, other):
        return self._create_lazy_operation_with(LazyOperation.summation, other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return self._create_lazy_operation_with(LazyOperation.multiplication, other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self._create_lazy_operation_with(LazyOperation.subtraction, other)

    def __isub__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._create_lazy_operation_with(LazyOperation.division, other)

    def __pow__(self, other):
        return self._create_lazy_operation_with(LazyOperation.power, other)

    def _create_lazy_operation_with(self, operation, other):
        if isinstance(other, Element):
            return operation(self, other)
        else:
            raise NotImplementedError

    def to_stencil(self, address):
        scheme = self.expand(address).shift(-address)
        return Stencil(scheme.weights, order=scheme.order)


class DynamicElement(Element):
    def __init__(self, builder):
        self._builder = builder

    def expand(self, address):
        return self._builder(address).expand(address)


class LazyOperation(Element):
    class Type(enum.Enum):
        MULTIPLICATION = 0
        SUMMATION = 1
        DIVISION = 2
        SUBTRACTION = 3
        POWER = 4

    def __init__(self, operator, element_1, element_2):
        self._operator = operator
        self._element_1 = element_1
        self._element_2 = element_2

    def __call__(self):
        self.expand()

    def expand(self, *args):
        return self._operators[self._operator](
            self._element_1.expand(*args),
            self._element_2.expand(*args)
        )

    @classmethod
    def summation(cls, *args):
        return cls(cls.Type.SUMMATION, *args)

    @classmethod
    def subtraction(cls, *args):
        return cls(cls.Type.SUBTRACTION, *args)

    @classmethod
    def multiplication(cls, *args):
        return cls(cls.Type.MULTIPLICATION, *args)

    @classmethod
    def division(cls, *args):
        return cls(cls.Type.DIVISION, *args)

    @classmethod
    def power(cls, *args):
        return cls(cls.Type.POWER, *args)

    _operators = {
        Type.MULTIPLICATION: lambda a, b: a * b,
        Type.SUMMATION: lambda a, b: a + b,
        Type.DIVISION: lambda a, b: a / b,
        Type.SUBTRACTION: lambda a, b: a - b,
        Type.POWER: lambda a, b: a ** b,
    }

    def __eq__(self, other):
        return self._operator == other._operator and \
               self._element_1 == other._element_1 and \
               self._element_2 == other._element_2


class Stencil(Element, MutateMixin):
    @classmethod
    def forward(cls, span=1.):
        return cls.by_addresses(0., span)

    @classmethod
    def backward(cls, span=1.):
        return cls.by_addresses(-span, 0.)

    @classmethod
    def central(cls, span=2.):
        return cls.by_addresses(- span / 2., span / 2.)

    @classmethod
    def by_addresses(cls, address_1, address_2):
        _range = address_2 - address_1
        weight = 1. / _range
        return cls(
            {address_1: -weight, address_2: weight}
        )

    @classmethod
    def uniform(cls, left_range, right_range, resolution, weights_provider, **kwargs):
        _range = right_range + left_range
        delta = _range / resolution
        stencil_nodes_addresses = [-left_range + i * delta for i in range(int(resolution + 1))]
        return cls(
            {node_address: weights_provider(i, node_address) for i, node_address in enumerate(stencil_nodes_addresses)},
            **kwargs)

    @classmethod
    def from_scheme(cls, scheme):
        return Stencil(scheme.weights, order=scheme.order)

    def __init__(self, weights, axis=1, order=1.):
        self._weights = weights
        self._axis = axis
        self._order = order

        MutateMixin.__init__(self, ('weights', '_weights'), ('axis', '_axis'), ('order', '_order'))

    @property
    def order(self):
        return self._order

    @property
    def start(self):
        return min(self._weights.keys())

    @property
    def end(self):
        return max(self._weights.keys())

    def expand(self, address):
        return Scheme(self._weights, order=self._order) + address

    def scale(self, multiplier):
        return self.mutate(weights={address * multiplier: value for address, value in self._weights.items()})

    def __repr__(self):
        return "{name}: {data} of order {order}".format(
            name=self.__class__.__name__, data=self._weights, order=self._order)

    def __eq__(self, other):
        if isinstance(other, Stencil):
            return self._weights == other._weights and self._axis == other._axis and self._order == other._order
        else:
            return False


class Operator(Element):
    def __init__(self, stencil, element=None):
        self._stencil = stencil
        self._element = element

    def expand(self, address):
        return operate(
            self._stencil.expand(address),
            self._dispatch(address) if callable(self._element) else self._element
        )

    def _dispatch(self, reference):
        def build_child_operator(address):
            relative = address - reference
            return self._element(
                relative,
                {
                    self._stencil.start: -1,
                    self._stencil.end: 1,
                }.get(relative, 0)
            )

        return DynamicElement(build_child_operator)

    def __repr__(self):
        return "{name}: {scheme}".format(name=self.__class__.__name__, scheme=self._stencil)


class Number(Element):
    def __init__(self, value):
        self._value = value

    def expand(self, address):
        return self._value(address) if callable(self._value) else self._value

    def __eq__(self, other):
        return self._value == other._value


#

def linear_interpolator(x, x_1, x_2, value_1, value_2):
    return value_1 + (value_2 - value_1) / (x_2 - x_1) * x


class NodeFunction:
    def __init__(self, _callable, interpolator=None):
        self._callable = _callable
        self._interpolator = interpolator

    def __call__(self, node_address, *args, **kwargs):
        return self.get(node_address, *args, **kwargs)

    def get(self, node_address):
        int_node_number = int(node_address)
        if int_node_number != node_address:
            return self._interpolate(int_node_number, node_address)
        else:
            return self._callable(node_address)

    def _interpolate(self, int_node_number, node_address):
        if self._interpolator is None:
            from fdm import logger
            logger.solver.debug('NodeFunction: node address {addr} is provided but interpolator is not defined'.format(
                addr=node_address))
            return self._callable(int(round(node_address)))
        else:
            return self._interpolator(node_address - int_node_number, int_node_number, int_node_number + 1,
                                      self._callable(int_node_number), self._callable(int_node_number + 1))

    @classmethod
    def with_linear_interpolator(cls, _callable):
        return cls(_callable, linear_interpolator)


#

LinearEquationTemplate = collections.namedtuple('LinearEquationTemplate', ('operator', 'free_value'))