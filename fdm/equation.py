import abc
import collections
import enum
import functools
import math
import numpy as np

import dicttools

from fdm.geometry import Point, FreeVector, Vector
from fdm.utils import Immutable

__all__ = ['Scheme', 'Element', 'Stencil', 'DynamicElement', 'LazyOperation', 'Operator', 'Number',
           'LinearEquation', 'LinearEquationTemplate']


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


def merge_weights(*weights):

    merged = collections.defaultdict(int)

    for w in weights:
        for point, factor in w.items():
            merged[point] += factor

    return merged


class Scheme(collections.Mapping):
    __slots__ = '_weights', '_sorted_points'

    def __init__(self, weights):
        self._weights = weights

        self._sorted_points = None

    def __iter__(self):
        return iter(self._weights)

    def __len__(self):
        return len(self._weights)

    def __getitem__(self, key):
        try:
            return self._weights[key]
        except KeyError:
            pass

    def __add__(self, other):
        if other is None:
            return self.duplicate()
        elif isinstance(other, Scheme):
            return Scheme(merge_weights(self._weights, other._weights))
        elif isinstance(other, Point):
            return self.shift(FreeVector(other))
        elif isinstance(other, FreeVector):
            return self.shift(other)
        else:
            raise NotImplementedError

    def __iadd__(self, other):
        return self.__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def shift(self, vector):
        return Scheme({point + vector: weight for point, weight in self._weights.items()})

    def __mul__(self, other):

        if isinstance(other, (int, float, np.float64)):
            return Scheme({point: e * other for point, e in self._weights.items()})
        else:
            raise NotImplementedError

    def __imul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):

        if isinstance(other, (int, float)):
            return Scheme({point: e ** other for point, e in self._weights.items()})
        else:
            raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, Scheme):
            return other._weights == self._weights
        else:
            raise NotImplementedError

    def __repr__(self):
        return "{name}: {data}".format(
            name=self.__class__.__name__, data=self._weights)

    def duplicate(self):
        return Scheme(self._weights)

    @property
    def start(self):
        return self._get_sorted_points()[0]

    @property
    def end(self):
        return self._get_sorted_points()[-1]

    def _get_sorted_points(self):
        if self._sorted_points is None:
            self._sorted_points = sorted(self._weights.keys(), key=lambda item: item.x)
        return self._sorted_points

    @classmethod
    def from_number(cls, point, value):
        return Scheme({point: value})

    def to_value(self, output):
        return functools.reduce(lambda _sum, point: _sum + self._weights[point] * output[point], self._weights.keys(), 0.)

    def distribute(self, distributor):
        return Scheme(
            merge_weights(*[distributor(point, factor) for point, factor in self._weights.items()])
        )


def operate(scheme, element):
    if element is None:
        return scheme

    if not len(scheme):
        raise AttributeError("Empty scheme can not operate on anything")

    def to_scheme(_element, _address):
        expanded = _element.expand(_address)
        return Scheme.from_number(_address, expanded) if isinstance(expanded, (int, float)) else expanded

    addends = []
    for address, weight in scheme.items():
        element_scheme = to_scheme(element, address) * weight
        if not len(element_scheme):
            raise AttributeError("Empty scheme can not be operated by scheme")
        addends.append(element_scheme)

    return sum(addends, None)


class Element:
    @abc.abstractmethod
    def expand(self, point, *args, **kw):
        raise NotImplementedError

    def __call__(self, point, *args, **kw):
        return self.expand(point, *args, **kw)

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

    def to_stencil(self, point):
        return Stencil.from_scheme(
            self.expand(point).shift(FreeVector(-point))
        )


class DynamicElement(Element):
    def __init__(self, builder):
        self._builder = builder

    def expand(self, point):
        return self._builder(point).expand(point)


def assert_not_numpy_float(value):
    if isinstance(value, np.float64):
        raise AttributeError("Float or Int value required. Numpy float provided")
    return value


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

    def __call__(self, *args):
        self.expand(*args)

    def expand(self, *args):
        return self._operators[self._operator](
            assert_not_numpy_float(self._element_1.expand(*args)),
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


class Stencil(Element):
    __slots__ = '_weights'

    @classmethod
    def forward(cls, span=1.):
        return cls.by_two_points(Point(0.), Point(span))

    @classmethod
    def backward(cls, span=1.):
        return cls.by_two_points(Point(-span), Point(0.))

    @classmethod
    def central(cls, span=2.):
        return cls.by_two_points(Point(-span / 2.), Point(span / 2.))

    @classmethod
    def by_two_points(cls, point_1, point_2):
        weight = 1. / Vector(point_1, point_2).length
        return cls(
            {point_1: -weight, point_2: weight}
        )

    @classmethod
    def uniform(cls, point_1, point_2, resolution, weights_provider, **kwargs):
        _range = Vector(point_1, point_2).length
        delta = _range / resolution
        stencil_points = [point_1 + FreeVector(Point(i * delta)) for i in range(int(resolution + 1))]
        return cls(
            {point: weights_provider(i, point) for i, point in enumerate(stencil_points)},
            **kwargs)

    @classmethod
    def from_scheme(cls, scheme):
        return Stencil(scheme._weights)

    def __init__(self, weights):
        self._weights = weights

    def expand(self, point):
        return Scheme(self._weights) + point

    def scale(self, multiplier):
        return Stencil({point * multiplier: value for point, value in self._weights.items()})

    def __repr__(self):
        return "{name}: {data}".format(
            name=self.__class__.__name__, data=self._weights)

    def __eq__(self, other):
        if isinstance(other, Stencil):
            return self._weights == other._weights
        else:
            return False


class Operator(Element):
    def __init__(self, stencil, element=None):
        self._stencil = stencil
        self._element = element

    def expand(self, point):
        operator_scheme = self._stencil.expand(point)
        return operate(
            operator_scheme,
            self._dispatch(point, operator_scheme) if self._is_element_dispachable() else self._element
        )

    def _dispatch(self, reference, operator_scheme):
        def build_child_operator(point):
            return self._element(
                operator_scheme.start - FreeVector(reference),
                operator_scheme.end - FreeVector(reference),
                point - FreeVector(reference),
            )

        return DynamicElement(build_child_operator)

    def __repr__(self):
        return "{name}: {scheme}".format(name=self.__class__.__name__, scheme=self._stencil)

    def _is_element_dispachable(self):
        return not (isinstance(self._element, Element) or self._element is None)


class Number(Element):
    def __init__(self, value):
        self._value = value

    def expand(self, point):
        return self._value(point) if callable(self._value) else self._value

    def __eq__(self, other):
        return self._value == other._value


#


LinearEquationTemplate = collections.namedtuple('LinearEquationTemplate', ('operator', 'free_value'))
LinearEquation = collections.namedtuple('LinearEquation', ('scheme', 'free_value'))