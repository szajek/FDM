import re
import collections
import enum

from fdm.equation import *


__all__ = ['Keyword', 'Tag', 'parse']


Keyword = collections.namedtuple('Keyword', ('tag', 'parameters', 'options'))


class Tag(enum.Enum):
    Operator = 'O'
    Stencil = 'S'

    def __init__(self, shortcut):
        self.shortcut = shortcut

    @classmethod
    def by_string(cls, tag):
        if '_shortcuts' not in cls.__dict__:
            cls._shortcuts = {}

        return cls[tag] if tag in cls.__members__ else cls._shortcuts.setdefault(tag, cls._get_by_shortcut(tag))

    @classmethod
    def _get_by_shortcut(cls, shortcut):
        for name, member in cls.__members__.items():
            if member.shortcut == shortcut:
                return member


def extract_keywords(string):
    m = re.match("([A-Za-z]+\[[^\[\]]*\])", string)
    return m.groups()


def extract_args(string):
    return string.split(',')


def parse_arguments(args):

    params = []
    opts = {}
    for arg in extract_args(args):
        if not arg:
            continue
        elif '=' in arg:
            key, value = arg.split('=')
            opts[key.strip()] = value.strip()
        else:
            params.append(arg.strip())
    return tuple(params), opts


def parse_keyword(string):
    m = re.match("([A-Za-z]+)(?:\[(.*)\])?", string)
    tag, args = m.groups()
    return Keyword(tag, *parse_arguments(args))


def parse(string):
    return create(parse_keyword(string))


# --------------------------------


def create_stencil(keyword):
    _type = keyword.parameters[0]

    factory = {
        'central': Stencil.central,
        'forward': Stencil.forward,
        'backward': Stencil.backward,
    }[_type]

    if 'span' in keyword.options:
        keyword.options['span'] = float(keyword.options['span'])

    return factory(**keyword.options)


def create_operator(keyword):
    return Operator(
        parse(keyword.parameters[0]),
        element=None
    )


# def create_position_dispatcher(pattern):  # todo:
#
#     if pattern == "C":
#         stencil = Stencil.central(1.)
#
#     elif pattern == "F":
#         stencil = Stencil.forward(1.)
#     elif pattern == "B":
#         stencil = Stencil.backward(1.)
#
#     def dispatcher(start, end, position):
#         return Operator(stencil)
#
#     return dispatcher


_dispatcher = {
    Tag.Operator: create_operator,
    Tag.Stencil: create_stencil,
}


def create(keyword):
    return _dispatcher[Tag.by_string(keyword.tag)](keyword)