import collections
import enum

import numpy as np

from .equation import Stencil, Operator, LinearEquationTemplate
from .geometry import Point


__all__ = ['Model', 'create_bc']


Model = collections.namedtuple("Model", ('equation', 'mesh', 'bcs'))


def _create_dirichlet_bc(value=0.):
    return LinearEquationTemplate(
        operator=Operator(
            Stencil({0: 1.})
        ),
        free_value=lambda node_address: value,
    )


def _create_neumann_bc(stencil, value=0.):
    return LinearEquationTemplate(
        operator=Operator(
            stencil
        ),
        free_value=lambda node_address: value,
    )


def _create_bc_by_equation(operator, free_value=0.):
    return LinearEquationTemplate(
        operator=operator,
        free_value=lambda node_address: free_value,
    )


class VirtualBoundaryStrategy(enum.Enum):
    AS_AT_BORDER = 0
    SYMMETRY = 1


def _create_virtual_nodes_bc(x, strategy):
    m = {
        VirtualBoundaryStrategy.SYMMETRY: 2.,
        VirtualBoundaryStrategy.AS_AT_BORDER: 1.,
    }[strategy]
    return LinearEquationTemplate(
            Stencil(
                {
                    0.: 1.,
                    -np.sign(x)*m*abs(x): -1.
                }
            ),
            lambda p: 0.
        )


_bc_generators = {
    'dirichlet': _create_dirichlet_bc,
    'neumann': _create_neumann_bc,
    'equation': _create_bc_by_equation,
    'virtual_node': _create_virtual_nodes_bc,
}


def create_bc(_type, *args, **kwargs):
    return _bc_generators[_type](*args, **kwargs)


