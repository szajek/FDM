import collections
from .equation import Stencil, Operator, LinearEquationTemplate


__all__ = ['Model', 'create_bc']


Model = collections.namedtuple("Model", ('equation', 'domain', 'bcs'))


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


_bc_generators = {
    'dirichlet': _create_dirichlet_bc,
    'neumann': _create_neumann_bc,
}


def create_bc(_type, *args, **kwargs):
    return _bc_generators[_type](*args, **kwargs)
