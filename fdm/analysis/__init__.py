import enum

from .analyzer import *

from . import up_to_down
from . import down_to_up


class Strategy(enum.Enum):
    UP_TO_DOWN = 0
    DOWN_TO_UP = 1


_solvers = {
    Strategy.UP_TO_DOWN: up_to_down.get_solvers(),
    Strategy.DOWN_TO_UP: down_to_up.get_solvers()
}


def solve(_type, model, strategy=Strategy.UP_TO_DOWN, spy=None):
    solver = _solvers[strategy][_type]
    solver.install_spy(spy)
    return solver(model)
