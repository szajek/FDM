import enum

from .analyzer import *

from . import up_to_down
from . import down_to_up


class AnalysisStrategy(enum.Enum):
    UP_TO_DOWN = 0
    DOWN_TO_UP = 1


_solvers = {
    AnalysisStrategy.UP_TO_DOWN: up_to_down.get_solvers(),
    AnalysisStrategy.DOWN_TO_UP: down_to_up.get_solvers()
}


def solve(_type, model, spy=None):
    solver = _solvers[model.analysis_strategy][_type]
    solver.install_spy(spy)
    return solver(model)
