import unittest

from fdm.parser import parse
from fdm.equation import Operator, Stencil


class ParseTest(unittest.TestCase):
    def test_Parse_OperatorWithoutElement_ResultOperatorWithCorrectStencil(self):
        string = 'Operator[Stencil[central, span=1]]'

        result = parse(string)

        expected = Operator(Stencil.central(span=1.))

        self._compare(expected, result)

    def _compare(self, operator_1, operator_2):
        return operator_1.expand(1) == operator_2.expand(1)

