import unittest

from fdm import Stencil
from fdm.builder import _create_virtual_nodes_bc, VirtualBoundaryStrategy
from fdm.geometry import Point


class CreateVirtualNodesBCTest(unittest.TestCase):
    def test_Call_Negative_Symmetry_ReturnCorrectStencil(self):

        x = -0.2

        result = _create_virtual_nodes_bc(x, VirtualBoundaryStrategy.SYMMETRY)[0]

        expected = Stencil({Point(0.): 1., Point(0.4): -1.})

        self.assertEqual(expected, result)

    def test_Call_Positive_Symmetry_ReturnCorrectStencil(self):

        x = 0.2

        result = _create_virtual_nodes_bc(x, VirtualBoundaryStrategy.SYMMETRY)[0]

        expected = Stencil({Point(0.): 1., Point(-0.4): -1.})

        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
