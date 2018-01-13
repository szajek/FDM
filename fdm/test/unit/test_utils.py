import unittest

from fdm.utils import Immutable


class Foo(metaclass=Immutable):
    def __init__(self):
        self.boo = 1


class ImmutableTest(unittest.TestCase):
    def test_Setattr_Always_RaiseAttributeError(self):

        obj = Foo()

        with self.assertRaises(AttributeError):
            obj.boo = 2