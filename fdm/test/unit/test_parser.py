import unittest
from mock import patch, MagicMock

from fdm.equation import Operator, Stencil
from fdm.parser import create_stencil, Keyword, parse_keyword, create_operator, parse, create, Tag, parse_arguments


class TagTest(unittest.TestCase):
    def test_ByString_FullName_ReturnTag(self):

        result = Tag.by_string('Operator')

        expected = Tag.Operator

        self.assertEqual(expected, result)

    def test_ByString_Shortcut_ReturnTag(self):
        result = Tag.by_string('O')

        expected = Tag.Operator

        self.assertEqual(expected, result)


@patch('fdm.parser.create')
class ParseTest(unittest.TestCase):
    def test_Call_SingleKeyword_ExtractAndSendToCreate(self, fake_create):

        pattern = 'fake[1]'

        self._parse(pattern)

        fake_create.assert_called_with(
            Keyword('fake', ('1',), {})
        )

    def test_Call_NestedKeyword_ExtractParentAnd(self, fake_create):

        pattern = 'fake[1]'

        self._parse(pattern)

        fake_create.assert_called_with(
            Keyword('fake', ('1',), {})
        )

    def _parse(self, *args, **kwargs):
        return parse(*args, **kwargs)


class ParseArgumentTest(unittest.TestCase):
    def test_Call_NoArgs_ReturnEmptyParamsListAndOptsDict(self):

        pattern = ''

        result = self._parse(pattern)

        expected = (), {}

        self.assertEqual(expected, result)

    def test_Call_SingleArgs_ReturnParamsListAndEmptyOptsDict(self):

        pattern = 'a'

        result = self._parse(pattern)

        expected = ('a',), {}

        self.assertEqual(expected, result)

    def test_Call_ManyParams_ReturnParamsListAndEmptyOptsDict(self):

        pattern = 'a, c,d,  w'

        result = self._parse(pattern)

        expected = ('a', 'c', 'd', 'w'), {}

        self.assertEqual(expected, result)

    def test_Call_SingleOpts_ReturnEmptyParamsListAndOneOption(self):

        pattern = 'a=w'

        result = self._parse(pattern)

        expected = (), {'a': 'w'}

        self.assertEqual(expected, result)

    def test_Call_ParamsAndOpts_ReturnParamsListAndOneOptions(self):

        pattern = 'c,a=w'

        result = self._parse(pattern)

        expected = ('c',), {'a': 'w'}

        self.assertEqual(expected, result)

    def test_Call_ParamsAsKeyword_ReturnKeywordDefinitionIntact(self):

        pattern = 'O[1,2]'

        result = self._parse(pattern)

        expected = ('O[1,2]',), {}

        self.assertEqual(expected, result)

    def _parse(self, *args, **kwargs):
        return parse_arguments(*args, **kwargs)


class ParseKeywordTest(unittest.TestCase):
    def test_Call_Full_ReturnKeywordWithParametersAndOptions(self):
        pattern = 'fake[1, 3,abc=4,www=cc]'

        result = self._parse(pattern)

        expected = Keyword('fake', ('1', '3',), {'abc': '4', 'www': 'cc'})

        self.assertEqual(expected, result)

    def test_Call_NestedIncluded_ReturnKeywordWithNestedPartIntact(self):
        pattern = 'O[2, S[3, w=d], w=c]'

        result = self._parse(pattern)

        expected = Keyword('O', ('2', 'S[3, w=d]',), {'w': 'c'})

        self.assertEqual(expected, result)

    def _parse(self, pattern):
        return parse_keyword(pattern)


class CreateStencilTest(unittest.TestCase):
    def test_Call_Central_ReturnStencilWithGivenWeights(self):
        keyword = Keyword('S', ('central',), {'span': '3.'})

        result = self._create(keyword)

        expected = Stencil.central(span=3.)

        self.assertEqual(expected, result)

    def _create(self, keyword):
        return create_stencil(keyword)


class CreateOperatorTest(unittest.TestCase):
    @patch('fdm.parser.parse')
    def test_Call_FakeStencilNoElement_ReturnOperatorWithElementNone(self, parse):
        fake_stencil = Stencil.central(5.)
        parse.return_value = fake_stencil

        keyword = Keyword('O', ('fake_stencil',), {})

        result = self._create(keyword)

        expected = Operator(fake_stencil)

        self.assertTrue(self._compare(expected, result))

    def test_Call_RealStencilNoElement_ReturnOperatorWithElementNone(self):

        keyword = Keyword('O', ('S[central, span=3.]',), {})

        result = self._create(keyword)

        expected = Operator(Stencil.central(span=3.))

        self.assertTrue(self._compare(expected, result))

    def _create(self, keyword):
        return create_operator(keyword)

    def _compare(self, operator_1, operator_2):
        return operator_1.expand(1) == operator_2.expand(1)

# class DispatcherFactoryTest(unittest.TestCase):
#     def _test_Call_F_ReturnDispatcherWithGivenPattern(self):
#         dispatcher = self._create('C(1)')
#
#         expected = fdm.Operator(fdm.Stencil.central())
#
#         self.assertEqual(expected, dispatcher(-1, 2, -1))
#
#     def _create(self, pattern):
#         return create_position_dispatcher(pattern)


class CreateTest(unittest.TestCase):
    def test_Call_SingleKeyword_CreateCorrectObject(self):

        keyword = Keyword('S', ('central',), {'span': '3'})

        result = self._create(keyword)

        expected = Stencil.central(span=3.)

        self.assertEqual(expected, result)

    def _create(self, *args, **kwargs):
        return create(*args, **kwargs)

if __name__ == '__main__':
    unittest.main()
