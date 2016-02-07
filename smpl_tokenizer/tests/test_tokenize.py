__title__ = 'smpl_tokenizer'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@linux.com'
__created_on__ = '02/06/2016'
__copyright__ = "smpl_tokenizer Copyright (C) 2015  Steven Cutting"


import pytest
import toolz as tlz
reduce_c = tlz.curry(tlz.reduce)

from smpl_tokenizer import tokenize as tkn


@pytest.mark.parametrize("n,string,expected",
                         [(1, "foo-bar", [("foo", ), ("bar", )]),
                          (1, "foobazbar", [("foobazbar", )]),
                          (1, "foo*bar*baz", [("foo", ), ("bar", ), ("baz", )]),
                          (2, "foo*bar*baz", [("foo", "bar"), ("bar", "baz")]),
                          ])
def test__ngram_tuples(n, string, expected):
    assert(list(tkn.ngram_tuples(n, string)) == expected)


@pytest.mark.parametrize("n,string,expected",
                         [(1, "foo-bar", ["foo", "bar"]),
                          (2, "foo-bar", ["foo_bar"]),
                          (3, "foo-bar", []),
                          (1, "foobazbar", ["foobazbar"]),
                          (2, "foobazbar", []),
                          (3, "foobazbar", []),
                          (1, "foo*bar*baz", ["foo", "bar", "baz"]),
                          (2, "foo*bar*baz", ["foo_bar", "bar_baz"]),
                          (3, "foo*bar*baz", ["foo_bar_baz"]),
                          ])
def test__ngram(n, string, expected):
    assert(list(tkn.ngram(n, string)) == expected)


@pytest.mark.parametrize("string,expected",
                         [("foo-bar", ["foo", "bar"]),
                          ("foobazbar", ["foobazbar"]),
                          ("foo*bar*baz", ["foo", "bar", "baz"]),
                          ])
def test__uni_gram(string, expected):
    assert(list(tkn.uni_gram(string)) == expected)


@pytest.mark.parametrize("string,expected",
                         [("foo-bar", ["foo_bar"]),
                          ("foobazbar", []),
                          ("foo*bar*baz", ["foo_bar", "bar_baz"]),
                          ])
def test__bi_gram(string, expected):
    assert(list(tkn.bi_gram(string)) == expected)


@pytest.mark.parametrize("string,expected",
                         [("foo-bar", []),
                          ("foobazbar", []),
                          ("foo*bar*baz", ["foo_bar_baz"]),
                          ])
def test__tri_gram(string, expected):
    assert(list(tkn.tri_gram(string)) == expected)


sum_tally_tuples = lambda tpls: reduce_c(lambda x, y: x+y[1], tpls, 0)


@pytest.mark.parametrize("string,length,total,parser",
                         [(tlz.reduce(lambda x, y: x+y,
                                      ["aaa " * 20,
                                       "bbb " * 10,
                                       "ccc " * 3,
                                       "ddd " * 1],
                                      ), 4, 34, tkn.uni_gram),

                          ])
def test___bag_of_words(string, length, total, parser):
    bow = tkn.bag_of_words(parser, string)
    assert(len(bow) == length)
    assert(sum_tally_tuples(bow) == total)


@pytest.mark.parametrize("string,length,total",
                         [(tlz.reduce(lambda x, y: x+y,
                                      ["aaa " * 20,
                                       "bbb " * 10,
                                       "ccc " * 3,
                                       "ddd " * 1],
                                      ), 4, 34),

                          ])
def test___bow_uni_gram(string, length, total):
    bow = tkn.bow_uni_gram(string)
    assert(len(bow) == length)
    assert(sum_tally_tuples(bow) == total)


@pytest.mark.parametrize("string,length,total",
                         [(tlz.reduce(lambda x, y: x+y,
                                      ["aaa " * 20,
                                       "bbb " * 10,
                                       "ccc " * 3,
                                       "ddd " * 1],
                                      ), 6, 33),

                          ])
def test___bow_bi_gram(string, length, total):
    bow = tkn.bow_bi_gram(string)
    assert(len(bow) == length)
    assert(sum_tally_tuples(bow) == total)



@pytest.mark.parametrize("string,length,total",
                         [(tlz.reduce(lambda x, y: x+y,
                                      ["aaa " * 20,
                                       "bbb " * 10,
                                       "ccc " * 3,
                                       "ddd " * 1],
                                      ), 8, 32),

                          ])
def test___bow_tri_gram(string, length, total):
    bow = tkn.bow_tri_gram(string)
    assert(len(bow) == length)
    assert(sum_tally_tuples(bow) == total)
