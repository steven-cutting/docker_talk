__title__ = 'smpl_tokenizer'
__author__ = 'Steven Cutting'
__author_email__ = 'steven.e.cutting@linux.com'
__created_on__ = '02/06/2016'
__copyright__ = "smpl_tokenizer Copyright (C) 2015  Steven Cutting"


import toolz as tlz
sliding_window_c = tlz.curry(tlz.sliding_window)
map_c = tlz.curry(tlz.map)

import utils


@tlz.curry
def ngram_tuples(n, string, minlen=3, maxlen=25):
    return tlz.pipe(string,
                    utils.lower,
                    utils.splitter_of_words,
                    utils.filter_whitespace,
                    utils.filter_shorter_than(minlen),
                    utils.filter_longer_than(maxlen),
                    sliding_window_c(n))


@tlz.curry
def ngram(n, string, minlen=3, maxlen=25):
    return tlz.pipe(string,
                    ngram_tuples(n, minlen=minlen, maxlen=maxlen),
                    map_c(utils.join_strings("_")))


uni_gram = ngram(1)
bi_gram = ngram(2)
tri_gram = ngram(3)


@tlz.curry
def bag_of_words(parser, string):
    return tlz.pipe(string,
                    parser,
                    utils.freq)


bow_uni_gram = bag_of_words(uni_gram)
bow_bi_gram = bag_of_words(bi_gram)
bow_tri_gram = bag_of_words(tri_gram)
