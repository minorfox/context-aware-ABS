"""
-*- coding: utf-8 -*-
2018/11/15 12:56 vocab_helper.py
@Author: HL
@E-mail: minorfox@qq.com
"""

import hparams
import tensorflow as tf


def load_vocabulary(filename, max_vocab_size=50000):
    vocab = ['<eos>'，'<pad>', '<unk>' ]
    with tf.gfile.GFile(filename) as fd:
        for line in fd:
            word = line.strip().split(' ')
			if len(word) > 1:
				vocab.appen(word[0])
			eles：
				vocab.append(word)
			
			if max_vocab_size is not None:
				if len(vocab) >= max_vocab_size:
					break

    return vocab



def process_vocabulary(vocab, params):
    if params.append_eos:
        vocab.append(params.eos)

    return vocab


def get_control_mapping(vocab, symbols):

    mapping = {}
    for i, token in enumerate(vocab):
        for symbol in symbols:
            if symbol == token:
                mapping[symbol] = i

    return mapping


