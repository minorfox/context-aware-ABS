"""
-*- coding: utf-8 -*-
2018/11/3 15:01 data_utils.py
@Author: HL
@E-mail: minorfox@qq.com
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
import tensorflow as tf
import math




def get_training_input(filenames, params):

    src_dataset = tf.data.TextLineDataset(filenames[0])
    sum_dataset = tf.data.TextLineDataset(filenames[1])
    dataset = tf.data.Dataset.zip((src_dataset, sum_dataset))
    # repeat默认参数count表示生成几个epoch数据，默认一直生成
    dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.repeat()

    # tf.string_split默认按照‘ ’进行切分，切分出来有三个值
    # indices：整个切分后的shape，以及每个词的位置
    # value: 切分后的值
    # shape: 切分后的整个
    dataset = dataset.map(
        lambda src, tgt: (tf.string_split([src]).values,
                          tf.string_split([tgt]).values),
        num_parallel_calls=params.num_threads
    )

    # dataset = dataset.map(
    #     lambda src, summ: (tf.cond(tf.greater(tf.shape(src)[0], params.max_src-1), lambda: src[:params.max_src-1], lambda: src),
    #                        tf.cond(tf.greater(tf.shape(summ)[0], params.max_sum-1), lambda: summ[:params.max_sum-1], lambda: summ)),
    #     num_parallel_calls=params.num_threads
    # )

    dataset = dataset.map(
        lambda src, tgt: (tf.concat([src, [tf.constant(params.eos)]], axis=0),
                           tf.concat([tgt, [tf.constant(params.eos)]], axis=0)),
        num_parallel_calls=params.num_threads
    )

    dataset = dataset.map(
        lambda src, tgt: {
            "source": src,
            "target": tgt,
            "source_length": tf.shape(src)[0],
            "target_length": tf.shape(tgt)[0],
        },
        num_parallel_calls=params.num_threads
    )


    dataset = dataset.padded_batch(
        batch_size=params.batch_size,
        padded_shapes={
            # "source": [params.max_src],
            "source": [tf.Dimension(None)],
            "source_length": [],
            "target": [tf.Dimension(None)],
            "target_length": []
        },
        padding_values={
            "source": params.pad,
            "source_length": 0,
            "target": params.pad,
            "target_length": 0
        }
    )

    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    # 对应词表和id，num_oov_buckets默认为0，表示允许的oov词数，defaul_value默认为-1，表示OOV里的词都给-1的id
    # table只是建立起了一个词表的对应规则，具体开始转换，还要再调用table.lookup(senten)
    # table的初始化要用tf.table_initializer()
    src_table = tf.contrib.lookup.index_table_from_tensor(tf.constant(params.vocabulary["src_vocab"]),
                                                     default_value=tf.cast(params.mapping["src_vocab"][params.unk], tf.int64))
    tgt_table = tf.contrib.lookup.index_table_from_tensor(tf.constant(params.vocabulary["tgt_vocab"]),
                                                     default_value=tf.cast(params.mapping["tgt_vocab"][params.unk],
                                                                           tf.int64))
    features["source"] = src_table.lookup(features["source"])
    features["target"] = tgt_table.lookup(features["target"])

    return features




def sort_input_file(filename, reverse=True):
    # Read file
    with tf.gfile.Open(filename) as fd:
        inputs = [line.strip() for line in fd]

    input_lens = [
        (i, len(line.strip().split())) for i, line in enumerate(inputs)
    ]

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i

    return sorted_keys, sorted_inputs


def sort_and_zip_files(names):
    inputs = []
    input_lens = []
    files = [tf.gfile.GFile(name) for name in names]

    count = 0

    for lines in zip(*files):
        lines = [line.strip() for line in lines]
        input_lens.append((count, len(lines[0].split())))
        inputs.append(lines)
        count += 1

    # Close files
    for fd in files:
        fd.close()

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=True)
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])

    return [list(x) for x in zip(*sorted_inputs)]


def get_evaluation_input(inputs, params):
    with tf.device("/cpu:0"):
        # Create datasets
        datasets = []

        for data in inputs:
            dataset = tf.data.Dataset.from_tensor_slices(data)
            dataset = dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=params.num_threads)

            # Append <eos>
            dataset = dataset.map(
                lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
                num_parallel_calls=params.num_threads)

            datasets.append(dataset)

        dataset = tf.data.Dataset.zip(tuple(datasets))

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda *x: {
                "source": x[0],
                "source_length": tf.shape(x[0])[0],
                # 单个refence
                "references": x[1]
            },
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.padded_batch(
            params.decode_batch_size,
            {
                # "source": [params.max_src],
                "source": [tf.Dimension(None)],
                "source_length": [],
                "references": tf.Dimension(None) * (len(inputs) - 1)
            },
            {
                "source": params.pad,
                "source_length": 0,
                "references": params.pad * (len(inputs) - 1)
            }
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Covert source symbols to ids
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["src_vocab"]),
                        default_value=params.mapping["src_vocab"][params.unk]
        )

        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["tgt_vocab"]),
                        default_value=params.mapping["tgt_vocab"][params.unk]
        )

        features["source"] = src_table.lookup(features["source"])
        features["references"] = tgt_table.lookup(features["references"])

    return features



def get_inference_input(inputs, params):
    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(inputs))

        # Split string
        dataset = dataset.map(lambda x: tf.string_split([x]).values,
                              num_parallel_calls=params.num_threads)

        # Append <eos>
        dataset = dataset.map(
            lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
            num_parallel_calls=params.num_threads
        )

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda x: {"source": x, "source_length": tf.shape(x)[0]},
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.padded_batch(
            params.decode_batch_size * len(params.device_list),
            {"source": [tf.Dimension(None)], "source_length": []},
            {"source": params.pad, "source_length": 0}
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["src_vocab"]),
                        default_value=params.mapping["src_vocab"][params.unk]
        )
        features["source"] = src_table.lookup(features["source"])

        return features
#
#
# def get_relevance_input(inputs, outputs, params):
#     # inputs
#     dataset = tf.data.Dataset.from_tensor_slices(
#         tf.constant(inputs)
#     )
#
#     # Split string
#     dataset = dataset.map(lambda x: tf.string_split([x]).values,
#                           num_parallel_calls=params.num_threads)
#
#     # Append <eos>
#     dataset = dataset.map(
#         lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
#         num_parallel_calls=params.num_threads
#     )
#
#     # Convert tuple to dictionary
#     dataset = dataset.map(
#         lambda x: {"source": x, "source_length": tf.shape(x)[0]},
#         num_parallel_calls=params.num_threads
#     )
#
#     dataset = dataset.padded_batch(
#         params.decode_batch_size,
#         {"source": [tf.Dimension(None)], "source_length": []},
#         {"source": params.pad, "source_length": 0}
#     )
#
#     iterator = dataset.make_one_shot_iterator()
#     features = iterator.get_next()
#
#     src_table = tf.contrib.lookup.index_table_from_tensor(
#         tf.constant(params.vocabulary["vocab"]),
#                     default_value=params.mapping["vocab"][params.unk]
#     )
#     features["source"] = src_table.lookup(features["source"])
#
#     # outputs
#     dataset_o = tf.data.Dataset.from_tensor_slices(
#         tf.constant(outputs)
#     )
#
#     # Split string
#     dataset_o = dataset_o.map(lambda x: tf.string_split([x]).values,
#                           num_parallel_calls=params.num_threads)
#
#     # Append <eos>
#     dataset_o = dataset_o.map(
#         lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
#         num_parallel_calls=params.num_threads
#     )
#
#     # Convert tuple to dictionary
#     dataset_o = dataset_o.map(
#         lambda x: {"target": x, "target_length": tf.shape(x)[0]},
#         num_parallel_calls=params.num_threads
#     )
#
#     dataset_o = dataset_o.padded_batch(
#         params.decode_batch_size,
#         {"target": [tf.Dimension(None)], "target_length": []},
#         {"target": params.pad, "target_length": 0}
#     )
#
#     iterator = dataset_o.make_one_shot_iterator()
#     features_o = iterator.get_next()
#
#     src_table = tf.contrib.lookup.index_table_from_tensor(
#         tf.constant(params.vocabulary["vocab"]),
#                     default_value=params.mapping["vocab"][params.unk]
#     )
#     features["target"] = src_table.lookup(features_o["target"])
#     features["target_length"] = features_o["target_length"]
#
#     return features


