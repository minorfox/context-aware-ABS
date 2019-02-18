"""
-*- coding: utf-8 -*-
2018/12/11 20:48 load_infer.py
@Author: HL
@E-mail: minorfox@qq.com
"""

import numpy as np
import tensorflow as tf
import inference
import data_utils
import hparams
import metric
import metric_v2
import rouge155
import utils
import vocab_helper
import local_gate_model as my_model

np.set_printoptions(threshold=np.nan)

def default_parameters():
    params = tf.contrib.training.HParams(
        test=hparams.test_src,
        references=[hparams.test_sum],
        output="./47.9",
        src_vocab=hparams.src_vocab,
        tgt_vocab=hparams.tgt_vocab,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        mapping=None,
        append_eos=False,
        device_list=[0],
        num_threads=1,
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_length=30,
        decode_batch_size=512,
    )

    return params


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().items():
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in params2.values().items():
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


# def import_params(model_dir, model_name, params):
#     if model_name.startswith("experimental_"):
#         model_name = model_name[13:]
#
#     model_dir = os.path.abspath(model_dir)
#     m_name = os.path.join(model_dir, model_name + ".json")
#
#     if not tf.gfile.Exists(m_name):
#         return params
#
#     with tf.gfile.Open(m_name) as fd:
#         tf.logging.info("Restoring model parameters from %s" % m_name)
#         json_str = fd.readline()
#         params.parse_json(json_str)
#
#     return params


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def set_variables(var_list, value_dict, prefix, feed_dict):
    ops = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var.name[:-2] == var_name:
                tf.logging.debug("restoring %s -> %s" % (name, var.name))
                placeholder = tf.placeholder(tf.float32,
                                             name="placeholder/" + var_name)
                with tf.device("/cpu:0"):
                    op = tf.assign(var, placeholder)
                    ops.append(op)
                feed_dict[placeholder] = value_dict[name]
                break

    return ops


def check_eos(inputs, eos_id):
    outs = []
    for i, out in enumerate(inputs):
        if eos_id in out:
            out = list(out)
            out = out[:out.index(eos_id)]
            outs.append(out)
    return outs


def decode_target_ids(inputs, params):
    decoded = []
    vocab = params.vocabulary["tgt_vocab"]

    for item in inputs:
        syms = []
        for idx in item:
            sym = vocab[idx]
            if sym == params.eos:
                break
            if sym == params.pad:
                break

            syms.append(sym)
        decoded.append(syms)

    return decoded



def shard_features(features, placeholders, predictions):
    num_shards = len(placeholders)
    feed_dict = {}
    n = 0

    for name in ["source", "source_length"]:
        feat = features[name]
        batch = feat.shape[0]
        shard_size = (batch + num_shards - 1) // num_shards

        for i in range(num_shards):
            shard_feat = feat[i * shard_size:(i + 1) * shard_size]

            if shard_feat.shape[0] != 0:
                feed_dict[placeholders[i][name]] = shard_feat
                n = i + 1
            else:
                break

    if isinstance(predictions, (list, tuple)):
        predictions = [item[:n] for item in predictions]

    return predictions, feed_dict


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    checkpoints_list = ["./multi_ckpt/model.ckpt-1130000", "./multi_ckpt/model.ckpt-1355000"]
    model_cls = my_model.Transformer

    params = default_parameters()
    params = merge_parameters(params, model_cls.get_parameters())
    params.vocabulary = {"src_vocab": vocab_helper.load_vocabulary(params.src_vocab),
                         "tgt_vocab": vocab_helper.load_vocabulary(params.tgt_vocab)}
    control_symbols = [params.pad, params.bos, params.eos, params.unk]
    params.mapping = {"src_vocab": vocab_helper.get_control_mapping(params.vocabulary["src_vocab"], control_symbols),
                      "tgt_vocab": vocab_helper.get_control_mapping(params.vocabulary["tgt_vocab"], control_symbols)}

    model_fn = model_cls(params)

    with tf.Graph().as_default():
        files = [params.test] + list(params.references)
        test_inputs = data_utils.sort_and_zip_files(files)
        features = data_utils.get_evaluation_input(test_inputs, params)

        placeholders = {
            "source": tf.placeholder(tf.int32, [None, None], "source"),
            "source_length": tf.placeholder(tf.int32, [None], "source_length")
        }

        predictions = inference.create_inference_graph([model_fn], placeholders, params)
        predictions = predictions[0][:, 0, :]

        all_refs = []
        all_outputs = []

        sess_creator = tf.train.ChiefSessionCreator(
            checkpoint_dir=params.output,
            config=session_config(params)
        )

        with tf.train.MonitoredSession(session_creator=sess_creator) as sess:
            while not sess.should_stop():
                feats = sess.run(features)
                outputs = sess.run(predictions,
                                   feed_dict={placeholders["source"]: feats["source"],
                                              placeholders["source_length"]: feats["source_length"]})
                # shape: [batch, len]
                outputs = outputs.tolist()
                references = feats["references"]

                all_outputs.extend(outputs)
                all_refs.extend(references)

        eos_id = params.mapping["tgt_vocab"][params.eos]
        model_outs = check_eos(all_outputs, eos_id)
        model_refs = check_eos(all_refs, eos_id)
        assert len(model_outs) == len(model_refs)
        rg1, rg2, rgl = rouge155.compute_rouge(model_outs, model_refs)
        tf.logging.info("test file has %d sentences" % np.shape(all_outputs))
        tf.logging.info("test file has %f RG-1 score" % rg1)
        tf.logging.info("test file has %f RG-2 score" % rg2)
        tf.logging.info("test file has %f RG-L score" % rgl)
        #
        out_seqs = decode_target_ids(all_outputs, params)
        ref_seqs = decode_target_ids(all_refs, params)
        utils.check_res(rgl, out_seqs, ref_seqs, params)


if __name__ == "__main__":
    main()
