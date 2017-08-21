#!/usr/bin/env python
#coding=utf-8
import os
import sys
import gzip
import logging

import reader
import paddle.v2 as paddle
from paddle.v2.layer import parse_network
from paddle_model import build_model
from paddle_train import load_config, choose_samples

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def load_reverse_dict(dict_file):
    word_dict = {}
    with open(dict_file, "r") as fin:
        for idx, line in enumerate(fin):
            word_dict[idx] = line.strip()
    return word_dict


def infer_a_batch(inferer, test_batch, ids_2_word):
    '''
    in inferring process, three layers are returned, they are:
    1. sentence predition
    2. prediction for start word
    3. predictions for end word
    '''

    sen_pred, start_pred, end_pred = inferer.infer(
        input=test_batch, field=["value"] * 3)

    # iterate over each testing sample
    cur_idx = 0
    for test_sample in test_batch:
        # iterate over each word of in document
        for i, document_word in enumerate(test_sample[1]):
            sen = "%d[%.4f %.4f]" % (test_sample[3][i], sen_pred[cur_idx][0],
                                     sen_pred[cur_idx][1])
            start = "%d[%.4f %.4f]" % (test_sample[4][i],
                                       start_pred[cur_idx][0],
                                       start_pred[cur_idx][1])
            end = "%d[%.4f %.4f]" % (test_sample[5][i], end_pred[cur_idx][0],
                                     end_pred[cur_idx][1])
            print("%s\t%s\t%s\t%s" %
                  (ids_2_word[document_word], sen, start, end))
            cur_idx += 1
        print("\n")


def infer(model_path, config):
    assert os.path.exists(model_path), "The model does not exist."
    paddle.init(use_gpu=True, trainer_count=1)

    ids_2_word = load_reverse_dict("featurized/vocab.txt")

    conf = load_config(config)
    predictions = build_model(conf, is_infer=True)

    # print(parse_network(predictions))  # for debug print

    # load the trained models
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open(model_path, "r"))
    inferer = paddle.inference.Inference(
        output_layer=predictions, parameters=parameters)

    _, valid_samples = choose_samples(conf.data_dir)
    test_reader = reader.train_reader(valid_samples, is_train=False)

    test_batch = []
    for i, item in enumerate(test_reader()):
        test_batch.append(item)
        if len(test_batch) == conf.batch_size:
            infer_a_batch(inferer, test_batch, ids_2_word)
            test_batch = []

    if len(test_batch):
        infer_a_batch(inferer, test_batch, ids_2_word)
        test_batch = []


if __name__ == "__main__":
    infer(
        model_path="checkpoint_param.latest.tar.gz",
        config="paddle-config.json")
