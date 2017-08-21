"""
A PaddlePaddle implementation of a question answering model.
"""
from __future__ import print_function

import json
import random
import collections
from collections import namedtuple

import paddle.v2 as paddle
from paddle.v2.layer import parse_network

__all__ = ["build_model"]

EMBEDDING_DIM = 300


def embedding_input(name, vocab_size, drop_rate=0.):
    """
    Create an embedding input to the network.

    Embeddings are static Glove vectors.
    """
    data = paddle.layer.data(
        name=name, type=paddle.data_type.integer_value_sequence(vocab_size))

    # CAUTIOUS: static parameters must be intialized by pre-trained parameter.
    # BUT, currently, if static parameters is not intialized,
    # Paddle will not warn you.
    embeddings = paddle.layer.embedding(
        input=data,
        size=EMBEDDING_DIM,
        param_attr=paddle.attr.Param(name="GloveVectors", is_static=True),
        layer_attr=paddle.attr.ExtraLayerAttribute(drop_rate=drop_rate), )
    return embeddings


def binary_output(name):
    """
    Create a binary output for the network.
    """
    data = paddle.layer.data(
        name=name, type=paddle.data_type.integer_value_sequence(2))
    return data


def binary_input(name):
    """
    Create a binary input for the network.
    """
    data = paddle.layer.data(
        name=name, type=paddle.data_type.dense_vector_sequence(1))
    return data


def bidirectional_lstm(inputs, size, depth, drop_rate=0., prefix=""):
    """
    Run a bidirectional LSTM on the inputs.
    """
    if not isinstance(inputs, collections.Sequence):
        inputs = [inputs]

    lstm_last = []
    for dirt in ["fwd", "bwd"]:
        for i in range(depth):
            input_proj = paddle.layer.mixed(
                name="%s_in_proj_%0d_%s__" % (prefix, i, dirt),
                size=size * 4,
                bias_attr=paddle.attr.Param(initial_std=0.),
                input=[paddle.layer.full_matrix_projection(lstm)] if i else [
                    paddle.layer.full_matrix_projection(in_layer)
                    for in_layer in inputs
                ])
            lstm = paddle.layer.lstmemory(
                input=input_proj,
                bias_attr=paddle.attr.Param(initial_std=0.),
                param_attr=paddle.attr.Param(initial_std=5e-4),
                reverse=(dirt == "bwd"))
        lstm_last.append(lstm)

    final_states = paddle.layer.concat(input=[
        paddle.layer.last_seq(input=lstm_last[0]),
        paddle.layer.first_seq(input=lstm_last[1]),
    ])
    return final_states, paddle.layer.concat(
        input=lstm_last,
        layer_attr=paddle.attr.ExtraLayerAttribute(drop_rate=drop_rate), )


def build_document_embeddings(config, documents, same_as_question,
                              question_vector):
    """
    Build the document word embeddings.
    """
    hidden = paddle.layer.concat(input=[
        documents,
        same_as_question,
    ])

    # Half the question embedding is the final states of the LSTMs.
    question_expanded = paddle.layer.expand(
        input=question_vector, expand_as=documents)
    _, hidden = bidirectional_lstm([hidden, question_expanded],
                                   config.layer_size, config.document_layers,
                                   config.hidden_dropout, "__document__")

    return hidden


def build_question_vector(config, questions):
    """
    Build the question vector.
    """

    final, lstm_hidden = bidirectional_lstm(
        questions, config.layer_size, config.question_layers,
        config.hidden_dropout, "__question__")

    # The other half is created by doing an affine transform to generate
    # candidate embeddings, doing a second affine transform followed by a
    # sequence softmax to generate weights for the embeddings, and summing over
    # the weighted embeddings to generate the second half of the question
    # embedding.
    candidates = paddle.layer.fc(
        input=lstm_hidden, size=config.layer_size, act=None)
    weights = paddle.layer.fc(
        input=questions, size=1, act=paddle.activation.SequenceSoftmax())
    weighted = paddle.layer.scaling(input=candidates, weight=weights)
    embedding = paddle.layer.pooling(
        input=weighted, pooling_type=paddle.pooling.Sum())

    return paddle.layer.concat(input=[final, embedding])


def pick_word(config, word_embeddings):
    """
    For each word, predict a one or a zero indicating whether it is the chosen
    word.

    This is done with a two-class classification.
    """
    predictions = paddle.layer.fc(
        input=word_embeddings, size=2, act=paddle.activation.Softmax())
    return predictions


def build_classification_loss(predictions, classes):
    """
    Build a classification loss given predictions and desired outputs.
    """
    # classification_cost is just multi-class cross entropy,
    # but it also add a classification error evaluator.
    return paddle.layer.classification_cost(input=predictions, label=classes)


def build_model(config, is_infer=False):
    """
    Build the PaddlePaddle model for a configuration.
    """
    questions = embedding_input("Questions", config.vocab_size,
                                config.embedding_dropout)
    documents = embedding_input("Documents", config.vocab_size,
                                config.embedding_dropout)

    same_as_question = binary_input("SameAsQuestion")

    correct_sentence = binary_output("CorrectSentence")
    correct_start_word = binary_output("CorrectStartWord")
    correct_end_word = binary_output("CorrectEndWord")

    # here the question vector is not a sequence
    question_vector = build_question_vector(config, questions)

    document_embeddings = build_document_embeddings(
        config, documents, same_as_question, question_vector)
    sentence_pred = pick_word(config, document_embeddings)
    start_word_pred = pick_word(config, document_embeddings)
    end_word_pred = pick_word(config, document_embeddings)

    if is_infer:
        return [sentence_pred, start_word_pred, end_word_pred]
    else:
        return [
            build_classification_loss(sentence_pred, correct_sentence),
            build_classification_loss(start_word_pred, correct_start_word),
            build_classification_loss(end_word_pred, correct_end_word)
        ]


if __name__ == "__main__":
    from paddle_train import load_config
    conf = load_config("paddle-config.json")
    losses = build_model(conf)
    print(parse_network(losses))
