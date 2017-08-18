"""
QA model code.
"""
from collections import namedtuple
import concurrent.futures
import glob
import json
import os
import random
import sys

import featurize
import ops
import numpy as np
import tensorflow as tf
from framework import Model
from constants import EMBEDDING_DIM, PAD

ModelConfig = namedtuple("QAModel", [
    "vocab_size",
    "question_layers",
    "document_layers",
    "pick_end_word_layers",
    "layer_size",
    "beam_size",
    "embedding_dropout_prob",
    "hidden_dropout_prob",
    "learning_rate",
    "anneal_every",
    "anneal_rate",
    "clip_norm",
    "l2_scale",
    "weight_noise",
])

SquadExample = namedtuple("SquadExample", [
    "question",
    "context",
    "sentence_lengths",
    "answer_sentence",
    "answer_start",
    "answer_end",
    "same_as_question_word",
    "repeated_words",
    "repeated_word_intensity",
    "id"
])


def load_sample(sample_file):
    """Load a single example and return it as a SquadExample tuple."""
    with open(sample_file, "rt") as handle:
        sample = json.load(handle)

    return SquadExample(question=sample["question"], context=sample["context"],
                        sentence_lengths=sample["sent_lengths"],
                        answer_sentence=sample["ans_sentence"],
                        answer_start=sample["ans_start"],
                        answer_end=sample["ans_end"],
                        same_as_question_word=sample["same_as_question_word"],
                        repeated_words=sample["repeated_words"],
                        repeated_word_intensity=sample["repeated_intensity"],
                        id=sample["qa_id"])


def make_batches(samples, augmented_samples, batch_size, cycle=False):
    """Convert samples from the samples generator into
    padded batches with `batch_size` as the first dimension."""
    current_batch = []

    while True:
        if augmented_samples is not None:
            augmented = featurize.random_sample(augmented_samples, 10000)
            epoch_samples = samples + augmented
        else:
            epoch_samples = samples

        random.shuffle(epoch_samples)
        for idx, sample in enumerate(epoch_samples):
            current_batch.append(load_sample(sample))

            if len(current_batch) == batch_size or idx == len(samples) - 1:

                questions = ops.lists_to_array(
                    [sample.question for sample in current_batch], padding=-1)

                contexts = ops.lists_to_array(
                    [sample.context for sample in current_batch], padding=-1)

                # Auxilary features
                same_as_question_word = ops.lists_to_array(
                    [sample.same_as_question_word for sample in current_batch],
                    padding=0)

                repeated_words = ops.lists_to_array(
                    [sample.repeated_words for sample in current_batch],
                    padding=0)

                repeated_word_intensity = ops.lists_to_array(
                    [sample.repeated_word_intensity for sample in current_batch],
                    padding=0)

                sent_lengths = ops.lists_to_array(
                    [sample.sentence_lengths for sample in current_batch],
                    padding=0)

                answer_sentences = np.array(
                    [sample.answer_sentence for sample in current_batch])

                answer_starts = np.array(
                    [sample.answer_start for sample in current_batch])

                answer_ends = np.array(
                    [sample.answer_end for sample in current_batch])

                ids = [sample.id for sample in current_batch]

                yield [questions, contexts, same_as_question_word, repeated_words,
                       repeated_word_intensity, sent_lengths,
                       answer_sentences, answer_starts, answer_ends, ids]

                current_batch = []

        if not cycle:
            break


def make_eval_batches(path, batch_size):
    with open(path, "rt") as handle:
        eval_samples = json.load(handle)

    current_batch = []
    for idx, sample in enumerate(eval_samples):
        current_batch.append(sample)
        if len(current_batch) == batch_size or idx == len(eval_samples) - 1:
            ids, tokenized_context, features = zip(*current_batch)

            questions, contexts, saq, rw, rwi, lens = zip(*features)
            questions = ops.lists_to_array(questions, padding=-1)
            contexts = ops.lists_to_array(contexts, padding=-1)
            same_as_question = ops.lists_to_array(saq, padding=0)
            repeated_words = ops.lists_to_array(rw, padding=0)
            repeated_word_intensity = ops.lists_to_array(rwi, padding=0)
            sent_lens = ops.lists_to_array(lens, padding=0)

            yield [ids, tokenized_context, [questions, contexts, same_as_question,
                   repeated_words, repeated_word_intensity, sent_lens]]

            current_batch = []


def load_input_data(path, batch_size, validation_size, current_iteration):
    """
    Load the input data from the provided directory, splitting it into
    validation batches and an infinite generator of training batches.

    Arguments:
        - path: Directory with one file or subdirectory per training sample.
        - batch_size: Size of each training batch (per GPU).
        - validation_size: Size of the validation set (not per GPU).
        - current_iteration: The number of training batches to skip.
    Return a tuple of (train data, validation data), where training and validation
    data are both generators of batches. A batch is a list of NumPy arrays,
    in the same order as returned by load_sample.
    """
    if not os.path.exists(os.path.join(path, "train")):
        print("Non-existent directory as input path: {}".format(path),
              file=sys.stderr)
        sys.exit(1)

    # Get paths to all samples that we want to load.
    train_samples = glob.glob(os.path.join(path, "train", "*"))
    valid_samples = glob.glob(os.path.join(path, "dev", "*"))
    augmented_samples = glob.glob(os.path.join(path, "augmented", "*"))

    # Sort then shuffle to ensure that the random order is deterministic
    # across runs.
    train_samples.sort()
    valid_samples.sort()
    augmented_samples.sort()
    random.shuffle(train_samples)
    random.shuffle(valid_samples)
    random.shuffle(augmented_samples)

    train = ops.prefetch_generator(make_batches(train_samples,
                                                augmented_samples,
                                                batch_size, cycle=True),
                                   to_fetch=2 * batch_size)
    valid = ops.prefetch_generator(make_batches(valid_samples, None, validation_size),
                                   to_fetch=validation_size)

    evals = make_eval_batches(os.path.join(path, "eval.json"), 1)

    return train, valid, evals


def featurize_question(model, questions, embedding_dropout, training):
    """Embed the question as the final hidden state of a stack of Bi-LSTMs
    and a "passage-indenpendent" embedding from Rasor.

    Arguments:
        model:              QA model hyperparameters.
        questions:          Question word indices with shape `[batch, length]`.
        embedding_dropout:  Dropout probability for the inputs to the LSTM.

    Returns:
        hiddens: Vector representation for the entire question with shape
                 `[batch, features]`.
    """
    with tf.variable_scope("GloveEmbeddings", reuse=True):
        embeds = tf.get_variable("GloveEmbeddings")
        question_embeddings = ops.masked_embedding_lookup(embeds, questions)
        question_embeddings = tf.nn.dropout(question_embeddings,
                                            embedding_dropout)

    with tf.variable_scope("QuestionLSTMs"):
        hiddens, final_h, _ = ops.cudnn_lstm(question_embeddings,
                                             model.question_layers,
                                             model.layer_size,
                                             model.weight_noise,
                                             training)

        batch = tf.shape(final_h)[0]
        final_states = tf.reshape(
            final_h[:, -2:, :], [batch, 2 * model.layer_size])

    with tf.variable_scope("PassageIndependentEmbedding"):
        features = hiddens.get_shape()[-1].value
        hiddens = tf.contrib.layers.fully_connected(
            inputs=hiddens,
            num_outputs=features,
            activation_fn=None)

        sentinel = tf.get_variable(
            shape=[features, 1],
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            name="sentinel")

        # [batch, words, 1]
        alphas = ops.semibatch_matmul(hiddens, sentinel)
        alphas = tf.nn.softmax(alphas, dim=1)

        passage_indep_embedding = tf.reduce_sum(
            alphas * hiddens, axis=1)

    return tf.concat(axis=1, values=[final_states, passage_indep_embedding])


def question_aligned_embeddings(documents, questions, hidden_dimension):

    def mlp(x, reuse=None):
        with tf.variable_scope("MLP0", reuse=reuse):
            h = tf.contrib.layers.fully_connected(
                inputs=x,
                num_outputs=hidden_dimension,
                activation_fn=tf.nn.relu)
        with tf.variable_scope("MLP1", reuse=reuse):
            h = tf.contrib.layers.fully_connected(
                inputs=h,
                num_outputs=hidden_dimension,
                activation_fn=None)
        return h

    docs = mlp(documents)
    qs = mlp(questions, reuse=True)

    # [batch, w_doc, w_ques]
    scores = tf.matmul(docs, qs, transpose_b=True)
    alphas = tf.nn.softmax(scores, dim=-1)

    return tf.matmul(alphas, questions)


def featurize_document(model, questions, documents, same_as_question,
                       repeated_words, repeated_word_intensity,
                       question_vector, embedding_dropout, training):
    """Run a stack of Bi-LSTM's over the document.
    Arguments:
        model:              QA model hyperparameters.
        documents:          Document word indices with shape `[batch, length]`.
        same_as_question:   Boolean: Does the question contain this word? with
                            shape `[batch, length]`
        question_vector:    Vector representation of the question with
                            shape `[batch, features]`
        embedding_dropout:  Dropout probability for the LSTM inputs.
        hidden_dropout:     Dropout probability for the LSTM hidden states.

    Returns:
        hiddens: Vector representation for each word in the document with
                 shape `[batch, length, features]`.
    """
    with tf.variable_scope("GloveEmbeddings", reuse=True):
        embeds = tf.get_variable("GloveEmbeddings")
        document_embeddings = ops.masked_embedding_lookup(embeds, documents)
        question_embeddings = ops.masked_embedding_lookup(embeds, questions)

    qa_embeds = question_aligned_embeddings(
        document_embeddings, question_embeddings, model.layer_size)

    # Tile the question vector across the length of the document.
    question_vector = tf.tile(
        tf.expand_dims(question_vector, 1),
        [1, tf.shape(documents)[1], 1])

    same_as_question = tf.expand_dims(same_as_question, 2)
    repeated_words = tf.expand_dims(repeated_words, 2)
    repeated_word_intensity = tf.expand_dims(repeated_word_intensity, 2)
    document_embeddings = tf.concat(
        axis=2, values=[document_embeddings, same_as_question, repeated_words,
                        repeated_word_intensity, question_vector, qa_embeds])

    document_embeddings = tf.nn.dropout(document_embeddings,
                                        embedding_dropout)

    with tf.variable_scope("DocumentLSTMs"):
        hiddens, _, _ = ops.cudnn_lstm(document_embeddings,
                                       model.document_layers,
                                       model.layer_size,
                                       model.weight_noise,
                                       training)

    return hiddens


def score_sentences(model, documents_features,
                    sentence_lengths, hidden_dropout):
    """Compute logits for selecting each sentence in the document.

    Arguments:
        documents_features: Feature representation of the document
                            with shape `[batch, length, features]`.
        sentence_lengths:   Length of each sentence in the document
                            with shape `[batch, num_sentences]`, used
                            for finding sentence boundaries in
                            documents_features.

    Returns:
        logits:  Scores for each sentence with shape
                 `[batch, num_sentences]`.
    """
    batch_size = tf.shape(documents_features)[0]
    length = tf.shape(documents_features)[1]
    features = documents_features.get_shape()[-1].value
    num_sentences = tf.shape(sentence_lengths)[1]

    # Find sentence boundary indices.
    sentence_start_positions = tf.cumsum(
        sentence_lengths, axis=1, exclusive=True)
    sentence_end_positions = tf.cumsum(sentence_lengths, axis=1) - 1

    # Flatten indices and document to `[batch * length, features]`
    # in preparation for a gather.
    offsets = length * tf.expand_dims(tf.range(batch_size), 1)
    sentence_start_positions = tf.reshape(
        sentence_start_positions + offsets, [-1])
    sentence_end_positions = tf.reshape(
        sentence_end_positions + offsets, [-1])
    documents_features = tf.reshape(documents_features, [-1, features])

    # Gather the final hidden state of the forward LSTM at the end
    # of each sentence.
    forward_features = documents_features[:, :(features // 2)]
    forward_states = tf.gather(forward_features, sentence_end_positions)

    # Gather the first hidden state of the backward LSTM at the
    # start of sentence (after it was run over the entire sentence).
    backward_features = documents_features[:, (features // 2):]
    backward_states = tf.gather(backward_features, sentence_start_positions)

    # Score each sentence using an MLP.
    sentence_states = tf.concat(axis=1, values=[forward_states, backward_states])
    sentence_states = tf.reshape(sentence_states,
                                 [batch_size, num_sentences, features])

    with tf.variable_scope("SentenceSelection"):
        logits = tf.contrib.layers.fully_connected(
            inputs=tf.nn.dropout(sentence_states, hidden_dropout),
            num_outputs=1,
            activation_fn=None)

    return tf.squeeze(logits, [2])


def slice_sentences(document_features, picks, sentence_lengths):
    """Extract selected sentence spans from the document features.

    Arguments:
        document_features:  A `[batch, length, features]` representation
                            of the documents.
        picks:              Sentence to extract with shape
                            `[batch, selections]`.
        sentence_lengths:   Length of each sentence in the document with shape
                            `[batch, num_sentences]`.

    Returns extracted features for each selected sentence as a tensor with shape
        `[batch, selections, max_sentence_len, features]`
    """
    sentence_offsets = tf.cumsum(
        sentence_lengths, axis=1, exclusive=True)

    starts = ops.gather_from_rows(sentence_offsets, picks)
    lengths = ops.gather_from_rows(sentence_lengths, picks)
    sentence_embeddings = ops.slice_fragments(
        document_features, starts, lengths)
    return sentence_embeddings


def slice_end_of_sentence(sentence_features,
                          start_word_picks,
                          sentence_lengths):
    """Extract the final span of each sentence after the selected
    starting words.

    Arguments:
        sentence_features:  Sentence representation with shape
                            `[batch, k, words, features]`.
        start_word_picks:   Starting word selections with shape
                            `[batch, k]`.
        sentence_lengths:   Length of each sentence in
                            sentence_features of shape `[batch, k]`.
                            Used for masking.

    Returns extracted sentence spans with shape
    `[batch, k, max_fragment_len, features]`.
    """
    fragment_lengths = sentence_lengths - start_word_picks

    # Flatten to `[batch, beam * words, features]`.
    beams = tf.shape(sentence_features)[1]
    words = tf.shape(sentence_features)[2]
    sentence_features = tf.reshape(
        sentence_features,
        [tf.shape(sentence_features)[0],
         beams * words,
         sentence_features.get_shape()[-1].value])

    # Offset the start locations by words * beams to account
    # for flattening.
    start_word_picks += words * tf.expand_dims(tf.range(beams), 0)

    sentence_fragments = ops.slice_fragments(
        sentence_features, start_word_picks, fragment_lengths)
    return sentence_fragments


def score_start_word(model, document_embeddings,
                     sentence_picks, sentence_lengths, hidden_dropout):
    """Score each possible span spart word in a sentence by
    passing it through an MLP.

    Arguments:
        model:               QA model hyperparameters.
        document_embeddings: Document representation with shape
                                `[batch, length, features]`.
        sentence_picks:      Selected sentences with shape
                            `[batch, beam_size]`.
        sentence_lengths:    Lengths of each sentence in the document
                             with shape `[batch, num_sentences]`.

    Returns:
        logits: [batch, beam_size, length] scores for each start
                word in the beam, where `length` is the maximum
                `length` of a selected sentence.
    """
    # [batch, beams, max_sentence_length, features].
    sentence_embeddings = slice_sentences(
        document_embeddings, sentence_picks, sentence_lengths)

    with tf.variable_scope("StartWordSelection"):
        logits = tf.contrib.layers.fully_connected(
            inputs=tf.nn.dropout(sentence_embeddings, hidden_dropout),
            num_outputs=1,
            activation_fn=None)

    return tf.squeeze(logits, [-1])


def score_end_words(model, document_embeddings,
                    sentence_picks, start_word_picks,
                    sentence_lengths, hidden_dropout, training):
    """Score each possible span end word in the sentence by
    running a Bi-LSTM over the remaining sentence span and
    passing the result through an MLP.

    Arguments:
        model:                  QA model hyperparameters
        document_embeddings:    A `[batch, length, features]`
                                representation of a document.
        sentence_picks:         Index of selected sentences in
                                the document with shape `[batch, k]`.
        start_word_picks:       Index of start words in each selected
                                sentence with shape `[batch, k]`.
        sentence_lengths:       Length of each sentence with shape
                                `[batch, num_sentences]`.

    Returns scores for each possible span end word with shape
    `[batch, k, max-span-length]`.
    """
    # Slice the document twice to get end word spans.
    sentence_embeddings = slice_sentences(
        document_embeddings, sentence_picks, sentence_lengths)

    picked_sentence_lengths = ops.gather_from_rows(sentence_lengths,
                                                   sentence_picks)
    sentence_fragments = slice_end_of_sentence(
        sentence_embeddings, start_word_picks, picked_sentence_lengths)

    # Collapse batch and beam dimension
    batch = tf.shape(sentence_fragments)[0]
    beam = tf.shape(sentence_fragments)[1]
    frag_len = tf.shape(sentence_fragments)[2]
    features = sentence_fragments.get_shape()[-1].value

    sentence_fragments = tf.reshape(
        sentence_fragments, [batch * beam, frag_len, features])

    with tf.variable_scope("PickEndWordLSTMs"):
        hiddens, _, _ = ops.cudnn_lstm(sentence_fragments,
                                       model.pick_end_word_layers,
                                       model.layer_size,
                                       model.weight_noise,
                                       training)

    hiddens = tf.reshape(
        hiddens, [batch, beam, frag_len, hiddens.get_shape()[-1].value])

    with tf.variable_scope("EndWordSelection"):
        logits = tf.contrib.layers.fully_connected(
            inputs=tf.nn.dropout(hiddens, hidden_dropout),
            num_outputs=1,
            activation_fn=None)

    return tf.squeeze(logits, [-1])


def globally_normalized_loss(beam_states, labels):
    """Global normalized loss with early updating.

    Arguments:
        beam_states: List of previous decisions and current decision scores
                     for each step in the search process, as well as the final
                     beam. Previous decisions are a tensor of indices with shape
                     `[batch, beam_size]` corresponding to the selection at
                     previous time steps and scores has shape
                     `[batch, beam_size, classes]`.
        labels:      List of correct labels at each step in the search process,
                     each with shape `[batch]`.

    Returns scalar loss averaged across each example in the batch.
    """
    loss, loss_mask = 0., 1.
    for i, (prev_decisions, scores) in enumerate(reversed(beam_states)):
        batch = tf.shape(scores)[0]
        beam = tf.shape(scores)[1]

        correct = tf.cast(tf.ones((batch, beam)), tf.bool)
        for decision, label in zip(prev_decisions, labels):
            correct = tf.logical_and(
                correct, tf.equal(tf.expand_dims(label, 1), decision))

        # Correct is one-hot in each row if previous decisions
        # were correct.
        correct_mask = tf.cast(correct, tf.float32)
        any_correct = tf.cast(
            tf.reduce_sum(correct_mask, axis=1), tf.bool)

        if len(prev_decisions) == len(labels):
            # Final step of the search procedure. Find the correct,
            # if any, beam and normalize only over the final candidate
            # set.
            targets = tf.argmax(correct_mask, axis=1)
            stage_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets, logits=scores)

            # Avoid Nans
            stage_loss = tf.where(
                any_correct,
                stage_loss,
                tf.zeros_like(stage_loss))

        else:
            # Early updates: Assuming the previous decisions are
            # correct for some beam, find the correct answer among
            # the candidates at this stage and maximize its
            # log-probability.

            # Get the labels for this decision and tile for each beam
            # [batch * k].
            targets = labels[len(prev_decisions)]
            targets = tf.reshape(
                tf.tile(tf.expand_dims(targets, 1), [1, beam]), [-1])

            # Prevent NANs from showing up if the correct start or end word
            # isn't scored by the model. Any "wrong" targets it introduces will
            # be masked out since the proceeding choices will be incorrect.
            targets = tf.minimum(targets, tf.shape(scores)[2] - 1)

            scores = tf.reshape(scores, [batch * beam, -1])
            stage_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets, logits=scores)

            stage_loss = tf.reshape(stage_loss, [batch, beam])

            # Avoid Nans
            stage_loss = tf.where(
                tf.cast(correct_mask, tf.bool),
                stage_loss,
                tf.zeros_like(stage_loss))

            stage_loss = tf.reduce_sum(stage_loss, axis=1)

        # Mask out items of the batch where elements have already
        # received loss
        loss += loss_mask * stage_loss

        # Update the loss mask. Any correct is {0, 1}^{batch}
        loss_mask *= 1. - tf.cast(any_correct, tf.float32)

    return tf.reduce_mean(loss)


def build_model(model):
    """Build a Tensorflow graph for the QA model.
    Return a model.Model for training, evaluation, etc.
    """
    with tf.name_scope("Inputs"):
        questions = tf.placeholder(
            tf.int32, name="Questions", shape=[None, None])
        documents = tf.placeholder(
            tf.int32, name="Documents", shape=[None, None])
        same_as_question_feature = tf.placeholder(
            tf.float32, name="SameAsQuestionFeature", shape=[None, None])
        repeated_words = tf.placeholder(
            tf.float32, name="RepeatedWordFeature", shape=[None, None])
        repeated_word_intensity = tf.placeholder(
            tf.float32, name="RepeatedWordIntensity", shape=[None, None])
        sentence_lengths = tf.placeholder(
            tf.int32, name="SentenceOffsets", shape=[None, None])
        sentence_labels = tf.placeholder(
            tf.int32, name="SentenceLabels", shape=[None])
        word_start_labels = tf.placeholder(
            tf.int32, name="WordStartLabels", shape=[None])
        word_end_labels = tf.placeholder(
            tf.int32, name="WordEndLabels", shape=[None])
        embedding_dropout = tf.placeholder_with_default(
            model.embedding_dropout_prob, shape=[])
        hidden_dropout = tf.placeholder_with_default(
            model.hidden_dropout_prob, shape=[])
        training = tf.placeholder_with_default(
            True, shape=[], name="TrainingIndicator")
        exact_match = tf.placeholder(
            tf.float32, name="ExactMatch", shape=[])
        f1 = tf.placeholder(
            tf.float32, name="F1", shape=[])

    with tf.variable_scope("GloveEmbeddings"):
        embeddings = tf.get_variable(
            shape=[model.vocab_size, EMBEDDING_DIM],
            initializer=tf.zeros_initializer(),
            trainable=False, name="GloveEmbeddings")
        embedding_placeholder = tf.placeholder(
            tf.float32, [model.vocab_size, EMBEDDING_DIM])
        embedding_init = embeddings.assign(embedding_placeholder)

    with tf.name_scope("QuestionEmbeddings"):
        question_vector = featurize_question(model, questions,
                                             embedding_dropout, training)

    with tf.name_scope("DocumentEmbeddings"):
        document_embeddings = featurize_document(
            model, questions, documents, same_as_question_feature,
            repeated_words, repeated_word_intensity,
            question_vector, embedding_dropout, training)

    # Keep track of the beam state at each decision point
    beam_states = []
    with tf.name_scope("PickSentence"):
        sentence_scores = score_sentences(
            model, document_embeddings, sentence_lengths, hidden_dropout)

        beam_states.append(([], tf.expand_dims(sentence_scores, 1)))
        beam_scores, sentence_picks = tf.nn.top_k(
            sentence_scores,
            k=tf.minimum(model.beam_size, tf.shape(sentence_scores)[1]),
            sorted=True)

        sentence_correct = tf.reduce_mean(
            tf.cast(tf.equal(sentence_labels, sentence_picks[:, 0]), tf.float32))

    with tf.name_scope("PickStartWord"):
        start_word_scores = score_start_word(
            model, document_embeddings, sentence_picks, sentence_lengths, hidden_dropout)
        beam_scores = tf.expand_dims(beam_scores, 2) + start_word_scores

        beam_states.append(([sentence_picks], beam_scores))
        beam_scores, kept_sentences, start_words = ops.prune_beam(
            beam_scores, sentence_picks, model.beam_size)

        start_word_correct = tf.reduce_mean(
            tf.cast(tf.logical_and(
                tf.equal(word_start_labels, start_words[:, 0]),
                tf.equal(sentence_labels, kept_sentences[:, 0])), tf.float32))

    with tf.name_scope("PickEndWord"):
        end_word_scores = score_end_words(
            model, document_embeddings, kept_sentences,
            start_words, sentence_lengths, hidden_dropout, training)
        beam_scores = tf.expand_dims(beam_scores, 2) + end_word_scores

        beam_states.append(([kept_sentences, start_words], beam_scores))
        beam_scores, (kept_sentences, kept_start_words), end_words = ops.prune_beam(
            beam_scores, [kept_sentences, start_words], model.beam_size)

        # Also track the final decisions.
        beam_states.append(([kept_sentences, kept_start_words, end_words],
                           beam_scores))

    # Get offset from start word
    end_word_picks = kept_start_words + end_words
    final_states = [kept_sentences, kept_start_words, end_word_picks]

    end_word_correct = tf.reduce_mean(
        tf.cast(tf.logical_and(
            tf.logical_and(
                tf.equal(word_end_labels, end_word_picks[:, 0]),
                tf.equal(word_start_labels, kept_start_words[:, 0])),
            tf.equal(sentence_labels, kept_sentences[:, 0])), tf.float32))

    with tf.name_scope("Loss"):
        # End prediction is based on the start word offset.
        end_labels = word_end_labels - word_start_labels
        labels = (sentence_labels, word_start_labels, end_labels)
        loss = globally_normalized_loss(beam_states, labels)

        l2_penalty = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(model.l2_scale),
            tf.trainable_variables())

        loss += l2_penalty

    with tf.name_scope("TrainStep"):
        iteration, (step, loss, gradnorm) = ops.default_train_step(
            model, loss)

    with tf.name_scope("TrainSummary"):
        train_summary = ops.scalar_summaries({
            "Train-Loss": loss,
            "Gradient-Norm": gradnorm,
            "Sentence-Correct": sentence_correct,
            "Start-Word-Correct": start_word_correct,
            "End-Word-Correct": end_word_correct})

    with tf.name_scope("ValidSummary"):
        valid_summary = ops.scalar_summaries({
            "Validation-Loss": loss,
            "Sentence-Correct": sentence_correct,
            "Start-Word-Correct": start_word_correct,
            "End-Word-Correct": end_word_correct})

    with tf.name_scope("SquadSummary"):
        squad_summary = ops.scalar_summaries({
            "Exact-Match": exact_match, "F1": f1})

    return Model(
        inputs=[questions, documents, same_as_question_feature,
                repeated_words, repeated_word_intensity,
                sentence_lengths, sentence_labels, word_start_labels,
                word_end_labels],
        outputs=[kept_sentences, kept_start_words, end_word_picks, sentence_correct,
                 start_word_correct, end_word_correct],
        loss=loss, training=training,
        dropout=[embedding_dropout, hidden_dropout],
        gradnorm=gradnorm, step=step, iteration=iteration,
        train_summary=train_summary, valid_summary=valid_summary,
        embedding_init=embedding_init,
        embedding_placeholder=embedding_placeholder,
        squad_summary=squad_summary,
        squad_inputs=[exact_match, f1])
