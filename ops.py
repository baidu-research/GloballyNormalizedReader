import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
from itertools import zip_longest
import queue
import threading

import numpy as np


def prefetch_generator(generator, to_fetch=10):
    q = queue.Queue(maxsize=to_fetch)

    def thread_worker(queue, gen):
        for val in gen:
            queue.put(val)

        # Put a poison pill.
        queue.put(None)

    t = threading.Thread(target=thread_worker, args=(q, generator))
    try:
        t.start()
        while True:
            val = q.get()
            if val is None:
                break
            else:
                yield val
    finally:
        t.join()


def gather_from_rows(inputs, indices):
    """Gather per-row elements according to indices.
    Arguments:
        inputs: [batch, M]
        indices: [batch, N]

    Returns a `[batch, N]` tensor T where T[i, j]
    is inputs[i, indices[i, j]].
    """
    batch = tf.shape(inputs)[0]
    rows = tf.shape(inputs)[1]

    collapsed_inputs = tf.reshape(inputs, [-1])
    collapsed_indices = indices + rows * tf.expand_dims(tf.range(batch), 1)
    gathered = tf.gather(collapsed_inputs, tf.reshape(collapsed_indices, [-1]))
    return tf.reshape(gathered, [batch, -1])


def prune_beam(scores, beam_state, beam_size):
    """
    Arguments:
        scores: [batch, beam_size, n-classes]
        beam_state: list([batch, beam_size])

    Return:
        scores: [batch, beam_size]
    """
    batch = tf.shape(scores)[0]
    num_beams = tf.shape(scores)[1]
    num_classes = tf.shape(scores)[2]

    collapsed_scores = tf.reshape(scores, [batch, num_beams * num_classes])

    topk_scores, topk_indices = tf.nn.top_k(
        collapsed_scores,
        k=tf.minimum(beam_size, num_beams * num_classes),
        sorted=True)

    beam_idx = topk_indices // num_classes
    choice_idx = topk_indices % num_classes

    if isinstance(beam_state, list):
        prev_decisions = []
        for i, state in enumerate(beam_state):
            prev_decisions.append(gather_from_rows(state, beam_idx))
    else:
        prev_decisions = gather_from_rows(beam_state, beam_idx)

    return topk_scores, prev_decisions, choice_idx


def slice_fragments(inputs, starts, lengths):
    """ Extract the documents_features corresponding to choosen sentences.
    Since sentences are different lengths, this will be jagged. Therefore,
    we extract the maximum length sentence and then pad appropriately.

    Arguments:
        inputs: [batch, time, features]
        starts: [batch, beam_size] starting locations
        lengths: [batch, beam_size] how much to trim.

    Returns:
        fragments: [batch, beam_size, max_length, features]
    """
    batch = tf.shape(inputs)[0]
    time = tf.shape(inputs)[1]
    beam_size = tf.shape(starts)[1]
    features = inputs.get_shape()[-1].value

    # Collapse the batch and time dimensions
    inputs = tf.reshape(
        inputs, [batch * time, features])

    # Compute the starting location of each sentence and adjust
    # the start locations to account for collapsed time dimension.
    starts += tf.expand_dims(time * tf.range(batch), 1)
    starts = tf.reshape(starts, [-1])

    # Gather idxs are consecutive rows beginning at start
    # and ending at start + length, for each start in starts.
    # If starts is [0; 6] and length is [0, 1, 2], then the
    # result is [0, 1, 2; 6, 7, 8], which is flattened to
    # [0; 1; 2; 6; 7; 8].
    # Ensure length is at least 1.
    max_length = tf.maximum(tf.reduce_max(lengths), 1)
    gather_idxs = tf.reshape(tf.expand_dims(starts, 1) +
                             tf.expand_dims(tf.range(max_length), 0), [-1])

    # Don't gather out of bounds
    gather_idxs = tf.minimum(gather_idxs, tf.shape(inputs)[0] - 1)

    # Pull out the relevant rows and partially reshape back.
    fragments = tf.gather(inputs, gather_idxs)
    fragments = tf.reshape(fragments, [batch * beam_size, max_length, features])

    # Mask out invalid entries
    length_mask = tf.sequence_mask(tf.reshape(lengths, [-1]), max_length)
    length_mask = tf.expand_dims(tf.cast(length_mask, tf.float32), 2)

    fragments *= length_mask

    return tf.reshape(fragments, [batch, beam_size, max_length, features])


def masked_embedding_lookup(embeddings, indices):
    """
    Construct an Embedding layer that gathers
    elements from a matrix with `size` rows,
    and `dim` features using the indices stored in `x`.
    """
    embedded = tf.nn.embedding_lookup(embeddings, tf.maximum(indices, 0))
    null_mask = tf.expand_dims(
        tf.cast(tf.greater_equal(indices, 0), tf.float32), -1)
    return embedded * null_mask


def weight_noise(weight, stddev, is_training):
    weight_shape = weight.get_shape().as_list()
    return tf.cond(is_training,
                   lambda: weight + tf.random_normal(shape=weight_shape,
                                                     stddev=stddev,
                                                     mean=0.0,
                                                     dtype=tf.float32),
                   lambda: weight)


def cudnn_lstm(inputs, num_layers, hidden_size, weight_noise_std, is_training):
    """Run the CuDNN LSTM.
    Arguments:
        - inputs:   A tensor of shape [batch, length, input_size] of inputs.
        - layers:   Number of RNN layers.
        - hidden_size:     Number of units in each layer.
        - is_training:     tf.bool indicating whether training mode is enabled.
    Return a tuple of (outputs, init_state, final_state).
    """
    input_size = inputs.get_shape()[-1].value
    if input_size is None:
        raise ValueError("Number of input dimensions to CuDNN RNNs must be "
                         "known, but was None.")

    # CUDNN expects the inputs to be time major
    inputs = tf.transpose(inputs, [1, 0, 2])

    cudnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers, hidden_size, input_size,
        input_mode="linear_input", direction="bidirectional")

    est_size = estimate_cudnn_parameter_size(
        num_layers=num_layers,
        hidden_size=hidden_size,
        input_size=input_size,
        input_mode="linear_input",
        direction="bidirectional")

    cudnn_params = tf.get_variable(
        "RNNParams",
        shape=[est_size],
        initializer=tf.contrib.layers.variance_scaling_initializer())

    if weight_noise_std is not None:
        cudnn_params = weight_noise(
            cudnn_params,
            stddev=weight_noise_std,
            is_training=is_training)

    init_state = tf.tile(
        tf.zeros([2 * num_layers, 1, hidden_size], dtype=tf.float32),
        [1, tf.shape(inputs)[1], 1])

    hiddens, output_h, output_c = cudnn_cell(
        inputs,
        input_h=init_state,
        input_c=init_state,
        params=cudnn_params,
        is_training=True)

    # Convert to batch major
    hiddens = tf.transpose(hiddens, [1, 0, 2])
    output_h = tf.transpose(output_h, [1, 0, 2])
    output_c = tf.transpose(output_c, [1, 0, 2])

    return hiddens, output_h, output_c


def cudnn_lstm_parameter_size(input_size, hidden_size):
    """Number of parameters in a single CuDNN LSTM cell."""
    biases = 8 * hidden_size
    weights = 4 * (hidden_size * input_size) + 4 * (hidden_size * hidden_size)
    return biases + weights


def direction_to_num_directions(direction):
    if direction == "unidirectional":
        return 1
    elif direction == "bidirectional":
        return 2
    else:
        raise ValueError("Unknown direction: %r." % (direction,))


def estimate_cudnn_parameter_size(num_layers,
                                  input_size,
                                  hidden_size,
                                  input_mode,
                                  direction):
    """
    Compute the number of parameters needed to
    construct a stack of LSTMs. Assumes the hidden states
    of bidirectional LSTMs are concatenated before being
    sent to the next layer up.
    """
    num_directions = direction_to_num_directions(direction)
    params = 0
    isize = input_size
    for layer in range(num_layers):
        for direction in range(num_directions):
            params += cudnn_lstm_parameter_size(
                isize, hidden_size
            )
        isize = hidden_size * num_directions
    return params


def lists_to_array(seq, padding):
    """Turn a list of lists into a padded numpy array.

       Given a ragged list of lists, turn it into a numpy array
       where each dimension is the same length, padded as necessary.
       Example:
            padding = -1
            [[[0, 0], [1], [2, 2, 2]], [[3], [3], [3]]]

            becomes

            [[[0, 0, -1], [1, -1, -1,], [2,2,2]],
              [[3, -1, -1], [3, -1, -1,], [3,-1,-1]]]

       Args:
        seq: list of lists to be converted
        padding: value to use for padding.

       Returns:
        padded_array: np.array

    Taken from http://stackoverflow.com/questions/27890052/convert-and-pad-a-list-to-numpy-array
    """

    def find_shape(seq):
        try:
            len_ = len(seq)
        except TypeError:
            return ()
        shapes = [find_shape(subseq) for subseq in seq]
        return (len_,) + tuple(max(sizes) for sizes in zip_longest(*shapes,
                                                                   fillvalue=1))

    def fill_array(arr, seq):
        if arr.ndim == 1:
            try:
                len_ = len(seq)
            except TypeError:
                len_ = 0
            arr[:len_] = seq
            arr[len_:] = padding
        else:
            for subarr, subseq in zip_longest(arr, seq, fillvalue=()):
                fill_array(subarr, subseq)

    padded_array = np.empty(find_shape(seq))
    fill_array(padded_array, seq)
    return padded_array


def semibatch_matmul(values, matrix, name=None):
    """Multiply a batch of matrices by a single matrix.
    Unlike tf.batch_matmul, which requires 2 3-D tensors, semibatch_matmul
    requires one 3-D tensor and one 2-D tensor.
    Arguments:
        values: A tensor of shape `[batch, n, p]`.
        matrix: A tensor of shape `[p, m]`.
        name: (Optional) A name for the operation.
    Returns a tensor of shape `[batch, n, m]`, where the outputs are:
        output[i, ...] = tf.matmul(values[i, ...], matrix)
    """
    with tf.name_scope(name or "SemibatchMatmul"):
        values = tf.convert_to_tensor(values, name="Values")
        matrix = tf.convert_to_tensor(matrix, name="Matrix")

        # Reshape input to be amenable to standard matmul
        values_shape = tf.shape(values, "ValuesShape")
        batch, n, p = values_shape[0], values_shape[1], values_shape[2]
        reshaped = tf.reshape(values, [-1, p],
                              name="CollapseBatchDim")

        output = tf.matmul(reshaped, matrix, name="Matmul")

        # Reshape output back to batched form
        m = matrix.get_shape()[1].value
        output = tf.reshape(output, [batch, n, m], name="Output")
        return output


def parameter_count():
    """Return the total number of parameters in all Tensorflow-defined
    variables, using `tf.trainable_variables()` to get the list of
    variables."""
    return sum(np.product(var.get_shape().as_list())
               for var in tf.trainable_variables())


def scalar_summaries(summaries):
    """Generate a `Summary` protocol buffer containing a set of scalar
    summaries.
    Arguments:
        summaries:  A dictionary mapping summary names to scalar Tensors.
    Returns a single summary op.
    """
    nodes = [tf.summary.scalar(key, value) for key, value in summaries.items()]
    return tf.summary.merge(nodes)


def adam_train_step(loss, iteration, learning_rate, anneal_rate, anneal_every,
                    clip_norm=None, scope=None):
    """Build the graph for a single training step with the Adam optimizer.
    Arguments:
        - loss:             Loss to optimize, as a scalar.
        - iteration:        tf.Variable for the current iteration.
        - learning_rate:    Learning rate to use.
        - anneal_rate:      Factor to anneal learning rate by.
        - anneal_every:     How often to anneal learning rate.
        - clip_norm:        How aggresively to clip the gradients.
        - scope:            Scope to use for ops.
    Return a tuple (gradnorm, loss, train step), where each are nodes in the
    resulting graph. Returned loss is reduced across nodes, if necessary.
    """
    with tf.name_scope(scope or "TrainStep"):
        annealed_rate = tf.train.exponential_decay(
            learning_rate, iteration, decay_steps=anneal_every,
            decay_rate=anneal_rate, staircase=True,
            name="AnnelealedLearningRate")

        # Don't warn on redefining the optimizer type.
        # pylint: disable=redefined-variable-type
        optimizer = tf.train.AdamOptimizer(learning_rate=annealed_rate)
        gradients = optimizer.compute_gradients(loss)

        if clip_norm is None:
            gradnorm = tf.global_norm([grad for (grad, _) in gradients])
        else:
            grads, tvars = list(zip(*gradients))
            clipped, gradnorm = tf.clip_by_global_norm(grads, clip_norm)
            gradients = zip(clipped, tvars)

        step = optimizer.apply_gradients(gradients, global_step=iteration)

    return step, loss, gradnorm


def default_train_summary(loss, gradnorm):
    """Create a summary that should be executed and saved at every iteration,
    using the training feed dict."""
    return scalar_summaries({
        "Train-Loss": loss,
        "Gradient-Norm": gradnorm,
    })


def default_train_step(model, loss):
    """Return an op to do one training step."""
    iteration = tf.Variable(0, trainable=False, name="CurrentIteration")
    return iteration, adam_train_step(loss, iteration, model.learning_rate,
                                      model.anneal_rate, model.anneal_every,
                                      model.clip_norm)
