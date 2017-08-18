"""
Utilities for saving and restoring models in a structured format to and from
metagraph files.
"""
from collections import namedtuple
from contextlib import contextmanager
import os
import random
import time

import tensorflow as tf

import ops

# File to store the model graph to.
GRAPH_FILENAME = "model.graph"

# File to store variable values to.
CHECKPOINT_FILENAME = "step"

# Keys for collections in the constructed TensorFlow graph.
MODEL_INPUTS = "ModelInputs"
MODEL_OUTPUTS = "ModelOutputs"
MODEL_GRADNORM = "ModelGradnorm"
MODEL_LOSS = "ModelLoss"
MODEL_TRAIN_STEP = "ModelTrainStep"
MODEL_CURRENT_ITERATION = "ModelIterationVar"
MODEL_TRAIN_SUMMARY = "ModelTrainSummary"
MODEL_VALID_SUMMARY = "ModelValidSummary"
MODEL_DROPOUT = "ModelDropout"
MODEL_TRAINING = "ModelTraining"
MODEL_EMBEDDING = "ModelEmbedding"
MODEL_INIT_EMBEDDING = "ModelInitEmbedding"
MODEL_SQUAD_INPUTS = "ModelSquadInputs"
MODEL_SQUAD_SUMMARY = "ModelSquadSummary"

# Base class for our Model namedtuple.
_Model = namedtuple("Model", [
    "loss",
    "step",
    "train_summary",
    "valid_summary",
    "iteration",
    "gradnorm",
    "inputs",
    "outputs",
    "dropout",
    "training",
    "embedding_init",
    "embedding_placeholder",
    "squad_summary",
    "squad_inputs",
])


class Model(_Model):
    """
    A Model is a tuple of TensorFlow graph nodes that are necessary for running
    training, inference, and graph analysis. Models can be saved and restored
    from metagraphs using store() and restore().
    """
    def store(self, graph):
        """Store references to key nodes in the graph (denoted by fields of
        this Model) in collections in the provided graph.

        If this is called prior to saving the metagraph, Model.restore can be
        used to recover the Model after loading the metagraph.
        """
        for node in self.inputs:
            graph.add_to_collection(MODEL_INPUTS, node)

        for node in self.outputs:
            graph.add_to_collection(MODEL_OUTPUTS, node)

        for node in self.squad_inputs:
            graph.add_to_collection(MODEL_SQUAD_INPUTS, node)

        graph.add_to_collection(MODEL_LOSS, self.loss)
        graph.add_to_collection(MODEL_GRADNORM, self.gradnorm)
        graph.add_to_collection(MODEL_TRAIN_STEP, self.step)
        graph.add_to_collection(MODEL_CURRENT_ITERATION, self.iteration)
        graph.add_to_collection(MODEL_TRAIN_SUMMARY, self.train_summary)
        graph.add_to_collection(MODEL_VALID_SUMMARY, self.valid_summary)
        graph.add_to_collection(MODEL_INIT_EMBEDDING, self.embedding_init)
        graph.add_to_collection(MODEL_EMBEDDING, self.embedding_placeholder)
        graph.add_to_collection(MODEL_SQUAD_SUMMARY, self.squad_summary)

        if self.dropout is not None:
            for node in self.dropout:
                graph.add_to_collection(MODEL_DROPOUT, node)

        if self.training is not None:
            graph.add_to_collection(MODEL_TRAINING, self.training)

    @staticmethod
    def restore(graph):
        """Recover a Model saved to a metagraph by finding the key nodes in the
        graph in collections in the graph. A model saved with store() prior to
        saving the metagraph can be restored with Model.restore().
        """
        dropout_collection = tf.get_collection(MODEL_DROPOUT)
        dropout = dropout_collection if dropout_collection else None

        training_collection = tf.get_collection(MODEL_TRAINING)
        training = training_collection[0] if training_collection else None

        return Model(
            inputs=graph.get_collection(MODEL_INPUTS),
            outputs=graph.get_collection(MODEL_OUTPUTS),
            loss=graph.get_collection(MODEL_LOSS)[0],
            gradnorm=graph.get_collection(MODEL_GRADNORM)[0],
            step=graph.get_collection(MODEL_TRAIN_STEP)[0],
            iteration=graph.get_collection(MODEL_CURRENT_ITERATION)[0],
            train_summary=graph.get_collection(MODEL_TRAIN_SUMMARY)[0],
            valid_summary=graph.get_collection(MODEL_VALID_SUMMARY)[0],
            dropout=dropout,
            training=training,
            embedding_init=graph.get_collection(MODEL_INIT_EMBEDDING)[0],
            embedding_placeholder=graph.get_collection(MODEL_EMBEDDING)[0],
            squad_inputs=graph.get_collection(MODEL_SQUAD_INPUTS),
            squad_summary=graph.get_collection(MODEL_SQUAD_SUMMARY)[0])


def split_batch(feed_dict):
    """Divide the batch into 2"""
    out = [{}, {}]

    for key, value in feed_dict.items():

        # If the input is a scalar
        if len(key.get_shape()) == 0:
            out[0][key] = value
            out[1][key] = value
        else:
            batch_size = value.shape[0]
            if batch_size == 1:
                return [feed_dict]

            else:
                out[0][key] = value[batch_size // 2:]
                out[1][key] = value[:batch_size // 2]

    return out


def train_loop(session, train_feeds, valid_feed, eval_feeds, model, saver, file_writer,
               checkpoint, save_every, test_every, max_iterations, eval_every,
               squad_eval):
    """Run many iterations of training for a model.

    Arguments:
        - session:        TensorFlow session for running ops.
        - train_feeds:    Feed dict generator with training data.
        - valid_feed:     Feed dict with validation data; used if a validation
                          test is done on this iteration.
        - model:          Model containing graph nodes.
        - saver:          tf.train.Saver for saving to checkpoints.
        - file_writer:    A FileWriter object used to save summaries,
                          or None to save no summaries in this process.
        - checkpoint:     Which checkpoint to save to.
        - save_every:     How often to save (how many iterations between saves)
        - test_every:     How often to run validation tests (in iterations).
        - max_iterations: Number of iterations to run training for.

    In addition to running a training iterations, print logs of the step count,
    time, gradnorm, etc, and use the train and validation summary nodes and the
    summary writer to log all metrics to TensorBoard. If necessary, run
    validation loss and save the weights to a checkpoint.
    """
    log_fmt = ("{:.2f} - \tStep: \t{:<5} \tTime: \t{:.3f} \tGradNorm: "
               "\t{:.3f}  \tTrain: \t{:.3f} \tSent: \t{:.3f} \tStart: \t{:.3f} \tEnd: \t{:.3f}")
    train_nodes = [model.step, model.loss, model.iteration,
                   model.gradnorm, model.train_summary, model.outputs[3],
                   model.outputs[4], model.outputs[5]]

    start_time = time.time()
    for train_feed in train_feeds:
        # Use an out-of-memory strategy with retries
        iterations = []
        to_run = [train_feed]
        i = 0
        while i < len(to_run):
            try:
                # Run the training iteration. Collect timing information and auxiliary
                # information such as gradnorm and loss.
                step_start_time = time.time()
                train_out = session.run(train_nodes, feed_dict=train_feed)
                _, train_loss, current_iteration, train_gradnorm, summary, sent_correct, ws_correct, we_correct = train_out
                duration = time.time() - step_start_time
                iterations.append(current_iteration)

                # Log statistics to console and to TensorBoard.
                total_time = time.time() - start_time
                log = log_fmt.format(total_time, current_iteration,
                                     duration, train_gradnorm, train_loss,
                                     sent_correct, ws_correct, we_correct)
                print(log, flush=True)
                if file_writer is not None:
                    file_writer.add_summary(summary, current_iteration)

                i += 1

            # Catch OOM
            except tf.errors.ResourceExhaustedError:
                prev_batches = len(to_run[i:])

                # Shrink the batch size
                to_run = [shrunk for unshrunk in to_run[i:] for shrunk in split_batch(unshrunk)]

                if prev_batches == len(to_run):
                    raise  # ("Unable to shrink enough. OOM")

                # reset so that we can process
                i = 0

            except tf.errors.InternalError as e:
                if "Failed to call ThenRnnForward" in str(e):
                    prev_batches = len(to_run[i:])

                    # Shrink the batch size
                    to_run = [shrunk for unshrunk in to_run[i:] for shrunk in split_batch(unshrunk)]

                    if prev_batches == len(to_run):
                        raise  # ("Unable to shrink enough. OOM")

                    # reset so that we can process
                    i = 0

                else:
                    raise

        # Run validation tests if it is the right iteration to do so.
        if any(current_iteration % test_every == 0 for current_iteration in iterations):
            valid_loss, summary = session.run(
                [model.loss, model.valid_summary], feed_dict=valid_feed)
            log += " \tValid: {:.3f}".format(valid_loss)
            if file_writer is not None:
                file_writer.add_summary(summary, current_iteration)

        if any(current_iteration % eval_every == 0 for current_iteration in iterations):
            print("Evaluating model on SQUAD...", flush=True)
            predictions = {}
            for ids, contexts, eval_feed in eval_feeds:
                sents, starts, ends = session.run(model.outputs[:3], feed_dict=eval_feed)
                for i, context in enumerate(contexts):
                    sent, start, end = sents[i, 0], starts[i, 0], ends[i, 0]
                    predictions[ids[i]] = "".join(context[sent][start:end + 1])
            exact_match, f1 = squad_eval(predictions)
            print("EM: {}, F1: {}".format(exact_match, f1), flush=True)
            summary_feed = dict(zip(model.squad_inputs, [exact_match, f1]))

            summary = session.run(model.squad_summary,
                                  feed_dict=summary_feed)

            if file_writer is not None:
                file_writer.add_summary(summary, current_iteration)

        # Save if it is the right iteration to do so.
        last_iteration = (max_iterations is not None
                          and current_iteration > max_iterations)
        if current_iteration % save_every == 0 or last_iteration:
                saver.save(session, checkpoint, model.iteration,
                           write_meta_graph=False)

        # Stop if we're done!
        if last_iteration:
            break


def create_model(model_type, name, config, module, embeddings, replace):
    """Create a new model from one of our available model types.

    Generate the TensorFlow graph, count and print the number of parameters,
    save the graph to a metagraph file, and, if `replace` is not enabled,
    initialize the model with a random set of initial weights.

    Arguments:
        - model_type:   Type of model, e.g. vocal, duration, grapheme, etc.
        - name:         Name for the model.
        - config:       A configuration namedtuple.
        - module:       A module to call build_model() on.
        - embeddings:   np.array to initialize the models word embeddings.
        - replace:      If True, don't initialize the model with new weights.

    The graph is created by calling `module.build_model(config)`.
    """
    graph = tf.Graph()
    with graph.as_default():
        # Ensure order is consistent across multiple runs.
        tf.set_random_seed(random.randint(0, 10000))

        model = module.build_model(config)
        model.store(graph)
        saver = tf.train.Saver()

        total_params = ops.parameter_count()
        print("Total Parameters: {:.2f}M".format(total_params / 1e6))

    checkpoint_dir = os.path.join("runs", model_type, name, "checkpoint")
    graph_file = os.path.join(checkpoint_dir, GRAPH_FILENAME)
    checkpoint = os.path.join(checkpoint_dir, "step")
    os.makedirs(checkpoint_dir, exist_ok=True)

    with tf.Session(graph=graph) as session:
        # Dump the graph separately from the saved variables, so that they can
        # be loaded independently.
        tf.train.export_meta_graph(graph_file)

        # Initialize variables and save them to a checkpoint.
        if not replace:
            session.run(tf.global_variables_initializer())
            session.run(model.embedding_init,
                        feed_dict={model.embedding_placeholder: embeddings})
            saver.save(session, checkpoint, model.iteration,
                       write_meta_graph=False)


def train_model(model_type, name, module, data, batch_size, validation_size,
                save_every, test_every, max_iterations, max_to_keep,
                keep_checkpoint_every_n_hours, save_on_exit, eval_every, squad_eval):
    """Train a previously created model.

    This function loads a model from disk, train it for `max_iterations`,
    saving and logging progress as it goes. When launched properly with SLURM,
    the model will be saved when the job times out (if `save_on_exit` is True)
    and the job will be restarted (if `restart` is True).

    Arguments:
        - model_type:   The type of model, e.g. duration, vocal, grapheme, etc.
        - name:         The name of the model, same as with create.
        - module:       The Python module corresponding to the model type.
        - data:         The path to the data files.
        - batch_size:   The batch size (per GPU) for training.
        - validation_size: The validation size (total) for testing.
        - save_every:   How many iterations to wait between saving.
        - test_every:   How many iterations to wait between testing.
        - max_iterations: How many iterations to run for.
        - max_to_keep:  Passed to the tf.Saver.
        - keep_checkpoint_every_n_hours:  Passed to the tf.Saver.
        - save_on_exit: If True, save on SLURM timeout signal.
    """
    run_dir = os.path.join("runs", model_type, name)
    checkpoint_dir = os.path.join(run_dir, "checkpoint")
    checkpoint = os.path.join(checkpoint_dir, CHECKPOINT_FILENAME)
    graph_file = os.path.join(checkpoint_dir, GRAPH_FILENAME)

    with tf.Session() as session:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver = tf.train.import_meta_graph(graph_file)
        saver.restore(session, latest_checkpoint)
        model = Model.restore(session.graph)

        current_iteration = session.run(model.iteration)
        train_stream, valid_stream, eval_stream = module.load_input_data(
            data, batch_size, validation_size, current_iteration)

        # Load a single validation sample at once
        valid_data = next(valid_stream)

        train_feeds = (dict(zip(model.inputs, sample))
                       for sample in train_stream)
        valid_feed = dict(zip(model.inputs, valid_data))

        # Set the dropout and training parameters for validation.
        if model.dropout is not None:
            for dropout in model.dropout:
                valid_feed[dropout] = 1.0
        if model.training is not None:
            valid_feed[model.training] = False

        eval_feeds = []
        for ids, contexts, features in eval_stream:
            eval_feed = dict(zip(model.inputs, features))
            if model.dropout is not None:
                for dropout in model.dropout:
                    eval_feed[dropout] = 1.0
            if model.training is not None:
                eval_feed[model.training] = False
            eval_feeds.append((ids, contexts, eval_feed))

        log_directory = os.path.join(run_dir, "logs")
        file_writer = tf.summary.FileWriter(log_directory, session.graph)

        saver = tf.train.Saver(
            max_to_keep=max_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
        train_loop(session=session, train_feeds=train_feeds,
                   valid_feed=valid_feed, eval_feeds=eval_feeds, model=model, saver=saver,
                   file_writer=file_writer, checkpoint=checkpoint,
                   save_every=save_every, test_every=test_every,
                   max_iterations=max_iterations, eval_every=eval_every, squad_eval=squad_eval)


@contextmanager
def session_with_model(model_type, name):
    """Create a context manager that will start a TensorSwift session and load
    a model into it, and return the session and the model.

    Example:

        with model.session_with_model("grapheme", name) as session, model:
            session.run(model.outputs, ...)
    """
    checkpoint_dir = os.path.join("runs", model_type, name, "checkpoint")
    graph_file = os.path.join(checkpoint_dir, GRAPH_FILENAME)
    graph = tf.Graph()
    with tf.Session(graph=graph) as session:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver = tf.train.import_meta_graph(graph_file)
        saver.restore(session, latest_checkpoint)
        model = Model.restore(graph)

        yield session, model
