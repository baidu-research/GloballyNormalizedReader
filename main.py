"""
Main training and evaluation module for QA models.
"""
import click
import json
import random
import featurize
import ops
import gnr
import subprocess
from tempfile import NamedTemporaryFile

from framework import create_model, train_model, session_with_model
from vocab import Vocab


@click.group()
def main():
    """Create, train, and evaluate QA models on SQUAD."""
    pass


@main.command("create")
@click.option("--name", required=True,
              help="Name for this run, used for stored info")
@click.option("--question-layers", default=3, type=int,
              help="Number of LSTM layers for question rep.")
@click.option("--document-layers", default=3, type=int,
              help="Number of LSTM layers for document rep.")
@click.option("--pick-end-word-layers", default=1, type=int,
              help="Number of LSTM layers for end-word-picking rep.")
@click.option("--layer-size", default=200, type=int,
              help="Size of all of the LSTM hidden layers.")
@click.option("--beam-size", default=32, type=int,
              help="Size of the beam during global norm.")
@click.option("--embedding-dropout", default=0.75, type=float,
              help="Dropout applied to word embeddings")
@click.option("--hidden-dropout", default=0.6, type=float,
              help="Dropout applied to LSTM hidden layers")
@click.option("--learning-rate", default=0.0005, type=float,
              help="Initial learning rate for optimizer")
@click.option("--anneal-every", default=1000, type=int,
              help="How often to anneal the learning rate (iterations)")
@click.option("--anneal-rate", default=0.9886, type=float,
              help="Every time annealing happens, multiply LR by this")
@click.option("--clip-norm", default=10., type=float,
              help="Maximum gradient norm")
@click.option("--l2-scale", default=1e-5, type=float,
              help="Scale of the L2 penalty on the weights")
@click.option("--weight-noise", default=1e-6, type=float,
              help="Std-Deviation of weight noise applied to recurrent LSTMs")
@click.option("--replace", default=False, is_flag=True,
              help="Replace the output metagraph, but not the weights")
@click.option("--vocab-path", required=True, type=str,
              help="Path to the vocabulary directory")
@click.option("--seed", type=int, default=1234)
def create(name, question_layers, document_layers, pick_end_word_layers,
           layer_size, beam_size, embedding_dropout, hidden_dropout,
           learning_rate, anneal_every, anneal_rate, clip_norm,
           l2_scale, weight_noise, replace, vocab_path, seed):
    """Create a new QA model."""
    vocab = Vocab(vocab_path)

    random.seed(seed)

    config = gnr.ModelConfig(
        vocab_size=vocab.size,
        question_layers=question_layers,
        document_layers=document_layers,
        pick_end_word_layers=pick_end_word_layers,
        layer_size=layer_size,
        beam_size=beam_size,
        embedding_dropout_prob=embedding_dropout,
        hidden_dropout_prob=hidden_dropout,
        learning_rate=learning_rate,
        anneal_every=anneal_every,
        anneal_rate=anneal_rate,
        clip_norm=clip_norm,
        l2_scale=l2_scale,
        weight_noise=weight_noise)

    create_model("gnr", name, config, gnr, vocab.word_embeddings, replace)


@main.command("train")
@click.option("--name", required=True,
              help="Name of the model to train")
@click.option("--eval-data", default="data/dev.json",
              help="Path to the raw data for evaluation")
@click.option("--data", required=True,
              help="Path to the data, one subdirectory or file per sample")
@click.option("--batch-size", default=32, type=int,
              help="Batch size per training process")
@click.option("--validation-size", default=32, type=int,
              help="Size of the validation set (in total, not per process)")
@click.option("--save-every", default=500, type=int,
              help="How often to save this model to disk")
@click.option("--test-every", default=250, type=int,
              help="How often to run the validation set through the model")
@click.option("--eval-every", default=1000, type=int,
              help="How often to run the entire validation set through the model and score on squad")
@click.option("--max-iterations", default=50000, type=int,
              help="Number of iterations to run model for before stopping")
def train(name, eval_data, data, batch_size, validation_size, save_every,
          test_every, eval_every, max_iterations):
    """Train a previously-created QA model."""
    def squad_eval(predictions):
        with NamedTemporaryFile("w+") as tmp_file:
            json.dump(predictions, tmp_file)
            tmp_file.seek(0)
            command = ["python", "evaluate.py", eval_data, tmp_file.name]
            output = subprocess.check_output(command).decode("utf-8").strip()
            results = json.loads(output)
            return results["exact_match"], results["f1"]

    train_model("gnr", name, gnr, data, batch_size, validation_size,
                save_every, test_every, max_iterations,
                max_to_keep=5, keep_checkpoint_every_n_hours=4,
                save_on_exit=True, eval_every=eval_every, squad_eval=squad_eval)


def evaluate_batch(session, qa_model, examples, vocab):
    """
    Arguments:
        session:  tensorflow session
        qa_model: model object
        examples: list(tuple(question, context))
        vocab:    vocabulary object

    Returns list of predictions in the same order as examples
    """
    batch_features = []
    batch_tokenized_contexts = []
    for question, context in examples:
        features, tokenized_context = featurize.featurize_example(
            question, context, vocab)
        batch_features.append(features)
        batch_tokenized_contexts.append(tokenized_context)

    # Batch together and pad
    features = list(zip(*batch_features))
    questions = ops.lists_to_array(features[0], padding=-1)
    contexts = ops.lists_to_array(features[1], padding=-1)
    same_as_question = ops.lists_to_array(features[2], padding=0)
    repeated_words = ops.lists_to_array(features[3], padding=0)
    repeated_intensity = ops.lists_to_array(features[4], padding=0)
    sent_lens = ops.lists_to_array(features[5], padding=0)

    feed_dict = dict(zip(qa_model.inputs,
                         [questions, contexts, same_as_question, repeated_words,
                          repeated_intensity, sent_lens]))

    # Put the model in evaluation mode
    if qa_model.dropout is not None:
        for dropout in qa_model.dropout:
            feed_dict[dropout] = 1.0

    if qa_model.training is not None:
        feed_dict[qa_model.training] = False

    # Run evaluation
    final_states = session.run(qa_model.outputs[:3], feed_dict=feed_dict)
    sentence_picks, start_word_picks, end_word_picks = final_states

    # Gather top-1 predictions
    predictions = []
    for i, tokenized_context in enumerate(batch_tokenized_contexts):
        sentence, start_word, end_word = sentence_picks[i, 0],\
            start_word_picks[i, 0], end_word_picks[i, 0]
        sentence = min(sentence, len(tokenized_context) - 1)
        start_word = min(start_word, len(tokenized_context[sentence]) - 1)
        predictions.append("".join(tokenized_context[sentence][start_word:end_word+1]))

    return predictions


@main.command("predict")
@click.option("--name", required=True,
              help="Name of the model to evaluate")
@click.option("--data", required=True,
              help="Path to the data")
@click.option("--vocab-path", required=True, type=str,
              help="Path to the vocabulary directory")
@click.option("--output", required=True,
              help="Name of the output .json file")
@click.option("--batch-size", type=int, default=64,
              help="Batch size to run inference")
def predict(name, data, vocab_path, output, batch_size):
    """Generate predictions for a trained model."""
    with session_with_model("gnr", name) as (session, qa_model):
        vocab = Vocab(vocab_path)

        eval_batches = []
        current_batch = []
        for question, context, _, qa_id in featurize.data_stream(data):
            current_batch.append((question, context, qa_id))

            if len(current_batch) == batch_size:
                eval_batches.append(current_batch)
                current_batch = []

        if len(current_batch) > 0:
            eval_batches.append(current_batch)

        predictions = {}
        for idx, batch in enumerate(eval_batches):
            questions, contexts, ids = list(zip(*batch))
            preds = evaluate_batch(
                session, qa_model, zip(questions, contexts), vocab)

            for i, pred in enumerate(preds):
                predictions[ids[i]] = pred

        with open(output, "wt") as handle:
            json.dump(predictions, handle)

        # run the evaluation code
        subprocess.check_call(["python", "evaluate.py", data, output])

if __name__ == "__main__":
    main()
