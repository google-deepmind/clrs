# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Run training of one or more algorithmic tasks from CLRS."""

import functools
import os
import shutil
import csv
import time
from typing import Any, Dict, List

from absl import app
from absl import flags
from absl import logging
import clrs
import jax
import numpy as np
import requests
import tensorflow as tf

# Parameters to tune
flags.DEFINE_list('algorithms', ['bfs'], 'Which algorithms to run.')
flags.DEFINE_list('train_lengths', ['16'],
                  'Which training sizes to use. A size of -1 means '
                  'use the benchmark dataset.')
flags.DEFINE_list('val_lengths', ['16', '32', '64'],
                  'Which validation sizes to use. A size of -1 means '
                  'use the benchmark dataset.')
flags.DEFINE_integer('train_samples', 150, 'Number of training samples for each graph size.')
flags.DEFINE_integer('val_samples', 100, 'Number of validation samples for each graph size.')
flags.DEFINE_integer('epochs', 3, 'Number of epochs for training.')
flags.DEFINE_integer('train_steps', 5, 'Number of training iterations per epoch.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate to use.')

# Other parameters
flags.DEFINE_integer('length_needle', -8,
                     'Length of needle for training and validation '
                     '(not testing) in string matching algorithms. '
                     'A negative value randomizes the length for each sample '
                     'between 1 and the opposite of the value. '
                     'A value of 0 means use always 1/4 of the length of '
                     'the haystack (the default sampler behavior).')
flags.DEFINE_integer('seed', 42, 'Random seed to set')

flags.DEFINE_boolean('random_pos', False,
                     'Randomize the pos input common to all algos.')
flags.DEFINE_boolean('enforce_permutations', False,
                     'Whether to enforce permutation-type node pointers.')
flags.DEFINE_boolean('enforce_pred_as_input', True,
                     'Whether to change pred_h hints into pred inputs.')
flags.DEFINE_integer('batch_size', 32, 'Batch size used for training.')
flags.DEFINE_boolean('chunked_training', False,
                     'Whether to use chunking for training.')
flags.DEFINE_integer('chunk_length', 16,
                     'Time chunk length used for training (if '
                     '`chunked_training` is True.')
flags.DEFINE_integer('hidden_size', 128,
                     'Number of hidden units of the model.')
flags.DEFINE_integer('nb_heads', 1, 'Number of heads for GAT processors')
flags.DEFINE_integer('nb_msg_passing_steps', 1,
                     'Number of message passing steps to run per hint.')
flags.DEFINE_float('grad_clip_max_norm', 0.0,
                   'Gradient clipping by norm. 0.0 disables grad clipping')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')
flags.DEFINE_float('hint_teacher_forcing', 0.0,
                   'Probability that ground-truth teacher hints are encoded '
                   'during training instead of predicted hints. Only '
                   'pertinent in encoded_decoded modes.')
flags.DEFINE_enum('hint_mode', 'encoded_decoded',
                  ['encoded_decoded', 'decoded_only', 'none'],
                  'How should hints be used? Note, each mode defines a '
                  'separate task, with various difficulties. `encoded_decoded` '
                  'requires the model to explicitly materialise hint sequences '
                  'and therefore is hardest, but also most aligned to the '
                  'underlying algorithmic rule. Hence, `encoded_decoded` '
                  'should be treated as the default mode for our benchmark. '
                  'In `decoded_only`, hints are only used for defining '
                  'reconstruction losses. Often, this will perform well, but '
                  'note that we currently do not make any efforts to '
                  'counterbalance the various hint losses. Hence, for certain '
                  'tasks, the best performance will now be achievable with no '
                  'hint usage at all (`none`).')
flags.DEFINE_enum('hint_repred_mode', 'soft', ['soft', 'hard', 'hard_on_eval'],
                  'How to process predicted hints when fed back as inputs.'
                  'In soft mode, we use softmaxes for categoricals, pointers '
                  'and mask_one, and sigmoids for masks. '
                  'In hard mode, we use argmax instead of softmax, and hard '
                  'thresholding of masks. '
                  'In hard_on_eval mode, soft mode is '
                  'used for training and hard mode is used for evaluation.')
flags.DEFINE_boolean('use_ln', True,
                     'Whether to use layer normalisation in the processor.')
flags.DEFINE_boolean('use_lstm', False,
                     'Whether to insert an LSTM after message passing.')
flags.DEFINE_integer('nb_triplet_fts', 8,
                     'How many triplet features to compute?')

flags.DEFINE_enum('encoder_init', 'default',
                  ['default', 'xavier_on_scalars', 'large'],
                  'Initialiser to use for the encoders.')
flags.DEFINE_enum('processor_type', 'triplet_mpnn',
                  ['deepsets', 'mpnn', 'pgn', 'pgn_mask',
                   'triplet_mpnn', 'triplet_pgn', 'triplet_pgn_mask',
                   'gat', 'gatv2', 'gat_full', 'gatv2_full',
                   'gpgn', 'gpgn_mask', 'gmpnn',
                   'triplet_gpgn', 'triplet_gpgn_mask', 'triplet_gmpnn'],
                  'Processor type to use as the network P.')
# Old checkpoint location /tmp/CLRS30
flags.DEFINE_string('checkpoint_path', './checkpoints',
                    'Path in which checkpoints are saved.')
flags.DEFINE_boolean('start_from_checkpoint', False,
                     'Whether to start training from a checkpoint.')
flags.DEFINE_string('checkpoint_name', 'bfs_epoch_0.pkl',
                    'Name of checkpoint pkl file to restore.')
flags.DEFINE_string('dataset_path', '/tmp/CLRS30',
                    'Path in which dataset is stored.')
flags.DEFINE_boolean('freeze_processor', False,
                     'Whether to freeze the processor of the model.')

FLAGS = flags.FLAGS


PRED_AS_INPUT_ALGOS = [
    'binary_search',
    'minimum',
    'find_maximum_subarray',
    'find_maximum_subarray_kadane',
    'matrix_chain_order',
    'lcs_length',
    'optimal_bst',
    'activity_selector',
    'task_scheduling',
    'naive_string_matcher',
    'kmp_matcher',
    'jarvis_march']


def unpack(v):
  try:
    return v.item()  # DeviceArray
  except (AttributeError, ValueError):
    return v


def _iterate_sampler(sampler, batch_size):
  while True:
    yield sampler.next(batch_size)


def _maybe_download_dataset(dataset_path):
  """Download CLRS30 dataset if needed."""
  dataset_folder = os.path.join(dataset_path, clrs.get_clrs_folder())
  if os.path.isdir(dataset_folder):
    logging.info('Dataset found at %s. Skipping download.', dataset_folder)
    return dataset_folder
  logging.info('Dataset not found in %s. Downloading...', dataset_folder)

  clrs_url = clrs.get_dataset_gcp_url()
  request = requests.get(clrs_url, allow_redirects=True)
  clrs_file = os.path.join(dataset_path, os.path.basename(clrs_url))
  os.makedirs(dataset_folder)
  open(clrs_file, 'wb').write(request.content)
  shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
  os.remove(clrs_file)
  return dataset_folder


def make_sampler(length: int,
                 rng: Any,
                 algorithm: str,
                 split: str,
                 batch_size: int,
                 multiplier: int,
                 randomize_pos: bool,
                 enforce_pred_as_input: bool,
                 enforce_permutations: bool,
                 chunked: bool,
                 chunk_length: int,
                 sampler_kwargs: Dict[str, Any],
                 set_num_samples=None):
  """Create a sampler with given options.

  Args:
    length: Size of samples (i.e., number of nodes in the graph).
      A length of -1 will mean that the benchmark
      dataset (for the given split) is used. Positive sizes will instantiate
      samplers of the corresponding size.
    rng: Numpy random state.
    algorithm: The name of the algorithm to sample from.
    split: 'train', 'val' or 'test'.
    batch_size: Samples per batch.
    multiplier: Integer multiplier for the number of samples in the dataset,
      only used for positive sizes. Negative multiplier means infinite samples.
    randomize_pos: Whether to randomize the `pos` input.
    enforce_pred_as_input: Whether to convert fixed pred_h hints to inputs.
    enforce_permutations: Whether to enforce permutation pointers.
    chunked: Whether to chunk the dataset.
    chunk_length: Unroll length of chunks, if `chunked` is True.
    sampler_kwargs: Extra args passed to the sampler.
  Returns:
    A sampler (iterator), the number of samples in the iterator (negative
    if infinite samples), and the spec.
  """
  if length < 0:  # load from file
    # Currently, val & test are taken from CLRS benchmarking dataset
    # with their pre-defined num_samples & sizes
    dataset_folder = _maybe_download_dataset(FLAGS.dataset_path)
    sampler, num_samples, spec = clrs.create_dataset(folder=dataset_folder,
                                                     algorithm=algorithm,
                                                     batch_size=batch_size,
                                                     split=split)
    sampler = sampler.as_numpy_iterator()
  else:
    # num_samples = clrs.CLRS30[split]['num_samples'] * multiplier
    # num_samples = batch_size * multiplier    # The reason they have multiplier seems to control "generate on the fly
                                               # with unlimited data" or "generate in advance with pre-defined size"
    if set_num_samples:
        num_samples = set_num_samples
    else:
        num_samples = batch_size * multiplier
    sampler, spec = clrs.build_sampler(
        algorithm,
        seed=rng.randint(2**32),
        num_samples=num_samples,
        length=length,
        **sampler_kwargs,
        )
    sampler = _iterate_sampler(sampler, batch_size)

  if randomize_pos:
    sampler = clrs.process_random_pos(sampler, rng)
  if enforce_pred_as_input and algorithm in PRED_AS_INPUT_ALGOS:
    spec, sampler = clrs.process_pred_as_input(spec, sampler)
  spec, sampler = clrs.process_permutations(spec, sampler, enforce_permutations)
  if chunked:
    sampler = clrs.chunkify(sampler, chunk_length)
  return sampler, num_samples, spec


def make_multi_sampler(sizes, rng, set_num_samples=None, **kwargs):
  """Create a sampler with cycling sample sizes."""
  ss = []
  tot_samples = 0
  for length in sizes:
    sampler, num_samples, spec = make_sampler(length, rng, set_num_samples=set_num_samples, **kwargs)
    ss.append(sampler)
    tot_samples += num_samples

  def cycle_samplers():
    while True:
      for s in ss:
        yield next(s)
  return cycle_samplers(), tot_samples, spec


def make_multi_sampler_list(sizes, rng, set_num_samples=None, **kwargs):
  ss = []
  tot_samples = 0
  for length in sizes:
    sampler, num_samples, spec = make_sampler(length, rng, set_num_samples=set_num_samples, **kwargs)
    ss.append(sampler)
    tot_samples += num_samples
  return ss, tot_samples, spec


def _concat(dps, axis):
  return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)


def _nb_nodes(feedback: clrs.Feedback, is_chunked) -> int:
  for inp in feedback.features.inputs:
    if inp.location in [clrs.Location.NODE, clrs.Location.EDGE]:
      if is_chunked:
        return inp.data.shape[2]  # inputs are time x batch x nodes x ...
      else:
        return inp.data.shape[1]  # inputs are batch x nodes x ...
  assert False


def collect_and_eval(sampler, model, algo_idx, sample_count, rng_key, decode_hints, extras):
  """Collect batches of output and hint preds and evaluate them."""
  processed_samples = 0
  preds = []
  outputs = []
  total_loss = 0.0

  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    new_rng_key, rng_key = jax.random.split(rng_key)
    cur_preds, cur_outs, hint_preds = model.predict(new_rng_key, feedback.features, algo_idx, return_hints=True)
    preds.append(cur_preds)

    nb_nodes = _nb_nodes(feedback, is_chunked=False)
    for truth in feedback.outputs:
        total_loss += clrs.output_loss(
            truth=truth,
            pred=cur_outs[truth.name],
            nb_nodes=nb_nodes,
        )

    # Optionally accumulate hint losses.
    if decode_hints:
        lengths = feedback.features.lengths
        for truth in feedback.features.hints:
            total_loss += clrs.hint_loss(
                truth=truth,
                preds=[x[truth.name] for x in hint_preds],
                lengths=lengths,
                nb_nodes=nb_nodes,
            )

    processed_samples += batch_size
  outputs = _concat(outputs, axis=0)
  preds = _concat(preds, axis=0)
  out = clrs.evaluate(outputs, preds)

  if extras:
    out.update(extras)
  return total_loss, {k: unpack(v) for k, v in out.items()}


def eval_score(model, feedback, algo_idx, rng_key):
  """Calculate evaluation score for current model given a set of features."""
  outputs = feedback.outputs
  preds, _, _ = model.predict(rng_key, feedback.features, algo_idx)
  out = clrs.evaluate(outputs, preds)
  return out['score']


def create_samplers(rng, train_lengths: List[int], val_lengths: List[int]):
  """Create all the samplers."""
  train_samplers = []
  val_samplers = []
  val_sample_counts = []
  test_samplers = []
  test_sample_counts = []
  spec_list = []

  for algo_idx, algorithm in enumerate(FLAGS.algorithms):
    # Make full dataset pipeline run on CPU (including prefetching).
    with tf.device('/cpu:0'):

      if algorithm in ['naive_string_matcher', 'kmp_matcher']:
        # Fixed haystack + needle; variability will be in needle
        # Still, for chunked training, we maintain as many samplers
        # as train lengths, since, for each length there is a separate state,
        # and we must keep the 1:1 relationship between states and samplers.
        max_length = max(train_lengths)
        if max_length > 0:  # if < 0, we are using the benchmark data
          max_length = (max_length * 5) // 4
        train_lengths = [max_length]
        if FLAGS.chunked_training:
          train_lengths = train_lengths * len(train_lengths)

      logging.info('Creating samplers for algo %s', algorithm)

      p = tuple([0.1 + 0.1 * i for i in range(9)])
      if p and algorithm in ['articulation_points', 'bridges',
                             'mst_kruskal', 'bipartite_matching']:
        # Choose a lower connection probability for the above algorithms,
        # otherwise trajectories are very long
        p = tuple(np.array(p) / 2)
      length_needle = FLAGS.length_needle
      sampler_kwargs = dict(p=p, length_needle=length_needle)
      if length_needle == 0:
        sampler_kwargs.pop('length_needle')

      common_sampler_args = dict(
          algorithm=FLAGS.algorithms[algo_idx],
          rng=rng,
          enforce_pred_as_input=FLAGS.enforce_pred_as_input,
          enforce_permutations=FLAGS.enforce_permutations,
          chunk_length=FLAGS.chunk_length,
          )

      train_args = dict(sizes=train_lengths,
                        split='train',
                        batch_size=FLAGS.batch_size,
                        multiplier=FLAGS.train_steps,
                        # Original code passes "-1", which generates unlimited samples
                        # on the fly. If we pass a positive integer, then it will
                        # generate multiplier * clrs.CLRS30["train"]["num_samples"] (=> I changed)
                        # samples in advance. Then each sampler.next() randomly samples
                        # a subset of data with batch_size from these pre-generated samples.
                        set_num_samples=FLAGS.train_samples,
                        randomize_pos=FLAGS.random_pos,
                        chunked=FLAGS.chunked_training,
                        sampler_kwargs=sampler_kwargs,
                        **common_sampler_args)
      train_sampler, _, spec = make_multi_sampler(**train_args)

      mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
      val_args = dict(sizes=val_lengths,
                      split='val',
                      batch_size=None,
                      multiplier=mult,
                      set_num_samples=FLAGS.val_samples,
                      randomize_pos=FLAGS.random_pos,
                      chunked=False,
                      sampler_kwargs=sampler_kwargs,
                      **common_sampler_args)
      val_sampler, val_samples, spec = make_multi_sampler_list(**val_args)

      test_args = dict(sizes=[-1],
                       split='test',
                       batch_size=32,
                       multiplier=2 * mult,
                       randomize_pos=False,
                       chunked=False,
                       sampler_kwargs={},
                       **common_sampler_args)
      test_sampler, test_samples, spec = make_multi_sampler(**test_args)

    spec_list.append(spec)
    train_samplers.append(train_sampler)
    val_samplers.append(val_sampler)
    val_sample_counts.append(val_samples)
    test_samplers.append(test_sampler)
    test_sample_counts.append(test_samples)

  return (train_samplers,
          val_samplers, val_sample_counts,
          test_samplers, test_sample_counts,
          spec_list)


def setup_csv(train_model):
  csv_path = FLAGS.checkpoint_path + "/csv/"
  if not os.path.exists(csv_path):
      os.makedirs(csv_path)
  csv_files = []
  csv_writers = []
  if FLAGS.start_from_checkpoint:
      write_flag = 'a'
  else:
      write_flag = 'w'
  for algo_idx, algorithm in enumerate(FLAGS.algorithms):
    csv_name = algorithm + '.csv'
    csv_file = open(csv_path + csv_name, write_flag, newline='')
    csv_files += [csv_file]
    fieldnames = ["train_loss",
                  "train_accuracy",
                  "learning_rate",
                  "len_train_ds",
                  "len_val_ds",
                  "batches_per_epoch",
                  "time_per_epoch",
                  "epoch",
                  "step",
                  "avg_val_loss",
                  "avg_val_accuracy"]

    # Validation result for each graph size
    for length in FLAGS.val_lengths:
        fieldnames.append('val_size_' + str(length) + '_loss')
        fieldnames.append('val_size_' + str(length) + '_accuracy')

    # Track model weight evolution
    for layer, _ in train_model.params.items():
      if 'w' in train_model.params[layer]:
        fieldnames.append(layer + '.weight_norm')
        # fieldnames.append(layer + '.weight_list')

    
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_writers += [csv_writer]

  return csv_files, csv_writers


def main(unused_argv):
  if FLAGS.hint_mode == 'encoded_decoded':
    encode_hints = True
    decode_hints = True
  elif FLAGS.hint_mode == 'decoded_only':
    encode_hints = False
    decode_hints = True
  elif FLAGS.hint_mode == 'none':
    encode_hints = False
    decode_hints = False
  else:
    raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')

  train_lengths = [int(x) for x in FLAGS.train_lengths]
  val_lengths = [int(x) for x in FLAGS.val_lengths]

  rng = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(rng.randint(2**32))

  # Create samplers
  (train_samplers,
   val_samplers, val_sample_counts,
   test_samplers, test_sample_counts,
   spec_list) = create_samplers(rng, train_lengths, val_lengths)

  processor_factory = clrs.get_processor_factory(
      FLAGS.processor_type,
      use_ln=FLAGS.use_ln,
      nb_triplet_fts=FLAGS.nb_triplet_fts,
      nb_heads=FLAGS.nb_heads
  )
  model_params = dict(
      processor_factory=processor_factory,
      hidden_dim=FLAGS.hidden_size,
      encode_hints=encode_hints,
      decode_hints=decode_hints,
      encoder_init=FLAGS.encoder_init,
      use_lstm=FLAGS.use_lstm,
      learning_rate=FLAGS.learning_rate,
      grad_clip_max_norm=FLAGS.grad_clip_max_norm,
      checkpoint_path=FLAGS.checkpoint_path,
      freeze_processor=FLAGS.freeze_processor,
      dropout_prob=FLAGS.dropout_prob,
      hint_teacher_forcing=FLAGS.hint_teacher_forcing,
      hint_repred_mode=FLAGS.hint_repred_mode,
      nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
      )

  eval_model = clrs.models.BaselineModel(
      spec=spec_list,
      dummy_trajectory=[next(val_samplers[0][0])],
      **model_params
  )
  if FLAGS.chunked_training:
    train_model = clrs.models.BaselineModelChunked(
        spec=spec_list,
        dummy_trajectory=[next(t) for t in train_samplers],
        **model_params
        )
  else:
    train_model = eval_model


  # Training loop.
  best_score = -1.0
  current_train_items = [0] * len(FLAGS.algorithms)
  step = 0
  epoch = 0

  # Make sure scores improve on first step, but not overcome best score
  # until all algos have had at least one evaluation.
  val_scores = [-99999.9] * len(FLAGS.algorithms)
  length_idx = 0

  while epoch < FLAGS.epochs:
    time_per_epoch = 0
    old_current_train_items = current_train_items.copy()
    while step < epoch*FLAGS.train_steps + FLAGS.train_steps:
      feedback_list = [next(t) for t in train_samplers]

      # Initialize model.
      if step == 0:
        all_features = [f.features for f in feedback_list]
        if FLAGS.chunked_training:
          # We need to initialize the model with samples of all lengths for
          # all algorithms. Also, we need to make sure that the order of these
          # sample sizes is the same as the order of the actual training sizes.
          all_length_features = [all_features] + [
              [next(t).features for t in train_samplers]
              for _ in range(len(train_lengths))]
          train_model.init(all_length_features[:-1], FLAGS.seed + 1)
        else:
          train_model.init(all_features, FLAGS.seed + 1)
        if FLAGS.start_from_checkpoint:
            epoch = int(FLAGS.checkpoint_name[FLAGS.checkpoint_name.index(".") - 1]) + 1
            step = epoch * FLAGS.train_steps
            train_model.restore_model(FLAGS.checkpoint_name, only_load_processor=False)

        csv_files, csv_writers = setup_csv(train_model)

      # Training step.
      for algo_idx in range(len(train_samplers)):
        feedback = feedback_list[algo_idx]
        rng_key, new_rng_key = jax.random.split(rng_key)
        if FLAGS.chunked_training:
          # In chunked training, we must indicate which training length we are
          # using, so the model uses the correct state.
          length_and_algo_idx = (length_idx, algo_idx)
        else:
          # In non-chunked training, all training lengths can be treated equally,
          # since there is no state to maintain between batches.
          length_and_algo_idx = algo_idx
    
        start_time = time.time()
        cur_loss, cur_lr = train_model.feedback(rng_key, feedback, length_and_algo_idx)
        time_per_step = time.time() - start_time
        time_per_epoch += time_per_step

        rng_key, new_rng_key = jax.random.split(new_rng_key)
        cur_score = eval_score(train_model, feedback, algo_idx, new_rng_key)

        if FLAGS.chunked_training:
          examples_in_chunk = np.sum(feedback.features.is_last).item()
        else:
          examples_in_chunk = len(feedback.features.lengths)
        current_train_items[algo_idx] += examples_in_chunk
        # to compare results with the standard 32-batch_size experiments
        logging.info('Algo %s step %i: current loss %f, current score %f, current lr %f, current_train_items %i, '
                     'time_per_step %f.',
                     FLAGS.algorithms[algo_idx], step,
                     cur_loss, cur_score, cur_lr, current_train_items[algo_idx],
                     time_per_step)

        # log at the last training step of each epoch.
        if step == epoch*FLAGS.train_steps + FLAGS.train_steps - 1:
            logging.info('Algo %s epoch %i: current loss %f, current score %f, current lr %f, current_train_items %i, '
                         'time_per_epoch %f.',
                         FLAGS.algorithms[algo_idx], epoch,
                         cur_loss, cur_score, cur_lr, current_train_items[algo_idx],
                         time_per_epoch)
            csv_writers[algo_idx].writerow({
                              "train_loss": cur_loss,
                              "train_accuracy": cur_score,
                              "learning_rate": cur_lr,
                              "len_train_ds": current_train_items[algo_idx] - old_current_train_items[algo_idx],
                              "batches_per_epoch": FLAGS.train_steps,
                              "time_per_epoch": time_per_epoch,
                              "epoch": epoch,
                              "step": step
                              })
            
            weight_logs = {"epoch": epoch}
            for layer, _ in train_model.params.items():
              if 'w' in train_model.params[layer]:
                norm = jax.numpy.linalg.norm(train_model.params[layer]['w'], ord=2)
                weight_logs[layer + '.weight_norm'] = norm
                # weight_logs[layer + '.weight_list'] = jax.numpy.ravel(train_model.params[layer]['w'])
            csv_writers[algo_idx].writerow(weight_logs)

      # Validation step at the last training step of each epoch.
      if step == epoch*FLAGS.train_steps + FLAGS.train_steps - 1:
        eval_model.params = train_model.params
        for algo_idx in range(len(train_samplers)):
          common_extras = {'examples_seen': current_train_items[algo_idx],
                           'step': step,
                           'algorithm': FLAGS.algorithms[algo_idx]}

          # Validation info.
          new_rng_key, rng_key = jax.random.split(rng_key)
          val_losses = []
          val_accuracies = []
          for i, sampler in enumerate(val_samplers[algo_idx]):
              val_loss, val_stats = collect_and_eval(
                  sampler,
                  eval_model,
                  algo_idx,
                  # functools.partial(eval_model.predict, algorithm_index=algo_idx),
                  FLAGS.val_samples,
                  new_rng_key,
                  decode_hints,
                  extras=common_extras)
              logging.info('(val) algo %s size %d epoch %d: loss=%f, %s',
                           FLAGS.algorithms[algo_idx], val_lengths[i], epoch,
                           val_loss, val_stats)
              val_losses += [val_loss]
              val_accuracies += [val_stats['score']]

          val_dict = {}
          for i, length in enumerate(val_lengths):
              val_dict[f"val_size_{length}_loss"] = val_losses[i]
              val_dict[f"val_size_{length}_accuracy"] = val_accuracies[i]
          val_dict["avg_val_loss"] = sum(val_losses) / len(val_losses)
          val_dict["avg_val_accuracy"] = sum(val_accuracies) / len(val_accuracies)
          val_dict["len_val_ds"] = val_sample_counts[algo_idx]
          val_dict["time_per_epoch"] = time_per_epoch
          val_dict["epoch"] = epoch
          csv_writers[algo_idx].writerow(val_dict)
          val_scores[algo_idx] = sum(val_accuracies) / len(val_accuracies)

          # logging.info('Checkpointing algorithm %s for epoch %d', FLAGS.algorithms[algo_idx], epoch)
          # checkpoint_name = FLAGS.algorithms[algo_idx] + '_epoch_' + str(epoch) + '.pkl' 
          # train_model.save_model(checkpoint_name)


        # If best total score, update best checkpoint.
        # Also save a best checkpoint on the first step.
        msg = (f'best avg val score was '
               f'{best_score/len(FLAGS.algorithms):.3f}, '
               f'current avg val score is {np.mean(val_scores):.3f}, '
               f'val scores are: ')
        msg += ', '.join(
            ['%s: %.3f' % (x, y) for (x, y) in zip(FLAGS.algorithms, val_scores)])
        if (sum(val_scores) > best_score) or step == 0:
          best_score = sum(val_scores)
          logging.info('Checkpointing best model, %s', msg)
          train_model.save_model('best.pkl')
        else:
          logging.info('Not saving new best model, %s', msg)

      step += 1
      length_idx = (length_idx + 1) % len(train_lengths)
    epoch += 1

  logging.info('Restoring best model from checkpoint...')
  eval_model.restore_model('best.pkl', only_load_processor=False)

  for algo_idx in range(len(train_samplers)):
    common_extras = {'examples_seen': current_train_items[algo_idx],
                     'step': step,
                     'algorithm': FLAGS.algorithms[algo_idx]}

    # Test info.
    new_rng_key, rng_key = jax.random.split(rng_key)
    test_loss, test_stats = collect_and_eval(
        test_samplers[algo_idx],
        # functools.partial(eval_model.predict, algorithm_index=algo_idx),
        eval_model,
        algo_idx,
        test_sample_counts[algo_idx],
        new_rng_key,
        decode_hints,
        extras=common_extras)
    logging.info('(test) algo %s : loss=%f, %s', FLAGS.algorithms[algo_idx],
                 test_loss, test_stats)

  logging.info('Done!')
  for csv_file in csv_files:
      csv_file.close()


if __name__ == '__main__':
  # import pdb; pdb.set_trace()
  app.run(main)
