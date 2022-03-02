# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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

"""Run a full test run for one or more algorithmic tasks from CLRS."""

import os
import shutil
import time
from absl import app
from absl import flags
from absl import logging

import clrs
import jax
import jax.numpy as jnp
import requests
import tensorflow as tf


flags.DEFINE_string('algorithm', 'bfs', 'Which algorithm to run.')
flags.DEFINE_integer('seed', 42, 'Random seed to set')

flags.DEFINE_integer('batch_size', 32, 'Batch size used for training.')
flags.DEFINE_boolean('chunked_training', False,
                     'Whether to use chunking for training.')
flags.DEFINE_integer('chunk_length', 100,
                     'Time chunk length used for training (if '
                     '`chunked_training` is True.')
flags.DEFINE_integer('train_items', 160000,
                     'Number of items (i.e., individual examples, possibly '
                     'repeated) processed during training. With non-chunked'
                     'training, this is the number of training batches times '
                     'the number of training steps. For chunked training, '
                     'as many chunks will be processed as needed to get these '
                     'many full examples.')
flags.DEFINE_integer('eval_every', 320,
                     'Logging frequency (in training examples).')
flags.DEFINE_boolean('verbose_logging', False, 'Whether to log aux losses.')

flags.DEFINE_integer('hidden_size', 128,
                     'Number of hidden size units of the model.')
flags.DEFINE_float('learning_rate', 0.003, 'Learning rate to use.')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')
flags.DEFINE_integer('nb_heads', 1, 'Number of heads for GAT processors')

flags.DEFINE_boolean('encode_hints', True,
                     'Whether to provide hints as model inputs.')
flags.DEFINE_boolean('decode_hints', True,
                     'Whether to provide hints as model outputs.')
flags.DEFINE_boolean('use_lstm', False,
                     'Whether to insert an LSTM after message passing.')
flags.DEFINE_boolean('decode_diffs', True,
                     'Whether to predict masks within the model.')
flags.DEFINE_enum(
    'processor_type', 'mpnn',
    ['deepsets', 'mpnn', 'pgn', 'pgn_mask',
     'gat', 'gatv2', 'gat_full', 'gatv2_full',
     'memnet_full', 'memnet_masked'],
    'Whether to predict masks within the model.')

flags.DEFINE_string('checkpoint_path', '/tmp/CLRS30',
                    'Path in which checkpoints are saved.')
flags.DEFINE_string('dataset_path', '/tmp/CLRS30',
                    'Path in which dataset is stored.')
flags.DEFINE_boolean('freeze_processor', False,
                     'Whether to freeze the processor of the model.')

FLAGS = flags.FLAGS

CLRS_FILE_NAME = 'CLRS30.tar.gz'
CLRS_FOLDER = 'CLRS30'
DATASET_GCP_URL = f'https://storage.googleapis.com/dm-clrs/{CLRS_FILE_NAME}'


def unpack(v):
  try:
    return v.item()  # DeviceArray
  except AttributeError:
    return v


def evaluate(rng_key, model, feedback, extras=None, verbose=False):
  """Evaluates a model on feedback."""
  out = {}
  predictions, aux = model.predict(rng_key, feedback.features)
  out.update(clrs.evaluate(feedback, predictions))
  if extras:
    out.update(extras)
  if verbose:
    out.update(model.verbose_loss(feedback, aux))
  return {k: unpack(v) for k, v in out.items()}


def download_dataset():
  """Downloads CLRS30 dataset."""
  request = requests.get(DATASET_GCP_URL, allow_redirects=True)
  clrs_file = os.path.join(FLAGS.dataset_path, CLRS_FILE_NAME)
  os.makedirs(FLAGS.dataset_path)
  open(clrs_file, 'wb').write(request.content)
  shutil.unpack_archive(clrs_file, extract_dir=FLAGS.dataset_path)
  extracted_folder = os.path.join(FLAGS.dataset_path, CLRS_FOLDER)
  for file in os.listdir(extracted_folder):
    shutil.move(os.path.join(extracted_folder, file),
                os.path.join(FLAGS.dataset_path, file))
  os.remove(clrs_file)
  shutil.rmtree(extracted_folder)


def main(unused_argv):
  # Use canonical CLRS-30 samplers.
  clrs30_spec = clrs.CLRS30
  logging.info('Using CLRS30 spec: %s', clrs30_spec)
  clrs_dataset_path = os.path.join(FLAGS.dataset_path, 'clrs_dataset')
  if not os.path.isdir(clrs_dataset_path):
    logging.info('Dataset not found in %s. Downloading...', clrs_dataset_path)
    download_dataset()

  common_args = dict(folder=FLAGS.dataset_path,
                     algorithm=FLAGS.algorithm,
                     batch_size=FLAGS.batch_size)
  # Make full dataset pipeline run on CPU (including prefetching).
  with tf.device('/cpu:0'):
    if FLAGS.chunked_training:
      train_sampler, spec = clrs.create_chunked_dataset(
          **common_args, split='train', chunk_length=FLAGS.chunk_length)
      train_sampler_for_eval, _ = clrs.create_dataset(
          split='train', **common_args)
      train_sampler_for_eval = train_sampler_for_eval.as_numpy_iterator()
    else:
      train_sampler, spec = clrs.create_dataset(**common_args, split='train')
      train_sampler = train_sampler.as_numpy_iterator()
      train_sampler_for_eval = None

    val_sampler, _ = clrs.create_dataset(**common_args, split='val')
    val_sampler = val_sampler.as_numpy_iterator()
    test_sampler, _ = clrs.create_dataset(**common_args, split='test')
    test_sampler = test_sampler.as_numpy_iterator()

  model_params = dict(
      hidden_dim=FLAGS.hidden_size,
      encode_hints=FLAGS.encode_hints,
      decode_hints=FLAGS.decode_hints,
      decode_diffs=FLAGS.decode_diffs,
      use_lstm=FLAGS.use_lstm,
      kind=FLAGS.processor_type,
      learning_rate=FLAGS.learning_rate,
      checkpoint_path=FLAGS.checkpoint_path,
      freeze_processor=FLAGS.freeze_processor,
      dropout_prob=FLAGS.dropout_prob,
      nb_heads=FLAGS.nb_heads,
      )

  eval_model = clrs.models.BaselineModel(
      spec=spec,
      dummy_trajectory=next(val_sampler),
      **model_params
  )
  if FLAGS.chunked_training:
    train_model = clrs.models.BaselineModelChunked(
        spec=spec,
        dummy_trajectory=next(train_sampler),
        **model_params
        )
  else:
    train_model = eval_model

  # Training loop.
  best_score = -1.0  # Ensure that there is overwriting
  rng_key = jax.random.PRNGKey(FLAGS.seed)
  current_train_items = 0
  step = 0
  next_eval = 0

  while current_train_items < FLAGS.train_items:
    feedback = next(train_sampler)

    # Initialize model.
    if current_train_items == 0:
      t = time.time()
      train_model.init(feedback.features, FLAGS.seed + 1)

    # Training step step.
    rng_key, new_rng_key = jax.random.split(rng_key)
    cur_loss = train_model.feedback(rng_key, feedback)
    rng_key = new_rng_key
    if current_train_items == 0:
      logging.info('Compiled feedback step in %f s.', time.time() - t)
    if FLAGS.chunked_training:
      examples_in_chunk = jnp.sum(feedback.features.is_last)
    else:
      examples_in_chunk = len(feedback.features.lengths)
    current_train_items += examples_in_chunk

    # Periodically evaluate model.
    if current_train_items >= next_eval:
      common_extras = {'examples_seen': current_train_items,
                       'step': step}
      eval_model.params = train_model.params
      # Training info.
      if FLAGS.chunked_training:
        train_feedback = next(train_sampler_for_eval)
      else:
        train_feedback = feedback
      rng_key, new_rng_key = jax.random.split(rng_key)
      train_stats = evaluate(
          rng_key,
          eval_model,
          train_feedback,
          extras=dict(loss=cur_loss, **common_extras),
          verbose=FLAGS.verbose_logging,
      )
      rng_key = new_rng_key
      logging.info('(train) step %d: %s', step, train_stats)

      # Validation info.
      val_feedback = next(val_sampler)
      rng_key, new_rng_key = jax.random.split(rng_key)
      val_stats = evaluate(
          rng_key, eval_model, val_feedback,
          extras=common_extras,
          verbose=FLAGS.verbose_logging)
      rng_key = new_rng_key
      logging.info('(val) step %d: %s', step, val_stats)

      # If best scores, update checkpoint.
      score = val_stats['score']
      if score > best_score:
        logging.info('Saving new checkpoint...')
        best_score = score
        train_model.save_model('best.pkl')
      next_eval += FLAGS.eval_every

    step += 1

  # Training complete, evaluate on test set.
  logging.info('Restoring best model from checkpoint...')
  eval_model.restore_model('best.pkl', only_load_processor=False)

  test_feedback = next(test_sampler)  # full-batch
  rng_key, new_rng_key = jax.random.split(rng_key)
  test_stats = evaluate(
      rng_key, eval_model, test_feedback,
      extras=dict(examples_seen=current_train_items, step=step),
      verbose=FLAGS.verbose_logging)
  rng_key = new_rng_key
  logging.info('(test) step %d: %s', step, test_stats)


if __name__ == '__main__':
  app.run(main)
