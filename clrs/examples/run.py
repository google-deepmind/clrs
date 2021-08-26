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

import time
from absl import app
from absl import flags
from absl import logging

import clrs


flags.DEFINE_string('algorithm', 'bfs', 'Which algorithm to run.')
flags.DEFINE_integer('seed', 42, 'Random seed to set')

flags.DEFINE_integer('batch_size', 32, 'Batch size used for training.')
flags.DEFINE_integer('train_steps', 5000, 'Number of training steps.')
flags.DEFINE_integer('log_every', 10, 'Logging frequency.')
flags.DEFINE_boolean('verbose_logging', False, 'Whether to log aux losses.')

flags.DEFINE_integer('hidden_size', 128,
                     'Number of hidden size units of the model.')
flags.DEFINE_float('learning_rate', 0.003, 'Learning rate to use.')

flags.DEFINE_boolean('encode_hints', True,
                     'Whether to provide hints as model inputs.')
flags.DEFINE_boolean('decode_hints', True,
                     'Whether to provide hints as model outputs.')
flags.DEFINE_boolean('decode_diffs', True,
                     'Whether to predict masks within the model.')
flags.DEFINE_enum('processor_type', 'mpnn',
                  ['deepsets', 'mpnn', 'pgn', 'gat'],
                  'Whether to predict masks within the model.')

flags.DEFINE_string('checkpoint_path', '/tmp/clrs3',
                    'Path in which checkpoints are saved.')
flags.DEFINE_boolean('freeze_processor', False,
                     'Whether to freeze the processor of the model.')

FLAGS = flags.FLAGS


def main(unused_argv):
  # Use canonical CLRS-21 samplers.
  clrs21_spec = clrs.CLRS21
  logging.info('Using CLRS21 spec: %s', clrs21_spec)
  train_sampler, spec = clrs.clrs21_train(FLAGS.algorithm)
  val_sampler, _ = clrs.clrs21_val(FLAGS.algorithm)

  model = clrs.models.BaselineModel(
      spec=spec,
      hidden_dim=FLAGS.hidden_size,
      encode_hints=FLAGS.encode_hints,
      decode_hints=FLAGS.decode_hints,
      decode_diffs=FLAGS.decode_diffs,
      kind=FLAGS.processor_type,
      learning_rate=FLAGS.learning_rate,
      checkpoint_path=FLAGS.checkpoint_path,
      freeze_processor=FLAGS.freeze_processor,
      dummy_trajectory=train_sampler.next(FLAGS.batch_size),
  )

  def evaluate(step, model, feedback, extras=None, verbose=False):
    """Evaluates a model on feedback."""
    examples_per_step = len(feedback.features.lengths)
    out = {'step': step, 'examples_seen': step * examples_per_step}
    predictions, aux = model.predict(feedback.features)
    out.update(clrs.evaluate(feedback, predictions))
    if extras:
      out.update(extras)
    if verbose:
      out.update(model.verbose_loss(feedback, aux))

    def unpack(v):
      try:
        return v.item()  # DeviceArray
      except AttributeError:
        return v
    return {k: unpack(v) for k, v in out.items()}

  # Training loop.
  best_score = 0.

  for step in range(FLAGS.train_steps):
    feedback = train_sampler.next(FLAGS.batch_size)

    # Initialize model.
    if step == 0:
      t = time.time()
      model.init(feedback.features, FLAGS.seed)

    # Training step step.
    cur_loss = model.feedback(feedback)
    if step == 0:
      logging.info('Compiled feedback step in %f s.', time.time() - t)

    # Periodically evaluate model.
    if step % FLAGS.log_every == 0:
      # Training info.
      train_stats = evaluate(
          step,
          model,
          feedback,
          extras={'loss': cur_loss},
          verbose=FLAGS.verbose_logging,
      )
      logging.info('(train) step %d: %s', step, train_stats)

      # Validation info.
      val_feedback = val_sampler.next()  # full-batch
      val_stats = evaluate(
          step, model, val_feedback, verbose=FLAGS.verbose_logging)
      logging.info('(val) step %d: %s', step, val_stats)

      # If best scores, update checkpoint.
      score = val_stats['score']
      if score > best_score:
        logging.info('Saving new checkpoint...')
        best_score = score
        model.save_model('best.pkl')

  # Training complete, evaluate on test set.
  test_sampler, _ = clrs.clrs21_test(FLAGS.algorithm)
  logging.info('Restoring best model from checkpoint...')
  model.restore_model('best.pkl', only_load_processor=False)

  test_feedback = test_sampler.next()  # full-batch
  test_stats = evaluate(
      step, model, test_feedback, verbose=FLAGS.verbose_logging)
  logging.info('(test) step %d: %s', step, test_stats)


if __name__ == '__main__':
  app.run(main)
