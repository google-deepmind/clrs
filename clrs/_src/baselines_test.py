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

"""Unit tests for `baselines.py`."""

import copy
import functools
from typing import Generator

from absl.testing import absltest
from absl.testing import parameterized
import chex

from clrs._src import baselines
from clrs._src import dataset
from clrs._src import processors
from clrs._src import samplers
from clrs._src import specs

import haiku as hk
import jax
import numpy as np

_Array = np.ndarray


def _error(x, y):
  return np.sum(np.abs(x-y))


def _make_sampler(algo: str, length: int) -> samplers.Sampler:
  sampler, _ = samplers.build_sampler(
      algo,
      seed=samplers.CLRS30['val']['seed'],
      num_samples=samplers.CLRS30['val']['num_samples'],
      length=length,
  )
  return sampler


def _make_iterable_sampler(
    algo: str, batch_size: int,
    length: int) -> Generator[samplers.Feedback, None, None]:
  sampler = _make_sampler(algo, length)
  while True:
    yield sampler.next(batch_size)


class BaselinesTest(parameterized.TestCase):

  def test_full_vs_chunked(self):
    """Test that chunking does not affect gradients."""

    batch_size = 4
    length = 8
    algo = 'insertion_sort'
    spec = specs.SPECS[algo]
    rng_key = jax.random.PRNGKey(42)

    full_ds = _make_iterable_sampler(algo, batch_size, length)
    chunked_ds = dataset.chunkify(
        _make_iterable_sampler(algo, batch_size, length),
        length)
    double_chunked_ds = dataset.chunkify(
        _make_iterable_sampler(algo, batch_size, length),
        length * 2)

    full_batches = [next(full_ds) for _ in range(2)]
    chunked_batches = [next(chunked_ds) for _ in range(2)]
    double_chunk_batch = next(double_chunked_ds)

    with chex.fake_jit():  # jitting makes test longer

      processor_factory = processors.get_processor_factory('mpnn', use_ln=False)
      common_args = dict(processor_factory=processor_factory, hidden_dim=8,
                         learning_rate=0.01, decode_diffs=True,
                         decode_hints=True, encode_hints=True)

      b_full = baselines.BaselineModel(
          spec, dummy_trajectory=full_batches[0], **common_args)
      b_full.init(full_batches[0].features, seed=0)
      full_params = b_full.params
      full_loss_0 = b_full.feedback(rng_key, full_batches[0])
      b_full.params = full_params
      full_loss_1 = b_full.feedback(rng_key, full_batches[1])
      new_full_params = b_full.params

      b_chunked = baselines.BaselineModelChunked(
          spec, dummy_trajectory=chunked_batches[0], **common_args)
      b_chunked.init(chunked_batches[0].features, seed=0)
      chunked_params = b_chunked.params
      jax.tree_map(np.testing.assert_array_equal,
                   full_params, chunked_params)
      chunked_loss_0 = b_chunked.feedback(rng_key, chunked_batches[0])
      b_chunked.params = chunked_params
      chunked_loss_1 = b_chunked.feedback(rng_key, chunked_batches[1])
      new_chunked_params = b_chunked.params

      b_chunked.params = chunked_params
      double_chunked_loss = b_chunked.feedback(rng_key, double_chunk_batch)

    # Test that losses match
    np.testing.assert_allclose(full_loss_0, chunked_loss_0, rtol=1e-4)
    np.testing.assert_allclose(full_loss_1, chunked_loss_1, rtol=1e-4)
    np.testing.assert_allclose(full_loss_0 + full_loss_1,
                               2 * double_chunked_loss,
                               rtol=1e-4)

    # Test that gradients are the same (parameters changed equally).
    # First check that gradients were not zero, i.e., parameters have changed.
    param_change, _ = jax.tree_flatten(
        jax.tree_map(_error, full_params, new_full_params))
    self.assertGreater(np.mean(param_change), 0.1)
    # Now check that full and chunked gradients are the same.
    jax.tree_map(functools.partial(np.testing.assert_allclose, rtol=1e-4),
                 new_full_params, new_chunked_params)

  def test_multi_vs_single(self):
    """Test that multi = single when we only train one of the algorithms."""

    batch_size = 4
    length = 16
    algos = ['insertion_sort', 'activity_selector', 'bfs']
    spec = [specs.SPECS[algo] for algo in algos]
    rng_key = jax.random.PRNGKey(42)

    full_ds = [_make_iterable_sampler(algo, batch_size, length)
               for algo in algos]
    full_batches = [next(ds) for ds in full_ds]
    full_batches_2 = [next(ds) for ds in full_ds]

    with chex.fake_jit():  # jitting makes test longer

      processor_factory = processors.get_processor_factory('mpnn', use_ln=False)
      common_args = dict(processor_factory=processor_factory, hidden_dim=8,
                         learning_rate=0.01, decode_diffs=True,
                         decode_hints=True, encode_hints=True)

      b_single = baselines.BaselineModel(
          spec[0], dummy_trajectory=full_batches[0], **common_args)
      b_multi = baselines.BaselineModel(
          spec, dummy_trajectory=full_batches, **common_args)
      b_single.init(full_batches[0].features, seed=0)
      b_multi.init([f.features for f in full_batches], seed=0)

      single_params = []
      single_losses = []
      multi_params = []
      multi_losses = []

      single_params.append(copy.deepcopy(b_single.params))
      single_losses.append(b_single.feedback(rng_key, full_batches[0]))
      single_params.append(copy.deepcopy(b_single.params))
      single_losses.append(b_single.feedback(rng_key, full_batches_2[0]))
      single_params.append(copy.deepcopy(b_single.params))

      multi_params.append(copy.deepcopy(b_multi.params))
      multi_losses.append(b_multi.feedback(rng_key, full_batches[0],
                                           algorithm_index=0))
      multi_params.append(copy.deepcopy(b_multi.params))
      multi_losses.append(b_multi.feedback(rng_key, full_batches_2[0],
                                           algorithm_index=0))
      multi_params.append(copy.deepcopy(b_multi.params))

    # Test that losses match
    np.testing.assert_array_equal(single_losses, multi_losses)
    # Test that loss decreased
    assert single_losses[1] < single_losses[0]

    # Test that param changes were the same in single and multi-algorithm
    for single, multi in zip(single_params, multi_params):
      assert hk.data_structures.is_subset(subset=single, superset=multi)
      for module_name, params in single.items():
        jax.tree_map(np.testing.assert_array_equal, params, multi[module_name])

    # Test that params change for the trained algorithm, but not the others
    for module_name, params in multi_params[0].items():
      param_changes = jax.tree_map(lambda a, b: np.sum(np.abs(a-b)),
                                   params, multi_params[1][module_name])
      param_change = sum(param_changes.values())
      if module_name in single_params[0]:  # params of trained algorithm
        assert param_change > 1e-3
      else:  # params of non-trained algorithms
        assert param_change == 0.0

  @parameterized.parameters(True, False)
  def test_multi_algorithm_idx(self, is_chunked):
    """Test that algorithm selection works as intended."""

    batch_size = 4
    length = 8
    algos = ['insertion_sort', 'activity_selector', 'bfs']
    spec = [specs.SPECS[algo] for algo in algos]
    rng_key = jax.random.PRNGKey(42)

    if is_chunked:
      ds = [dataset.chunkify(_make_iterable_sampler(algo, batch_size, length),
                             2 * length) for algo in algos]
    else:
      ds = [_make_iterable_sampler(algo, batch_size, length) for algo in algos]
    batches = [next(d) for d in ds]

    with chex.fake_jit():  # jitting makes test longer
      processor_factory = processors.get_processor_factory('mpnn', use_ln=False)
      common_args = dict(processor_factory=processor_factory, hidden_dim=8,
                         learning_rate=0.01, decode_diffs=True,
                         decode_hints=True, encode_hints=True)
      if is_chunked:
        baseline = baselines.BaselineModelChunked(
            spec, dummy_trajectory=batches, **common_args)
      else:
        baseline = baselines.BaselineModel(
            spec, dummy_trajectory=batches, **common_args)
      baseline.init([f.features for f in batches], seed=0)

      # Find out what parameters change when we train each algorithm
      def _change(x, y):
        changes = {}
        for module_name, params in x.items():
          changes[module_name] = sum(
              jax.tree_map(
                  lambda a, b: np.sum(np.abs(a-b)), params, y[module_name]
                  ).values())
        return changes

      param_changes = []
      for algo_idx in range(len(algos)):
        init_params = copy.deepcopy(baseline.params)
        _ = baseline.feedback(
            rng_key, batches[algo_idx], algorithm_index=algo_idx)
        param_changes.append(_change(init_params, baseline.params))

    # Test that non-changing parameters correspond to encoders/decoders
    # associated with the non-trained algorithms
    unchanged = [[k for k in pc if pc[k] == 0] for pc in param_changes]

    def _get_other_algos(algo_idx, modules):
      return set([k for k in modules if '_construct_encoders_decoders' in k
                  and f'algo_{algo_idx}' not in k])

    for algo_idx in range(len(algos)):
      expected_unchanged = _get_other_algos(algo_idx, baseline.params.keys())
      self.assertNotEmpty(expected_unchanged)
      self.assertSetEqual(expected_unchanged, set(unchanged[algo_idx]))


if __name__ == '__main__':
  absltest.main()
