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

import functools
from typing import Generator

from absl.testing import absltest
import chex

from clrs._src import baselines
from clrs._src import dataset
from clrs._src import samplers
from clrs._src import specs

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


class BaselinesTest(absltest.TestCase):

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

      common_args = dict(hidden_dim=8,
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
      jax.tree_multimap(np.testing.assert_array_equal,
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
    param_change, _ = jax.tree_flatten(jax.tree_multimap(
        _error, full_params, new_full_params))
    self.assertGreater(np.mean(param_change), 0.1)
    # Now check that full and chunked gradients are the same.
    jax.tree_multimap(functools.partial(np.testing.assert_allclose, rtol=1e-4),
                      new_full_params, new_chunked_params)


if __name__ == '__main__':
  absltest.main()
