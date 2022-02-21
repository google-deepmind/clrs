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

"""Unit tests for `dataset.py`."""

from typing import Generator, List

from absl.testing import absltest
from absl.testing import parameterized

from clrs._src import dataset
from clrs._src import samplers
from clrs._src import specs
import numpy as np

_Array = np.ndarray


def _stack_to_shortest(x: List[_Array]) -> _Array:
  min_len = min(map(len, x))
  return np.array([a[:min_len] for a in x])


def _make_sampler(algo: str) -> samplers.Sampler:
  sampler, _ = samplers.build_sampler(
      algo,
      seed=samplers.CLRS30['val']['seed'],
      num_samples=samplers.CLRS30['val']['num_samples'],
      length=samplers.CLRS30['val']['length'],
  )
  return sampler


def _make_iterable_sampler(
    algo: str, batch_size: int) -> Generator[samplers.Feedback, None, None]:
  sampler = _make_sampler(algo)
  while True:
    yield sampler.next(batch_size)


class DatasetTest(parameterized.TestCase):

  @parameterized.product(
      name=specs.CLRS_30_ALGS[:5],
      chunk_length=[20, 50])
  def test_chunkify(self, name: str, chunk_length: int):
    """Test that samples are concatenated and split in chunks correctly."""
    batch_size = 8

    ds = _make_iterable_sampler(name, batch_size)
    chunked_ds = dataset.chunkify(
        _make_iterable_sampler(name, batch_size),
        chunk_length)

    samples = [next(ds) for _ in range(20)]
    cum_lengths = np.cumsum([s.features.lengths for s in samples], axis=0)
    n_chunks = np.amax(cum_lengths[-1]).astype(int) // chunk_length + 1
    chunks = [next(chunked_ds) for _ in range(n_chunks)]

    # Check correctness of `is_first` and `is_last` markers
    start_idx = _stack_to_shortest([np.where(x)[0] for x in np.concatenate(
        [c.features.is_first for c in chunks]).T]).T
    end_idx = _stack_to_shortest([np.where(x)[0] for x in np.concatenate(
        [c.features.is_last for c in chunks]).T]).T
    assert len(start_idx) >= len(cum_lengths)
    start_idx = start_idx[:len(cum_lengths)]
    assert len(end_idx) >= len(cum_lengths)
    end_idx = end_idx[:len(cum_lengths)]

    np.testing.assert_equal(start_idx[0], 0)
    np.testing.assert_array_equal(cum_lengths - 1, end_idx)
    np.testing.assert_array_equal(cum_lengths[:-1], start_idx[1:])

    # Check that inputs, outputs and hints have been copied correctly
    all_input = np.concatenate([c.features.inputs[0].data for c in chunks])
    all_output = np.concatenate([c.outputs[0].data for c in chunks])
    all_hint = np.concatenate([c.features.hints[0].data for c in chunks])
    for i in range(batch_size):
      length0 = int(samples[0].features.lengths[i])
      length1 = int(samples[1].features.lengths[i])
      # Check first sample
      np.testing.assert_array_equal(
          all_input[:length0, i],
          np.tile(samples[0].features.inputs[0].data[i], [length0, 1]))
      np.testing.assert_array_equal(
          all_output[:length0, i],
          np.tile(samples[0].outputs[0].data[i], [length0, 1]))
      np.testing.assert_array_equal(
          all_hint[:length0, i],
          samples[0].features.hints[0].data[:length0, i])
      # Check second sample
      np.testing.assert_array_equal(
          all_input[length0:length0 + length1, i],
          np.tile(samples[1].features.inputs[0].data[i], [length1, 1]))
      np.testing.assert_array_equal(
          all_output[length0:length0 + length1, i],
          np.tile(samples[1].outputs[0].data[i], [length1, 1]))
      np.testing.assert_array_equal(
          all_hint[length0:length0 + length1, i],
          samples[1].features.hints[0].data[:length1, i])


if __name__ == '__main__':
  absltest.main()
