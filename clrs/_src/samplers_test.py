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

"""Unit tests for `samplers.py`."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs
import jax
import numpy as np


class SamplersTest(parameterized.TestCase):

  @parameterized.parameters(*specs.CLRS_30_ALGS)
  def test_sampler_determinism(self, name):
    num_samples = 3
    num_nodes = 10
    sampler, _ = samplers.build_sampler(name, num_samples, num_nodes)

    np.random.seed(47)  # Set seed
    feedback = sampler.next()
    expected = feedback.outputs[0].data.copy()

    np.random.seed(48)  # Set a different seed
    feedback = sampler.next()
    actual = feedback.outputs[0].data.copy()

    # Validate that datasets are the same.
    np.testing.assert_array_equal(expected, actual)

  @parameterized.parameters(*specs.CLRS_30_ALGS)
  def test_sampler_batch_determinism(self, name):
    num_samples = 10
    batch_size = 5
    num_nodes = 10
    seed = 0
    sampler_1, _ = samplers.build_sampler(
        name, num_samples, num_nodes, seed=seed)
    sampler_2, _ = samplers.build_sampler(
        name, num_samples, num_nodes, seed=seed)

    feedback_1 = sampler_1.next(batch_size)
    feedback_2 = sampler_2.next(batch_size)

    # Validate that datasets are the same.
    jax.tree_util.tree_map(np.testing.assert_array_equal, feedback_1,
                           feedback_2)

  def test_end_to_end(self):
    num_samples = 7
    num_nodes = 3
    sampler, _ = samplers.build_sampler("dfs", num_samples, num_nodes)
    feedback = sampler.next()

    inputs = feedback.features.inputs
    self.assertLen(inputs, 4)
    self.assertEqual(inputs[0].name, "pos")
    self.assertEqual(inputs[0].data.shape, (num_samples, num_nodes))

    outputs = feedback.outputs
    self.assertLen(outputs, 1)
    self.assertEqual(outputs[0].name, "pi")
    self.assertEqual(outputs[0].data.shape, (num_samples, num_nodes))

  def test_batch_size(self):
    num_samples = 7
    num_nodes = 3
    sampler, _ = samplers.build_sampler("bfs", num_samples, num_nodes)

    # Full-batch.
    feedback = sampler.next()
    for dp in feedback.features.inputs:  # [B, ...]
      self.assertEqual(dp.data.shape[0], num_samples)

    for dp in feedback.outputs:  # [B, ...]
      self.assertEqual(dp.data.shape[0], num_samples)

    for dp in feedback.features.hints:  # [T, B, ...]
      self.assertEqual(dp.data.shape[1], num_samples)

    self.assertLen(feedback.features.lengths, num_samples)

    # Specified batch.
    batch_size = 5
    feedback = sampler.next(batch_size)

    for dp in feedback.features.inputs:  # [B, ...]
      self.assertEqual(dp.data.shape[0], batch_size)

    for dp in feedback.outputs:  # [B, ...]
      self.assertEqual(dp.data.shape[0], batch_size)

    for dp in feedback.features.hints:  # [T, B, ...]
      self.assertEqual(dp.data.shape[1], batch_size)

    self.assertLen(feedback.features.lengths, batch_size)

  def test_batch_io(self):
    sample = [
        probing.DataPoint(
            name="x",
            location=specs.Location.NODE,
            type_=specs.Type.SCALAR,
            data=np.zeros([1, 3]),
        ),
        probing.DataPoint(
            name="y",
            location=specs.Location.EDGE,
            type_=specs.Type.MASK,
            data=np.zeros([1, 3, 3]),
        ),
    ]

    trajectory = [sample.copy(), sample.copy(), sample.copy(), sample.copy()]
    batched = samplers._batch_io(trajectory)

    np.testing.assert_array_equal(batched[0].data, np.zeros([4, 3]))
    np.testing.assert_array_equal(batched[1].data, np.zeros([4, 3, 3]))

  def test_batch_hint(self):
    sample0 = [
        probing.DataPoint(
            name="x",
            location=specs.Location.NODE,
            type_=specs.Type.MASK,
            data=np.zeros([2, 1, 3]),
        ),
        probing.DataPoint(
            name="y",
            location=specs.Location.NODE,
            type_=specs.Type.POINTER,
            data=np.zeros([2, 1, 3]),
        ),
    ]

    sample1 = [
        probing.DataPoint(
            name="x",
            location=specs.Location.NODE,
            type_=specs.Type.MASK,
            data=np.zeros([1, 1, 3]),
        ),
        probing.DataPoint(
            name="y",
            location=specs.Location.NODE,
            type_=specs.Type.POINTER,
            data=np.zeros([1, 1, 3]),
        ),
    ]

    trajectory = [sample0, sample1]
    batched, lengths = samplers._batch_hints(trajectory, 0)

    np.testing.assert_array_equal(batched[0].data, np.zeros([2, 2, 3]))
    np.testing.assert_array_equal(batched[1].data, np.zeros([2, 2, 3]))
    np.testing.assert_array_equal(lengths, np.array([2, 1]))

    batched, lengths = samplers._batch_hints(trajectory, 5)

    np.testing.assert_array_equal(batched[0].data, np.zeros([5, 2, 3]))
    np.testing.assert_array_equal(batched[1].data, np.zeros([5, 2, 3]))
    np.testing.assert_array_equal(lengths, np.array([2, 1]))

  def test_padding(self):
    lens = np.random.choice(10, (10,), replace=True) + 1
    trajectory = []
    for len_ in lens:
      trajectory.append([
          probing.DataPoint(
              name="x",
              location=specs.Location.NODE,
              type_=specs.Type.MASK,
              data=np.ones([len_, 1, 3]),
          )
      ])

    batched, lengths = samplers._batch_hints(trajectory, 0)
    np.testing.assert_array_equal(lengths, lens)

    for i in range(len(lens)):
      ones = batched[0].data[:lens[i], i, :]
      zeros = batched[0].data[lens[i]:, i, :]
      np.testing.assert_array_equal(ones, np.ones_like(ones))
      np.testing.assert_array_equal(zeros, np.zeros_like(zeros))


class ProcessRandomPosTest(parameterized.TestCase):

  @parameterized.parameters(["insertion_sort", "naive_string_matcher"])
  def test_random_pos(self, algorithm_name):
    batch_size, length = 12, 10
    def _make_sampler():
      sampler, _ = samplers.build_sampler(
          algorithm_name,
          seed=0,
          num_samples=100,
          length=length,
          )
      while True:
        yield sampler.next(batch_size)
    sampler_1 = _make_sampler()
    sampler_2 = _make_sampler()
    sampler_2 = samplers.process_random_pos(sampler_2, np.random.RandomState(0))

    batch_without_rand_pos = next(sampler_1)
    batch_with_rand_pos = next(sampler_2)
    pos_idx = [x.name for x in batch_without_rand_pos.features.inputs].index(
        "pos")
    fixed_pos = batch_without_rand_pos.features.inputs[pos_idx]
    rand_pos = batch_with_rand_pos.features.inputs[pos_idx]
    self.assertEqual(rand_pos.location, specs.Location.NODE)
    self.assertEqual(rand_pos.type_, specs.Type.SCALAR)
    self.assertEqual(rand_pos.data.shape, (batch_size, length))
    self.assertEqual(rand_pos.data.shape, fixed_pos.data.shape)
    self.assertEqual(rand_pos.type_, fixed_pos.type_)
    self.assertEqual(rand_pos.location, fixed_pos.location)

    assert (rand_pos.data.std(axis=0) > 1e-3).all()
    assert (fixed_pos.data.std(axis=0) < 1e-9).all()
    if "string" in algorithm_name:
      expected = np.concatenate([np.arange(4*length//5)/(4*length//5),
                                 np.arange(length//5)/(length//5)])
    else:
      expected = np.arange(length)/length
    np.testing.assert_array_equal(
        fixed_pos.data, np.broadcast_to(expected, (batch_size, length)))

    batch_with_rand_pos.features.inputs[pos_idx] = fixed_pos
    chex.assert_trees_all_equal(batch_with_rand_pos, batch_without_rand_pos)


if __name__ == "__main__":
  absltest.main()
