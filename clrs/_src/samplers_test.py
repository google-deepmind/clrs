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

from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs
import numpy as np


class SamplersTest(parameterized.TestCase):

  @parameterized.parameters(*specs.CLRS_21_ALGS)
  def test_sampler_determinism(self, name):
    sampler, _ = samplers.clrs21_val(name)

    np.random.seed(47)  # Set seed
    feedback = sampler.next()
    expected = feedback.outputs[0].data.copy()

    np.random.seed(48)  # Set a different seed
    feedback = sampler.next()
    actual = feedback.outputs[0].data.copy()

    # Validate that datasets are the same.
    np.testing.assert_array_equal(expected, actual)

  def test_end_to_end(self):
    num_samples = 7
    num_nodes = 3
    sampler, _ = samplers.build_sampler("bfs", num_samples, num_nodes)
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
    batched, lengths = samplers._batch_hints(trajectory)

    np.testing.assert_array_equal(batched[0].data, np.zeros([2, 2, 3]))
    np.testing.assert_array_equal(batched[1].data, np.zeros([2, 2, 3]))
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

    batched, lengths = samplers._batch_hints(trajectory)
    np.testing.assert_array_equal(lengths, lens)

    for i in range(len(lens)):
      ones = batched[0].data[:lens[i], i, :]
      zeros = batched[0].data[lens[i]:, i, :]
      np.testing.assert_array_equal(ones, np.ones_like(ones))
      np.testing.assert_array_equal(zeros, np.zeros_like(zeros))


if __name__ == "__main__":
  absltest.main()
