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
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        feedback_1,
        feedback_2,
    )

  @parameterized.product(
      algo_name=specs.CLRS_30_ALGS,
      track_max_steps=[True, False],
  )
  def test_sampler_rounding(self, algo_name, track_max_steps):
    num_samples = 1
    batch_size = 1
    num_nodes = 10
    truncate_decimals = 5
    seed = 0
    sampler_1, _ = samplers.build_sampler(
        algo_name,
        num_samples,
        num_nodes,
        seed=seed,
        track_max_steps=track_max_steps,
        truncate_decimals=None,
    )
    sampler_2, _ = samplers.build_sampler(
        algo_name,
        num_samples,
        num_nodes,
        seed=seed,
        track_max_steps=track_max_steps,
        truncate_decimals=truncate_decimals,
    )

    sample_1 = sampler_1.next(batch_size)
    sample_2 = sampler_2.next(batch_size)

    jax.tree_util.tree_map(
        lambda *args: np.testing.assert_array_almost_equal(
            *args, decimal=truncate_decimals - 2
        ),
        sample_1,
        sample_2,
    )

    def check_for_close_equality():
      jax.tree_util.tree_map(
          lambda *args: np.testing.assert_array_almost_equal(
              *args, decimal=truncate_decimals + 1
          ),
          sample_1,
          sample_2,
      )
      jax.tree_util.tree_map(
          np.testing.assert_array_equal,
          sample_1,
          sample_2,
      )

    if sampler_1.CAN_TRUNCATE_INPUT_DATA:
      with self.assertRaises(AssertionError):
        check_for_close_equality()
    else:
      check_for_close_equality()

  def test_end_to_end(self):
    num_samples = 7
    num_nodes = 3
    sampler, _ = samplers.build_sampler('bfs', num_samples, num_nodes)
    feedback = sampler.next()

    inputs = feedback.features.inputs
    self.assertLen(inputs, 4)
    self.assertEqual(inputs[0].name, 'pos')
    self.assertEqual(inputs[0].data.shape, (num_samples, num_nodes))

    outputs = feedback.outputs
    self.assertLen(outputs, 1)
    self.assertEqual(outputs[0].name, 'pi')
    self.assertEqual(outputs[0].data.shape, (num_samples, num_nodes))

  def test_batch_size(self):
    num_samples = 7
    num_nodes = 3
    sampler, _ = samplers.build_sampler('bfs', num_samples, num_nodes)

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
            name='x',
            location=specs.Location.NODE,
            type_=specs.Type.SCALAR,
            data=np.zeros([1, 3]),
        ),
        probing.DataPoint(
            name='y',
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
    sample_0 = [
        probing.DataPoint(
            name='x',
            location=specs.Location.NODE,
            type_=specs.Type.MASK,
            data=np.zeros([2, 1, 3]),
        ),
        probing.DataPoint(
            name='y',
            location=specs.Location.NODE,
            type_=specs.Type.POINTER,
            data=np.zeros([2, 1, 3]),
        ),
    ]

    sample_1 = [
        probing.DataPoint(
            name='x',
            location=specs.Location.NODE,
            type_=specs.Type.MASK,
            data=np.zeros([1, 1, 3]),
        ),
        probing.DataPoint(
            name='y',
            location=specs.Location.NODE,
            type_=specs.Type.POINTER,
            data=np.zeros([1, 1, 3]),
        ),
    ]

    trajectory = [sample_0, sample_1]
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
              name='x',
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


class AuxiliaryFunctionsTest(parameterized.TestCase):
  """Unit tests for auxiliary functions in `samplers.py`."""

  @parameterized.named_parameters(
      dict(
          testcase_name='single_float_4_decimals_array',
          input_data=[np.array([1.123456789])],
          expected_output=[np.array([1.1234])],
          truncate_decimals=4,
      ),
      dict(
          testcase_name='single_float_6_decimals_array',
          input_data=[np.array([1.123456789])],
          expected_output=[np.array([1.123456])],
          truncate_decimals=6,
      ),
      dict(
          testcase_name='single_float_4_decimals',
          input_data=[[1.123456789]],
          expected_output=[[1.123456789]],
          truncate_decimals=4,
      ),
      dict(
          testcase_name='single_float_6_decimals',
          input_data=[[1.123456789]],
          expected_output=[[1.123456789]],
          truncate_decimals=6,
      ),
      dict(
          testcase_name='two_floats_4_decimals',
          input_data=[1.123456789, 2.123156789],
          expected_output=[1.1234, 2.1231],
          truncate_decimals=4,
      ),
      dict(
          testcase_name='list_of_floats_4_decimals',
          input_data=[[1.123456789, 2.123156789]],
          expected_output=[[1.123456789, 2.123156789]],
          truncate_decimals=4,
      ),
      dict(
          testcase_name='list_of_floats_4_decimals_array',
          input_data=[np.array([1.123456789, 2.123156789])],
          expected_output=[np.array([1.1234, 2.1231])],
          truncate_decimals=4,
      ),
      dict(
          testcase_name='integers_4_decimals',
          input_data=[[1, 2]],
          expected_output=[[1, 2]],
          truncate_decimals=4,
      ),
      dict(
          testcase_name='integers_4_decimals_np_array',
          input_data=[np.array([1, 2])],
          expected_output=[np.array([1, 2])],
          truncate_decimals=4,
      ),
      dict(
          testcase_name='mixed_4_decimals',
          input_data=np.array([1.0, 2.0, 3.14159]),
          expected_output=np.array([1.0, 2.0, 3.1415]),
          truncate_decimals=4,
      ),
      dict(
          testcase_name='list_mixed_4_decimals',
          input_data=[np.array([1.0, 2.0, 3.14159])],
          expected_output=[np.array([1.0, 2.0, 3.1415])],
          truncate_decimals=4,
      ),
      dict(
          testcase_name='with_strings_4_decimals_np_array',
          input_data=np.array([1.0, 2.0, 3.14159, 'test']),
          expected_output=np.array([1.0, 2.0, 3.14159, 'test']),
          truncate_decimals=4,
      ),
      dict(
          testcase_name='mixed_data',
          input_data=[np.array([1.0, 2.0, 3.14159]), 'test'],
          expected_output=[np.array([1.0, 2.0, 3.1415]), 'test'],
          truncate_decimals=4,
      ),
  )
  def test_trunc_array(self, input_data, expected_output, truncate_decimals):
    sampler, _ = samplers.build_sampler(
        'insertion_sort',
        1,
        16,
        seed=0,
        truncate_decimals=truncate_decimals,
    )

    actual = sampler._trunc_array(input_data)
    chex.assert_trees_all_equal(actual, expected_output)

  @parameterized.named_parameters(
      dict(
          testcase_name='single_float',
          input_data=1.123456789,
          expected_output=False,
      ),
      dict(
          testcase_name='single_int',
          input_data=1,
          expected_output=False,
      ),
      dict(
          testcase_name='single_float_list',
          input_data=[1.123456789],
          expected_output=False,
      ),
      dict(
          testcase_name='single_int_list',
          input_data=[1],
          expected_output=False,
      ),
      dict(
          testcase_name='single_float_array',
          input_data=np.array(1.123456789),
          expected_output=True,
      ),
      dict(
          testcase_name='single_int_array',
          input_data=np.array(1),
          expected_output=False,
      ),
      dict(
          testcase_name='single_float_in_list_array',
          input_data=[np.array([1.123456789])],
          expected_output=False,
      ),
      dict(
          testcase_name='single_int_in_list_array',
          input_data=[np.array([1])],
          expected_output=False,
      ),
      dict(
          testcase_name='multiple_float_list',
          input_data=[1.123456789, 12.342453],
          expected_output=False,
      ),
      dict(
          testcase_name='multiple_ints_list',
          input_data=[1, 2],
          expected_output=False,
      ),
      dict(
          testcase_name='multiple_mixed_list',
          input_data=[1, 2.0],
          expected_output=False,
      ),
      dict(
          testcase_name='multiple_float_list_array',
          input_data=np.array([1.123456789, 12.342453]),
          expected_output=True,
      ),
      dict(
          testcase_name='multiple_ints_list_array',
          input_data=np.array([1, 2]),
          expected_output=False,
      ),
      dict(
          testcase_name='multiple_mixed_list_array',
          input_data=np.array([1, 2.0]),
          expected_output=True,
      ),
      dict(
          testcase_name='multiple_float_in_list_array',
          input_data=[np.array([1.123456789, 12.342453])],
          expected_output=False,
      ),
      dict(
          testcase_name='multiple_mixed_in_list_array',
          input_data=[np.array([1, 2.0])],
          expected_output=False,
      ),
      dict(
          testcase_name='multiple_int_in_list_array',
          input_data=[np.array([1, 2])],
          expected_output=False,
      ),
      dict(
          testcase_name='list_of_multiple_mixed_in_list_array',
          input_data=[np.array([1, 2.0])],
          expected_output=False,
      ),
      dict(
          testcase_name='string',
          input_data='test',
          expected_output=False,
      ),
      dict(
          testcase_name='empty_list',
          input_data=[],
          expected_output=False,
      ),
      dict(
          testcase_name='empty_tuple',
          input_data=(),
          expected_output=False,
      ),
      dict(
          testcase_name='none',
          input_data=None,
          expected_output=False,
      ),
  )
  def test_is_float_array(self, input_data, expected_output):
    self.assertEqual(samplers._is_float_array(input_data), expected_output)

  @parameterized.named_parameters(
      dict(
          testcase_name='collinear_points',
          point_1=np.array([1, 1]),
          point_2=np.array([2, 2]),
          point_3=np.array([3, 3]),
          eps=1e-6,
          expected_output=True,
      ),
      dict(
          testcase_name='non_collinear_points',
          point_1=np.array([1, 1]),
          point_2=np.array([2, 2]),
          point_3=np.array([1, 2]),
          eps=1e-6,
          expected_output=False,
      ),
      dict(
          testcase_name='points_within_tolerance',
          point_1=np.array([1, 1]),
          point_2=np.array([2, 2]),
          point_3=np.array([2.00, 1.9999]),
          eps=1e-4,
          expected_output=True,
      ),
      dict(
          testcase_name='points_outside_tolerance',
          point_1=np.array([1, 1]),
          point_2=np.array([2, 2]),
          point_3=np.array([2, 1.9998]),
          eps=1e-4,
          expected_output=False,
      ),
  )
  def test_is_collinear(self, point_1, point_2, point_3, eps, expected_output):
    self.assertEqual(
        samplers._is_collinear(point_1, point_2, point_3, eps), expected_output
    )

  def test_is_collinear_raise_error(self):
    with self.assertRaises(ValueError):
      samplers._is_collinear(
          np.array([1, 1, 3]), np.array([2,]), np.array([3, 3]), eps=1e-6,
      )


class ProcessRandomPosTest(parameterized.TestCase):

  @parameterized.parameters(['insertion_sort', 'naive_string_matcher'])
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
        'pos',
    )
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
    if 'string' in algorithm_name:
      expected = np.concatenate([np.arange(4*length//5)/(4*length//5),
                                 np.arange(length//5)/(length//5)])
    else:
      expected = np.arange(length)/length
    np.testing.assert_array_equal(
        fixed_pos.data, np.broadcast_to(expected, (batch_size, length)))

    batch_with_rand_pos.features.inputs[pos_idx] = fixed_pos
    chex.assert_trees_all_equal(batch_with_rand_pos, batch_without_rand_pos)


if __name__ == '__main__':
  absltest.main()
