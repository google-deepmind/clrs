# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for clrs.src_.clrs_text.clrs_utils."""


from absl.testing import absltest
from absl.testing import parameterized
import clrs
from clrs._src import probing
from clrs._src.clrs_text import clrs_utils
import numpy as np


class TestFormatCLRSExamples(parameterized.TestCase):

  @parameterized.product(
      algo_name=list(clrs.CLRS_30_ALGS_SETTINGS.keys()),
      use_hints=[True, False],
  )
  def test_format(self, algo_name, use_hints):
    """Test that we can format samples from any algo into strings."""
    sampler, _ = clrs.build_sampler(
        algo_name,
        seed=0,
        num_samples=-1,
        length=16,
        track_max_steps=False,
        use_padding=False,
    )

    for _ in range(100):
      sample = sampler.next(batch_size=1)

      question, answer = clrs_utils.format_clrs_example(
          algo_name,
          sample,
          use_hints=use_hints,
      )

      self.assertTrue(question.startswith(f'{algo_name}:\n'))
      self.assertTrue(question.endswith(':\n'))
      self.assertTrue(answer.endswith('\n\n'))

      if use_hints and algo_name in clrs_utils.CLRS_TASKS_WITH_HINTS:
        self.assertIn('trace | ', question)
        self.assertIn('initial_trace:', question)
      else:
        self.assertNotIn('trace | ', question)
        self.assertNotIn('initial_trace:', question)


class TestPredecessorToOrder(parameterized.TestCase):
  def test_predecessor_to_order(self):
    """Test that `predecessor_to_order` matches the slower clrs conversion."""
    for i in range(20):
      length = np.random.randint(4, 16)
      sampler, unused_spec = clrs.build_sampler(
          'insertion_sort',
          seed=i,
          num_samples=-1,
          length=length,
          track_max_steps=False,
      )
      x = sampler.next(batch_size=1)
      pred = x.outputs[0].data[0]
      expected_order = probing.predecessor_pointers_to_permutation_matrix(
          pred
      ) @ np.arange(pred.shape[0])
      order = clrs_utils.predecessors_to_order(pred)
      np.testing.assert_array_equal(expected_order, order)


if __name__ == '__main__':
  absltest.main()
