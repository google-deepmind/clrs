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

"""Unit tests for `dynamic_programming.py`."""

from absl.testing import absltest

from clrs._src.algorithms import dynamic_programming
import numpy as np


class DynamicProgrammingTest(absltest.TestCase):

  def test_matrix_chain_order_1(self):

    expected = np.array([
        [0, 1, 1, 3, 3, 3],
        [0, 0, 2, 3, 3, 3],
        [0, 0, 0, 3, 3, 3],
        [0, 0, 0, 0, 4, 5],
        [0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0],
    ])
    for shift in [0, 1, 2]:
      for scale in [1, 3, 17]:
        ps = shift + scale * np.array([30, 35, 15, 5, 10, 20, 25])
        order, _ = dynamic_programming.matrix_chain_order(ps)
        np.testing.assert_array_equal(expected, order)

  def test_matrix_chain_order_2(self):

    expected = np.array([
        [0, 1, 2, 2, 4, 2],
        [0, 0, 2, 2, 2, 2],
        [0, 0, 0, 3, 4, 4],
        [0, 0, 0, 0, 4, 4],
        [0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0],
    ])

    for shift in [0, 1]:
      for scale in [1, 3, 17]:
        ps = shift + scale * np.array([5, 10, 3, 12, 5, 50, 6])
        order, _ = dynamic_programming.matrix_chain_order(ps)
        np.testing.assert_array_equal(expected, order)

  def test_lcs_length(self):
    xs = np.array([0, 1, 2, 1, 3, 0, 1])
    ys = np.array([1, 3, 2, 0, 1, 0])

    expected = np.array([
        [1, 1, 1, 0, 2, 0],
        [0, 2, 2, 1, 0, 2],
        [1, 1, 0, 2, 1, 1],
        [0, 1, 1, 1, 0, 2],
        [1, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 1],
    ])
    out, _ = dynamic_programming.lcs_length(xs, ys)
    np.testing.assert_array_equal(expected, out)

  def test_optimal_bst(self):
    p = np.array([0.15, 0.10, 0.05, 0.10, 0.2])
    q = np.array([0.05, 0.10, 0.05, 0.05, 0.05, 0.10])
    assert p.sum() + q.sum() == 1.

    expected = np.array([
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 3],
        [0, 0, 0, 2, 3, 4],
        [0, 0, 0, 0, 3, 4],
        [0, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0, 0],
    ])

    out, _ = dynamic_programming.optimal_bst(p, q)
    np.testing.assert_array_equal(expected, out)


if __name__ == "__main__":
  absltest.main()
