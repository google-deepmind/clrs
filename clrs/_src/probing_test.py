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

"""Unit tests for `probing.py`."""

from absl.testing import absltest

from clrs._src import probing
import numpy as np


# pylint: disable=invalid-name


class ProbingTest(absltest.TestCase):

  def test_array(self):
    A_pos = np.array([1, 2, 0, 4, 3])
    expected = np.array([2, 1, 1, 4, 0])
    out = probing.array(A_pos)
    np.testing.assert_array_equal(expected, out)

  def test_array_cat(self):
    A = np.array([2, 1, 0, 1, 1])
    expected = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0]
    ])
    out = probing.array_cat(A, 3)
    np.testing.assert_array_equal(expected, out)

  def test_heap(self):
    A_pos = np.array([1, 3, 5, 0, 7, 4, 2, 6])
    expected = np.array([3, 1, 2, 1, 5, 1, 6, 3])
    out = probing.heap(A_pos, heap_size=6)
    np.testing.assert_array_equal(expected, out)

  def test_graph(self):
    G = np.array([
        [0.0, 7.0, -1.0, -3.9, 7.452],
        [0.0, 0.0, 133.0, 0.0, 9.3],
        [0.5, 0.1, 0.22, 0.55, 0.666],
        [7.0, 6.1, 0.2, 0.0, 0.0],
        [0.0, 3.0, 0.0, 1.0, 0.5]
    ])
    expected = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 1.0]
    ])
    out = probing.graph(G)
    np.testing.assert_array_equal(expected, out)

  def test_mask_one(self):
    expected = np.array([0, 0, 0, 1, 0])
    out = probing.mask_one(3, 5)
    np.testing.assert_array_equal(expected, out)

  def test_strings_id(self):
    T_pos = np.array([0, 1, 2, 3, 4])
    P_pos = np.array([0, 1, 2])
    expected = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    out = probing.strings_id(T_pos, P_pos)
    np.testing.assert_array_equal(expected, out)

  def test_strings_pair(self):
    pair_probe = np.array([
        [0.5, 3.1, 9.1, 7.3],
        [1.0, 0.0, 8.0, 9.3],
        [0.1, 5.0, 0.0, 1.2]
    ])
    expected = np.array([
        [0.0, 0.0, 0.0, 0.5, 3.1, 9.1, 7.3],
        [0.0, 0.0, 0.0, 1.0, 0.0, 8.0, 9.3],
        [0.0, 0.0, 0.0, 0.1, 5.0, 0.0, 1.2],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    out = probing.strings_pair(pair_probe)
    np.testing.assert_equal(expected, out)

  def test_strings_pair_cat(self):
    pair_probe = np.array([
        [0, 2, 1],
        [2, 2, 0]
    ])
    expected = np.array([
        [
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ],
        [
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
        ],
        [
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
        ],
        [
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
        ],
        [
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
            [0, 0, 0, -1],
        ],
    ])
    out = probing.strings_pair_cat(pair_probe, 3)
    np.testing.assert_equal(expected, out)

  def test_strings_pi(self):
    T_pos = np.array([0, 1, 2, 3, 4, 5])
    P_pos = np.array([0, 1, 2, 3])
    pi = np.array([3, 1, 0, 2])
    expected = np.array(
        [0, 1, 2, 3, 4, 5, 9, 7, 6, 8]
    )
    out = probing.strings_pi(T_pos, P_pos, pi)
    np.testing.assert_array_equal(expected, out)

  def test_strings_pos(self):
    T_pos = np.array([0, 1, 2, 3, 4])
    P_pos = np.array([0, 1, 2, 3])
    expected = np.array(
        [0.0, 0.2, 0.4, 0.6, 0.8,
         0.0, 0.25, 0.5, 0.75]
    )
    out = probing.strings_pos(T_pos, P_pos)
    np.testing.assert_array_equal(expected, out)

  def test_strings_pred(self):
    T_pos = np.array([0, 1, 2, 3, 4])
    P_pos = np.array([0, 1, 2])
    expected = np.array([0, 0, 1, 2, 3, 5, 5, 6])
    out = probing.strings_pred(T_pos, P_pos)
    np.testing.assert_array_equal(expected, out)

if __name__ == "__main__":
  absltest.main()
