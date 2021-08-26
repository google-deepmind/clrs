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

"""Unit tests for `greedy.py`."""

from absl.testing import absltest

from clrs._src.algorithms import greedy
import numpy as np


class GreedyTest(absltest.TestCase):

  def test_greedy_activity_selector(self):
    s = np.array([1, 3, 0, 5, 3, 5, 6, 8, 8, 2, 12])
    f = np.array([4, 5, 6, 7, 9, 9, 10, 11, 12, 14, 16])
    expected = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
    out, _ = greedy.activity_selector(s, f)
    np.testing.assert_array_equal(expected, out)

  def test_task_scheduling(self):
    d = np.array([4, 2, 4, 3, 1, 4, 6])
    w = np.array([70, 60, 50, 40, 30, 20, 10])
    expected = np.array([1, 1, 1, 0, 0, 1, 1])
    out, _ = greedy.task_scheduling(d, w)
    np.testing.assert_array_equal(expected, out)


if __name__ == "__main__":
  absltest.main()
