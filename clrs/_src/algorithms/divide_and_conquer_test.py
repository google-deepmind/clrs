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

"""Unit tests for `divide_and_conquer.py`."""
# pylint: disable=invalid-name

from absl.testing import absltest
from absl.testing import parameterized

from clrs._src.algorithms import divide_and_conquer
import numpy as np


class DivideAndConquerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("Maximum subarray", divide_and_conquer.find_maximum_subarray),
      ("Kadane's variant", divide_and_conquer.find_maximum_subarray_kadane),
  )
  def test_find_maximum_subarray_pos(self, algorithm):
    A = np.random.randint(0, 100, size=(13,))
    (low, high, sum_), _ = algorithm(A)
    self.assertEqual(low, 0)
    self.assertEqual(high, len(A) - 1)
    self.assertEqual(sum_, np.sum(A))

  @parameterized.named_parameters(
      ("Maximum subarray", divide_and_conquer.find_maximum_subarray),
      ("Kadane's variant", divide_and_conquer.find_maximum_subarray_kadane),
  )
  def test_find_maximum_subarray(self, algorithm):
    A = np.random.randint(-100, 100, size=(13,))
    (low, high, sum_), _ = algorithm(A.copy())

    # Brute force solution.
    best = (0, len(A) - 1)
    best_sum = np.sum(A)
    for start in range(len(A)):
      for stop in range(start, len(A)):
        range_sum = np.sum(A[start:stop + 1])
        if range_sum > best_sum:
          best = (start, stop)
          best_sum = range_sum

    self.assertEqual(low, best[0])
    self.assertEqual(high, best[1])
    self.assertEqual(sum_, best_sum)


if __name__ == "__main__":
  absltest.main()
