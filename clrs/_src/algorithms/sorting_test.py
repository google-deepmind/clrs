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

"""Unit tests for `sorting.py`."""
# pylint: disable=invalid-name

from absl.testing import absltest
from absl.testing import parameterized

from clrs._src.algorithms import sorting
import numpy as np


class SortingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("Insertion sort", sorting.insertion_sort),
      ("Bubble sort", sorting.bubble_sort),
      ("Heapsort", sorting.heapsort),
      ("Quicksort", sorting.quicksort),
  )
  def test_sorted(self, algorithm):
    for _ in range(17):
      A = np.random.randint(0, 100, size=(13,))
      output, _ = algorithm(A)
      np.testing.assert_array_equal(sorted(A), output)


if __name__ == "__main__":
  absltest.main()
