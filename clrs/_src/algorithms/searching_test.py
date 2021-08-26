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

"""Unit tests for `searching.py`."""
# pylint: disable=invalid-name

from absl.testing import absltest

from clrs._src.algorithms import searching
import numpy as np


EmptyArray = np.asarray([], dtype=np.int32)


class SearchingTest(absltest.TestCase):

  def test_minimum(self):
    for _ in range(17):
      A = np.random.randint(0, 100, size=(13,))
      idx, _ = searching.minimum(A)
      self.assertEqual(A.min(), A[idx])

  def test_binary_search(self):
    A = np.random.randint(0, 100, size=(13,))
    A.sort()
    x = np.random.choice(A)
    idx, _ = searching.binary_search(x, A)
    self.assertEqual(A[idx], x)

  def test_quickselect(self):
    A = np.random.randint(0, 100, size=(13,))
    idx, _ = searching.quickselect(A)
    self.assertEqual(sorted(A)[len(A) // 2], idx)


if __name__ == '__main__':
  absltest.main()
