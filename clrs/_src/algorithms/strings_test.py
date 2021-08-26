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

"""Unit tests for `strings.py`."""

from absl.testing import absltest
from absl.testing import parameterized

from clrs._src.algorithms import strings
import numpy as np


class StringsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("Naive string matching", strings.naive_string_matcher),
      ("KMP string matching", strings.kmp_matcher),
  )
  def test_string_matching(self, algorithm):
    offset, _ = algorithm(np.array([1, 2, 3]), np.array([1, 2, 3]))
    self.assertEqual(offset, 0)
    offset, _ = algorithm(np.array([1, 2, 3, 1, 2]), np.array([1, 2, 3]))
    self.assertEqual(offset, 0)
    offset, _ = algorithm(np.array([1, 2, 3, 1, 2, 3]), np.array([1, 2, 3]))
    self.assertEqual(offset, 0)
    offset, _ = algorithm(np.array([1, 2, 1, 2, 3]), np.array([1, 2, 3]))
    self.assertEqual(offset, 2)
    offset, _ = algorithm(np.array([3, 2, 1]), np.array([1, 2, 3]))
    self.assertEqual(offset, 3)


if __name__ == "__main__":
  absltest.main()
