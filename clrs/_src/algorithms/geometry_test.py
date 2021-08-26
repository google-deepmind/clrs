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

"""Unit tests for `geometry.py`."""

from absl.testing import absltest
from absl.testing import parameterized

from clrs._src.algorithms import geometry
import numpy as np


class GeometryTest(parameterized.TestCase):

  def test_segments_simple(self):
    xs_no = np.array([0, 0, 1, 1])
    ys_no = np.array([0, 1, 0, 1])
    out, _ = geometry.segments_intersect(xs_no, ys_no)
    self.assertFalse(out)

    xs_yes = np.array([0, 1, 1, 0])
    ys_yes = np.array([0, 1, 0, 1])
    out, _ = geometry.segments_intersect(xs_yes, ys_yes)
    self.assertTrue(out)

    xs_just = np.array([-3, 5, 5, -4])
    ys_just = np.array([-3, 5, 5, -4])
    out, _ = geometry.segments_intersect(xs_just, ys_just)
    self.assertTrue(out)

  def test_segments_colinear(self):
    xs_no = np.array([-1, 1, 2, 4])
    ys_no = np.array([-1, 1, 2, 4])
    out, _ = geometry.segments_intersect(xs_no, ys_no)
    self.assertFalse(out)

    xs_yes = np.array([-3, 5, 1, 2])
    ys_yes = np.array([-3, 5, 1, 2])
    out, _ = geometry.segments_intersect(xs_yes, ys_yes)
    self.assertTrue(out)

    xs_just = np.array([-3, 5, 5, 7])
    ys_just = np.array([-3, 5, 5, 7])
    out, _ = geometry.segments_intersect(xs_just, ys_just)
    self.assertTrue(out)

  @parameterized.named_parameters(
      ("Graham scan convex hull", geometry.graham_scan),
      ("Jarvis' march convex hull", geometry.jarvis_march),
  )
  def test_convex_hull_simple(self, algorithm):
    tt = np.linspace(-np.pi, np.pi, 10)[:-1]
    xs = np.cos(tt)
    ys = np.sin(tt)
    in_hull, _ = algorithm(xs, ys)
    self.assertTrue(np.all(in_hull == 1))

    xs = np.append(xs, [0.1])
    ys = np.append(ys, [0.1])
    in_hull, _ = algorithm(xs, ys)
    self.assertTrue(np.all(in_hull[:-1] == 1))
    self.assertTrue(np.all(in_hull[-1:] == 0))

  @parameterized.named_parameters(
      ("Graham scan convex hull", geometry.graham_scan),
      ("Jarvis' march convex hull", geometry.jarvis_march),
  )
  def test_convex_hull_points(self, algorithm):
    xs = np.array([0, 15, 20, 30, 50, 50, 55, 70])
    ys = np.array([30, 25, 0, 60, 40, 10, 20, 30])
    expected = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    out, _ = algorithm(xs, ys)
    np.testing.assert_array_equal(expected, out)


if __name__ == "__main__":
  absltest.main()
