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
"""Tests for lazy import of model components in clrs."""

from absl.testing import absltest
import clrs


class LazyImportTest(absltest.TestCase):

  def test_core_attributes_available(self):
    self.assertTrue(hasattr(clrs, "CLRSDataset"))
    self.assertTrue(hasattr(clrs, "build_sampler"))
    self.assertTrue(hasattr(clrs, "algorithms"))
    self.assertTrue(hasattr(clrs, "specs"))
    self.assertTrue(hasattr(clrs, "clrs_text"))

  def test_dir_contains_lazy_attributes(self):
    names = dir(clrs)
    self.assertIn("models", names)
    self.assertIn("decoders", names)
    self.assertIn("processors", names)
    self.assertIn("get_processor_factory", names)

  def test_missing_attribute_raises_attribute_error(self):
    with self.assertRaises(AttributeError):
      _ = clrs.non_existent_attribute_xyz


if __name__ == "__main__":
  absltest.main()
