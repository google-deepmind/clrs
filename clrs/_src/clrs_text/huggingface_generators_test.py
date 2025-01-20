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
"""Tests for clrs._src.clrs_text.huggingface_generators.

This module contains tests for the huggingface_generators module.
"""

import functools

from absl.testing import absltest
from absl.testing import parameterized

from clrs._src.clrs_text import clrs_utils
from clrs._src.clrs_text import huggingface_generators

import clrs._src.specs as clrs_spec

import datasets


class TestCLRSGenerator(parameterized.TestCase):
  """Check that the generator output matches the expected format."""

  @parameterized.product(
      algo_name=list(clrs_spec.CLRS_30_ALGS_SETTINGS.keys()),
      lengths=[[4, 8]],
      use_hints=[True, False],
      dataset_from_generator_and_num_samples=[
          (
              functools.partial(
                  datasets.Dataset.from_generator,
                  streaming=True,
              ),
              10,
          ),
          (datasets.IterableDataset.from_generator, None),
      ],
  )
  def test_generator_output_format(
      self,
      algo_name,
      lengths,
      use_hints,
      dataset_from_generator_and_num_samples,
  ):
    """Test that the output format of the generator is correct."""
    dataset_from_generator, num_samples = dataset_from_generator_and_num_samples
    clrs_ds = dataset_from_generator(
        huggingface_generators.clrs_generator,
        gen_kwargs={
            "algos_and_lengths": {algo_name: lengths},
            "num_samples": num_samples,
            "use_hints": use_hints,
            "seed": 0,
        },
    )
    # only test the first 10 samples.
    for _, sample in zip(range(10), clrs_ds):
      if use_hints and algo_name in clrs_utils.CLRS_TASKS_WITH_HINTS:
        # question should have schema for trace if hints are used.
        q_regex_hints = (
            rf'^{sample["algo_name"]}:\n.*?initial_trace:.*?\ntrace \| .*?:\n$'
        )
        # answer should have trace ('|' symbol is the identifier for that).
        a_regex_hints = r"^[^a-zA-Z]*\|[^a-zA-Z]*\n\n$"
        self.assertRegex(sample["question"], q_regex_hints)
        self.assertRegex(sample["answer"], a_regex_hints)
      else:
        # question shouldn't have trace part (trace and initial_trace keywords).
        q_regex = (
            rf'^{sample["algo_name"]}:\n(?!.*?initial_trace:).*?\n'
            r"(?!trace \| ).*?:\n$"
        )
        self.assertRegex(sample["question"], q_regex)
        # answer shouldn't have trace identifier ('|' symbol) with hints off.
        self.assertRegex(sample["answer"], r"^[^a-zA-Z\|]*\n\n$")
      self.assertEqual(sample["question"] + sample["answer"], sample["text"])

  @parameterized.product(
      lengths=[[6], [4, 8]],
      use_hints=[True, False],
  )
  def test_auxiliary_fields(self, lengths, use_hints):
    """Test that the auxiliary fields are set correctly."""
    algos_and_lengths = {
        algo_name: lengths for algo_name in clrs_spec.CLRS_30_ALGS_SETTINGS
    }
    clrs_ds = datasets.Dataset.from_generator(
        huggingface_generators.clrs_generator,
        gen_kwargs={
            "algos_and_lengths": algos_and_lengths,
            "num_samples": 200,
            "use_hints": use_hints,
            "seed": 0,
        },
        streaming=True,
    )
    sample_lengths = set()
    sample_algorithms = set()
    for sample in clrs_ds:
      sample_lengths.add(sample["length"])
      sample_algorithms.add(sample["algo_name"])

      self.assertIn(sample["length"], lengths)
      self.assertEqual(sample["use_hints"], use_hints)

    # check that all lengths and algos were sampled from.
    self.assertSetEqual(sample_lengths, set(lengths))
    self.assertSetEqual(sample_algorithms, set(algos_and_lengths.keys()))

  @parameterized.product(
      num_samples=[10, 50, 100],
      num_decimals_in_float=[None, 3],
  )
  def test_dataset_size(self, num_samples, num_decimals_in_float):
    """Test that the dataset size is correct."""
    clrs_ds = datasets.Dataset.from_generator(
        huggingface_generators.clrs_generator,
        gen_kwargs={
            "algos_and_lengths": {
                "insertion_sort": [16],
                "bfs": [8, 10],
            },
            "num_samples": num_samples,
            "seed": 0,
            "num_decimals_in_float": num_decimals_in_float,
        },
        streaming=True,
    )
    ds_iterator = iter(clrs_ds)
    for _ in range(num_samples):
      next(ds_iterator)
    self.assertRaises(StopIteration, next, ds_iterator)


if __name__ == "__main__":
  absltest.main()
