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
"""Functions to allow for Huggingface Integration."""

import random
from typing import Dict, List, Optional

from clrs._src import samplers
from clrs._src.clrs_text import clrs_utils


def clrs_generator(
    algos_and_lengths: Dict[str, List[int]],
    num_samples: Optional[int] = None,
    use_hints: bool = False,
    seed: int = 0,
    num_decimals_in_float: int = 3,
):
  """Huggingface compatible generator function for CLRS-text dataset.

  Example usage for a finite dataset:
    algos_and_lengths = {"insertion_sort": [16]}
    ds = datasets.Dataset.from_generator(
        clrs_generator, gen_kwargs={
            "algos_and_lengths": algos_and_lengths,
            "num_samples": 100
        }
  )

  Example usage for infinite dataset:
    algos_and_lengths = {"insertion_sort": [16]}
    ds = IterableDataset.from_generator(
        clrs_generator,
        features=Features(
            {
                "text": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
                "answer": Value(dtype="string", id=None),
                "algo_name": Value(dtype="string", id=None),
                "length": Value(dtype="int32", id=None),
                "use_hints": Value(dtype="bool_", id=None),
            }
        ),
        gen_kwargs={"algos_and_lengths": algos_and_lengths},
    )
    # features is an optional argument but can be included for better
    # integration with huggingface.


  Huggingface references:
    - https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#datasets.Dataset.from_generator  # pylint: disable=line-too-long
    - https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#datasets.IterableDataset.from_generator  # pylint: disable=line-too-long

  Args:
    algos_and_lengths: keys = algorithm names
        [Must be same as in clrs.CLRS_30_ALGS_SETTINGS.keys()],
        values = list of lengths required for that algorithm.
    num_samples: The size of the output dataset,
        if None the output dataset will be infinite to be used with
        IterableDataset.from_generator.
    use_hints: Whether hints should be included in the question and answer.
    seed: The random seed for all of the generators.
    num_decimals_in_float: The number of decimals to truncate floats to. Defaults
        to 3.

  Yields:
    A dictionary with the following keys:
        text: The question and answer concatenated.
        question: The question from the CLRS-text dataset.
        answer: The correct output for the given algorithm question.
        algo_name: The name of the algorithm.
        length: The length of the input.
        use_hints: Whether hints were used.
  """
  clrs_samplers = []

  # make all of the possible generators.
  for algo_name, lengths in algos_and_lengths.items():
    for length in lengths:
      sampler, _ = samplers.build_sampler(
          algo_name,
          seed=seed,
          num_samples=-1,
          length=length,
          track_max_steps=False,
          use_padding=False,
          truncate_decimals=num_decimals_in_float,
      )
      clrs_samplers.append((sampler, algo_name, length))

  random.seed(seed)

  # num_samples is set to None for infinite generator.
  infinite_loop = num_samples is None
  sample_count = 0

  while infinite_loop or sample_count < num_samples:
    sampler, algo_name, length = random.choice(clrs_samplers)
    sample = sampler.next(batch_size=1)  # get one sample from the sampler.
    question, answer = clrs_utils.format_clrs_example(
        algo_name,
        sample,
        use_hints=use_hints,
    )

    # There is no added separator between the question and answer because the
    # question ends with a newline.
    text = question + answer
    sample_count += 1
    yield {
        "text": text,
        "question": question,
        "answer": answer,
        "algo_name": algo_name,
        "length": length,
        "use_hints": use_hints,
    }
