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
"""Generate CLRS text dataset.

This script generates a dataset of CLRS text samples in json format.

This dataset is generated with the same parameters which were used in:
"The CLRS-Text Algorithmic Reasoning Language Benchmark". ICML DMLR'24.
 https://arxiv.org/abs/2406.04229

Generate train dataset:
```
python generate_clrs_text.py --split_name train
```

Generate eval dataset:
```
python generate_clrs_text.py --split_name val
```
"""

from collections.abc import Callable
import functools
import itertools
import json
import os
import shutil
from typing import Any, Generator, Sequence

from absl import app
from absl import flags
from absl import logging
from clrs._src import samplers as clrs_samplers
from clrs._src.clrs_text import clrs_utils
import jax
from ml_collections import config_dict
from ml_collections import config_flags
import tensorflow as tf
import tqdm

_DEFAULT_TRAIN_ALGOS_AND_LENGTHS = {
    'articulation_points': [4, 5, 10, 11, 12, 15, 19],
    'activity_selector': [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    'bellman_ford': [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    'bfs': [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    'binary_search': [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    'bridges': [4, 5],
    'bubble_sort': [4, 5, 10],
    'dag_shortest_paths': [4, 5, 10, 11, 12, 15, 19],
    'dfs': [4, 5, 10, 11, 12, 15, 19, 23],
    'dijkstra': [4, 5, 10, 11, 12, 15, 19, 23, 28],
    'find_maximum_subarray_kadane': [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    'floyd_warshall': [4, 5, 10],
    'graham_scan': [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    'heapsort': [4, 5, 10],
    'insertion_sort': [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    'jarvis_march': [4, 5, 10, 11, 12],
    'kmp_matcher': [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    'lcs_length': [4, 5, 10],
    'matrix_chain_order': [4, 5, 10],
    'minimum': [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    'mst_kruskal': [4, 5, 10],
    'mst_prim': [4, 5, 10, 11, 12, 15, 19, 23, 28],
    'naive_string_matcher': [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    'optimal_bst': [4, 5, 10],
    'quickselect': [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    'quicksort': [4, 5, 10],
    'segments_intersect': [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    'strongly_connected_components': [4, 5, 10, 11, 12, 15],
    'task_scheduling': [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    'topological_sort': [4, 5, 10, 11, 12, 15, 19, 23],
}
_DEFAULT_TRAIN_NUMBER_OF_SAMPLES = 10_000
_DEFAULT_TRAIN_SEEDS = [0]

_DEFAULT_VAL_ALGOS_AND_LENGTHS = {
    'activity_selector': list(range(4, 41)),
    'articulation_points': list(range(4, 20)),
    'bellman_ford': list(range(4, 33)),
    'bfs': list(range(4, 42)),
    'binary_search': list(range(4, 65)),
    'bridges': list(range(4, 8)),
    'bubble_sort': list(range(4, 12)),
    'dag_shortest_paths': list(range(4, 20)),
    'dfs': list(range(4, 21)),
    'dijkstra': list(range(4, 26)),
    'find_maximum_subarray_kadane': list(range(4, 65)),
    'floyd_warshall': list(range(4, 12)),
    'graham_scan': list(range(4, 32)),
    'heapsort': list(range(4, 12)),
    'insertion_sort': list(range(4, 26)),
    'jarvis_march': list(range(4, 14)),
    'kmp_matcher': list(range(4, 65)),
    'lcs_length': list(range(4, 13)),
    'matrix_chain_order': list(range(4, 13)),
    'minimum': list(range(4, 65)),
    'mst_kruskal': list(range(4, 11)),
    'mst_prim': list(range(4, 27)),
    'naive_string_matcher': list(range(4, 65)),
    'optimal_bst': list(range(4, 11)),
    'quickselect': list(range(4, 65)),
    'quicksort': list(range(4, 13)),
    'segments_intersect': list(range(4, 65)),
    'strongly_connected_components': list(range(4, 17)),
    'task_scheduling': list(range(4, 42)),
    'topological_sort': list(range(4, 22)),
}
_DEFAULT_VAL_NUMBER_OF_SAMPLES = 125
_DEFAULT_VAL_SEEDS = [3, 14, 35, 81, 94]

_ALGOS_AND_LENGTHS = config_flags.DEFINE_config_dict(
    'algos_and_lengths',
    config_dict.ConfigDict(),
    "The algorithm and lengths. If it's empty and `split_name` is `train` or"
    ' `val`, the default algos and lengths will be used.',
)
_USE_HINTS = flags.DEFINE_bool('use_hints', False, 'Whether to use hints.')
_NUMBER_OF_SAMPLES = flags.DEFINE_integer(
    'number_of_samples',
    -1,
    "The number of samples to generate. If it's -1 and `algos_and_lengths` is"
    ' empty, the default number of samples will be used for `train` and `val`'
    ' split_name.',
)
_PATH_TO_SAVE = flags.DEFINE_string(
    'path_to_save',
    './tmp/clrs_text',
    'The path to save the dataset.',
)
_SPLIT_NAME = flags.DEFINE_string('split_name', 'train', 'The split name.')
_SEEDS = flags.DEFINE_list(
    'seeds',
    [],
    "The seeds to use. If it's empty and `algos_and_lengths` is"
    ' empty, the default seeds will be used for `train` and `val`'
    ' split_name.',
)
_NUM_DECIMALS_IN_FLOAT = flags.DEFINE_integer(
    'num_decimals_in_float',
    None,
    'The number of decimals in float.',
)


CLRS_SAMPLE_SPEC = {
    'prompt': tf.TensorSpec(shape=(), dtype=tf.string),
    'references': tf.TensorSpec(shape=[None], dtype=tf.string),
    'auxiliary': {
        'length': tf.TensorSpec(shape=(), dtype=tf.int32),
        'seed': tf.TensorSpec(shape=(), dtype=tf.int32),
        'use_hints': tf.TensorSpec(shape=(), dtype=tf.bool),
    },
}

CLRS_TF_TENSORS_CONVERTERS = {
    'prompt': lambda x: x.numpy().decode('utf-8'),
    'references': lambda x: [y.numpy().decode('utf-8') for y in x],
    'auxiliary': {
        'length': lambda x: x.numpy().tolist(),
        'seed': lambda x: x.numpy().tolist(),
        'use_hints': lambda x: x.numpy().tolist(),
    },
}


def _convert_to_basic_types(
    sample: dict[str, Any],
    converters: dict[str, Callable[[Any], Any]],
) -> dict[str, Any]:
  """Converts a sample to basic types from tf.Tensor.

  Args:
    sample: The sample to convert.
    converters: The converters to use.

  Returns:
    The converted sample.
  """
  vals, sample_tree = jax.tree.flatten(sample)
  converters, converters_tree = jax.tree.flatten(converters)
  if sample_tree != converters_tree:
    raise ValueError(
        f'Sample tree {sample_tree} and converters tree {converters_tree} '
        'do not match.'
    )
  converted_vals = [converter(val) for val, converter in zip(vals, converters)]
  return jax.tree.unflatten(sample_tree, converted_vals)


def get_dataset_config(
    algo_name: str,
    length: int,
    number_of_samples: int,
    use_hints: bool,
    seed: int,
    num_decimals_in_float: int | None,
) -> config_dict.ConfigDict:
  """Returns the dataset config.

  Args:
    algo_name: The name of the algorithm.
    length: The length of the task.
    number_of_samples: The number of samples to generate.
    use_hints: Whether to use hints.
    seed: The seed to use.
    num_decimals_in_float: The number of decimals in float.

  Returns:
    A config_dict.ConfigDict containing the dataset config.
  """
  config = config_dict.ConfigDict()

  config.algo_name = algo_name
  config.length = length
  config.number_of_samples = number_of_samples
  config.use_hints = use_hints
  config.seed = seed
  config.num_decimals_in_float = num_decimals_in_float

  config.lock()
  return config


def sample_generator(
    algo_name: str,
    number_of_samples: int,
    task_len: int,
    use_hints: bool,
    sampler: clrs_samplers.Sampler,
    seed: int,
) -> Generator[dict[str, Any], None, None]:
  """Generates CLRS samples.

  Args:
    algo_name: The name of the algorithm.
    number_of_samples: The number of samples to generate.
    task_len: The length of the task.
    use_hints: Whether to use hints.
    sampler: The sampler to use.
    seed: The seed to use.

  Yields:
    A dict containing the prompt, references, and auxiliary data.
  """
  for _ in range(number_of_samples):
    sample = sampler.next(batch_size=1)
    question, answer = clrs_utils.format_clrs_example(
        algo_name,
        sample,
        use_hints=use_hints,
    )
    yield {
        'prompt': question,
        'references': [answer],
        'auxiliary': {
            'length': task_len,
            'seed': seed,
            'use_hints': use_hints,
        },
    }


def generate_clrs_algo_dataset(
    config: config_dict.ConfigDict,
) -> tf.data.Dataset:
  """Generates a dataset of CLRS samples.

  Args:
    config: The config for the dataset.

  Returns:
    A tf.data.Dataset of CLRS samples.
  """
  sampler, _ = clrs_samplers.build_sampler(
      name=config.algo_name,
      seed=config.seed,
      num_samples=-1,  # data is generated on the fly.
      length=config.length,
      track_max_steps=False,
      truncate_decimals=config.num_decimals_in_float,
  )

  generator_fn = functools.partial(
      sample_generator,
      config.algo_name,
      config.number_of_samples,
      config.length,
      config.use_hints,
      sampler,
      config.seed,
  )

  dataset = tf.data.Dataset.from_generator(
      generator_fn,
      output_signature=CLRS_SAMPLE_SPEC,
  )

  # Return the dataset.
  return dataset


def _update_generation_params(
    algos_and_lengths: dict[str, list[int]],
    number_of_samples: int,
    seeds: list[str],
) -> tuple[dict[str, list[int]], int, list[int]]:
  """Updates generation params.

  Args:
    algos_and_lengths: The algos and lengths.
    number_of_samples: The number of samples to generate.
    seeds: The seeds to use.

  Returns:
    The updated algos and lengths, number of samples, and seeds.
  """
  seeds = [int(seed) for seed in seeds]

  if not algos_and_lengths:
    logging.info(
        'Setting default algos and lengths for split `%s`. Because'
        ' `algos_and_lengths` is None.',
        _SPLIT_NAME.value,
    )
    match _SPLIT_NAME.value:
      case 'train':
        algos_and_lengths = _DEFAULT_TRAIN_ALGOS_AND_LENGTHS
      case 'val':
        algos_and_lengths = _DEFAULT_VAL_ALGOS_AND_LENGTHS
      case _:
        raise ValueError(
            f'Unsupported split name {_SPLIT_NAME.value} for empty'
            ' `algos_and_lengths` flag. Supported split names are train and'
            ' val.'
        )

    if not seeds:
      logging.info(
          'Setting default seeds for split `%s`. Because `seeds` is None.',
          _SPLIT_NAME.value,
      )
      match _SPLIT_NAME.value:
        case 'train':
          seeds = _DEFAULT_TRAIN_SEEDS
        case 'val':
          seeds = _DEFAULT_VAL_SEEDS
        case _:
          raise ValueError(
              f'Unsupported split name {_SPLIT_NAME.value} for empty'
              ' `algos_and_lengths` flag. Supported split names are train and'
              ' val.'
          )
    if number_of_samples == -1:
      logging.info(
          'Setting default number of samples for split `%s`. Because'
          ' `number_of_samples` is None.',
          _SPLIT_NAME.value,
      )
      match _SPLIT_NAME.value:
        case 'train':
          number_of_samples = _DEFAULT_TRAIN_NUMBER_OF_SAMPLES
        case 'val':
          number_of_samples = _DEFAULT_VAL_NUMBER_OF_SAMPLES
        case _:
          raise ValueError(
              f'Unsupported split name {_SPLIT_NAME.value} for empty'
              ' `algos_and_lengths` flag. Supported split names are train and'
              ' val.'
          )
  else:
    if number_of_samples == -1:
      raise ValueError(
          'Number of samples must be set when `algos_and_lengths` is not None.'
      )
    if not seeds:
      raise ValueError(
          'Seeds must be set when `algos_and_lengths` is not None.'
      )
  return algos_and_lengths, number_of_samples, seeds


def main(_: Sequence[str]) -> None:
  # Remove previous dataset if existed.
  if os.path.exists(_PATH_TO_SAVE.value):
    logging.info('Removing previous dataset dirat %s.', _PATH_TO_SAVE.value)
    shutil.rmtree(_PATH_TO_SAVE.value)

  # Recreate the dataset directory.
  logging.info('Creating dataset dir at %s.', _PATH_TO_SAVE.value)
  os.makedirs(_PATH_TO_SAVE.value)

  # Recreate the split directory.
  algos_and_lengths, number_of_samples, seeds = _update_generation_params(
      _ALGOS_AND_LENGTHS.value,
      _NUMBER_OF_SAMPLES.value,
      _SEEDS.value,
  )

  logging.info(
      'Creating split dir %s at %s.',
      _SPLIT_NAME.value,
      _PATH_TO_SAVE.value,
  )
  split_path = f'{_PATH_TO_SAVE.value}/{_SPLIT_NAME.value}'
  os.makedirs(split_path)

  # Generate JSON one per algorithm.
  for algo_name, lengths in tqdm.tqdm(algos_and_lengths.items()):
    samples = []
    for length, seed in itertools.product(lengths, seeds):
      config = get_dataset_config(
          algo_name=algo_name,
          length=length,
          number_of_samples=number_of_samples,
          use_hints=_USE_HINTS.value,
          seed=seed,
          num_decimals_in_float=_NUM_DECIMALS_IN_FLOAT.value,
      )
      dataset = generate_clrs_algo_dataset(config)
      samples.extend(
          [
              _convert_to_basic_types(sample, CLRS_TF_TENSORS_CONVERTERS)
              for sample in dataset
          ],
      )

    dataset_json = {
        'name': f'clrs_text_{algo_name}',
        'examples': samples,
    }

    file_path = f'{split_path}/{algo_name}.json'
    with open(file_path, 'w') as f:
      logging.info('Writing dataset JSON to %s. Path %s.', f.name, file_path)
      json.dump(dataset_json, f, indent=2)

  logging.info(
      'Done generating CLRS text dataset. Stored at %s.',
      _PATH_TO_SAVE.value,
  )


if __name__ == '__main__':
  app.run(main)
