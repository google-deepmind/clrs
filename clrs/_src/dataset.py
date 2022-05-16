# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""CLRS dataset."""

import dataclasses

import functools
from typing import Iterator

from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _correct_axis_filtering(tensor, index, name):
  if 'hint_' in name:
    return tensor[:, index]
  else:
    return tensor[index]


@dataclasses.dataclass
class CLRSConfig(tfds.core.BuilderConfig):
  """Specify the split in the variant because they have different shapes."""
  split: str = ''


DEFAULT_BUILDER_CONFIGS = []


def _build_default_builder_configs():
  for split in ['train', 'val', 'test']:
    for alg in specs.CLRS_30_ALGS:
      DEFAULT_BUILDER_CONFIGS.append(
          CLRSConfig(name=f'{alg}_{split}', split=split))


_build_default_builder_configs()


class CLRSDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  BUILDER_CONFIGS = DEFAULT_BUILDER_CONFIGS

  _instantiated_dataset = None
  _instantiated_dataset_name = ''
  _instantiated_dataset_split = ''

  def _num_samples(self, algorithm_name):
    num_samples = samplers.CLRS30[self._builder_config.split]['num_samples']
    if self._builder_config.split != 'train':
      # Generate more samples for those algorithms in which the number of
      # signals is small.
      num_samples *= specs.CLRS_30_ALGS_SETTINGS[algorithm_name][
          'num_samples_multiplier']
    return num_samples

  def _create_data(self, single_sample):
    algorithm_name = '_'.join(self._builder_config.name.split('_')[:-1])
    num_samples = self._num_samples(algorithm_name)
    sampler, _ = samplers.build_sampler(
        algorithm_name,
        seed=samplers.CLRS30[self._builder_config.split]['seed'],
        num_samples=num_samples,
        length=samplers.CLRS30[self._builder_config.split]['length'],
    )
    sampled_dataset = sampler.next(batch_size=1 if single_sample else None)
    data = {'input_' + t.name: t.data for t in sampled_dataset.features.inputs}
    # All other data points have input_, hint_, and output_ prefixes, so we
    # guarantee that this key is unused.
    data['lengths'] = sampled_dataset.features.lengths
    data.update({'output_' + t.name: t.data for t in sampled_dataset.outputs})
    data.update({
        'hint_' + t.name: t.data for t in sampled_dataset.features.hints})
    self._instantiated_dataset = data

  def _info(self) -> tfds.core.DatasetInfo:
    if (self._instantiated_dataset_name != self._builder_config.name
        or self._instantiated_dataset_split != self._builder_config.split):
      self._create_data(single_sample=True)

    data = {k: _correct_axis_filtering(v, 0, k)
            for k, v in self._instantiated_dataset.items()}
    data_info = {
        k: tfds.features.Tensor(shape=v.shape, dtype=tf.dtypes.as_dtype(
            v.dtype)) for k, v in data.items()}
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict(data_info),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    if (self._instantiated_dataset_name != self._builder_config.name
        or self._instantiated_dataset_split != self._builder_config.split):
      self._create_data(single_sample=False)
      self._instantiated_dataset_name = self._builder_config.name
      self._instantiated_dataset_split = self._builder_config.split
    return {self._builder_config.split: self._generate_examples()}

  def _generate_examples(self):
    """Generator of examples for each split."""
    algorithm_name = '_'.join(self._builder_config.name.split('_')[:-1])
    for i in range(self._num_samples(algorithm_name)):
      data = {k: _correct_axis_filtering(v, i, k)
              for k, v in self._instantiated_dataset.items()}
      yield str(i), data


def _get_clrs_file_name():
  return f'CLRS30_v{CLRSDataset.VERSION}.tar.gz'


def get_dataset_gcp_url():
  return f'https://storage.googleapis.com/dm-clrs/{_get_clrs_file_name()}'


def get_clrs_folder():
  return f'CLRS30_v{CLRSDataset.VERSION}'


def _preprocess(data_point, algorithm=None):
  """Convert sampled inputs into DataPoints."""
  inputs = []
  outputs = []
  hints = []
  lengths = None

  for name, data in data_point.items():
    if name == 'lengths':
      lengths = data
      continue
    data_point_name = name.split('_')
    name = '_'.join(data_point_name[1:])
    (stage, location, dp_type) = specs.SPECS[algorithm][name]
    assert stage == data_point_name[0]
    if stage == specs.Stage.HINT:
      data = tf.experimental.numpy.swapaxes(data, 0, 1)
    dp = probing.DataPoint(name, location, dp_type, data)
    if stage == specs.Stage.INPUT:
      inputs.append(dp)
    elif stage == specs.Stage.OUTPUT:
      outputs.append(dp)
    else:
      hints.append(dp)
  return samplers.Feedback(
      samplers.Features(tuple(inputs), tuple(hints), lengths), tuple(outputs))


def create_dataset(folder, algorithm, split, batch_size):
  dataset = tfds.load(f'clrs_dataset/{algorithm}_{split}',
                      data_dir=folder, split=split)
  num_samples = len(dataset)  # Must be done here for correct size
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  return (dataset.map(lambda d: _preprocess(d, algorithm=algorithm)),
          num_samples,
          specs.SPECS[algorithm])


def _copy_hint(source, dest, i, start_source, start_dest, to_add):
  """Copy from full-sample hint to a hint chunk."""
  assert np.all(dest[start_dest:, i:] == 0)
  assert start_dest < dest.shape[0]
  assert start_dest + to_add <= dest.shape[0]
  assert start_source < source.shape[0]
  assert start_source + to_add <= source.shape[0]
  dest[start_dest:start_dest+to_add, i] = source[
      start_source:start_source+to_add, i]
  return dest


def _copy_io(source, dest, i, start_dest, to_add):
  """Copy from an input or output to an input or output chunk."""
  assert np.all(dest[start_dest:, i:] == 0)
  dest[start_dest:start_dest+to_add, i] = source[i]
  return dest


def chunkify(dataset: Iterator[samplers.Feedback], chunk_length: int):
  """Generator of fixed-length chunks from full-trajectory samples.

  Args:
    dataset: full-sample dataset as numpy iterator.
    chunk_length: time length of chunks.
  Yields:
    Fixed-timelength chunks of data. Each tensor of inputs, hints and outputs
    has dimensions chunk_length x batch_size x ... Samples are not time-padded,
    after the end of one sample immediately comes the next. Since different
    samples can have different time lengths, the beginnings and ends of samples
    within a batch do not need to coincide. For this reason, the chunked
    dataset features include two chunk_length x batch_size int tensors,
    `is_first` and `is_last`, that mark the beginning and end of each sample.
    For example, if `chunk_legnth`==6 and `batch_size`==2 and the first
    full-sample batch had one sample of length 3 and one of length 5,
    we would have a first chunked batch with the following `is_first` and
    `is_last` tensors:

    is_first = [[1, 1]    is_last = [[0, 0]     ( sample id [[0 1]
                [0, 0]               [0, 0]                  [0 1]
                [0, 0]               [1, 0]                  [0 1]
                [1, 0]               [0, 0]                  [2 1]
                [0, 0]               [0, 1]                  [2 1]
                [0, 1]]              [0, 0]]                 [2 3]] )

    while the data in the inputs, outputs and hints tensors would correspond
    to samples as identified by the sample_id indicated above for reference.
    Notice that, while in the full-sample dataset inputs and outputs have
    no time dimension, here they do; the input and output tensors are simply
    repeated along each sample's time length.
  """
  def _get_batch():
    d = next(dataset)
    return (d.features.inputs, d.features.hints, d.outputs,
            d.features.lengths.astype(int))

  inputs, hints, outputs, lengths = _get_batch()
  for inp in inputs:
    if inp.location in [specs.Location.NODE, specs.Location.EDGE]:
      batch_size = inp.data.shape[0]
      break

  io_chunk = lambda x: np.zeros((chunk_length,) + x.shape, dtype=x.dtype)
  chunk_inputs = jax.tree_map(io_chunk, inputs)
  chunk_outputs = jax.tree_map(io_chunk, outputs)

  hint_chunk = lambda x: np.zeros((chunk_length,) + x.shape[1:], dtype=x.dtype)
  chunk_hints = jax.tree_map(hint_chunk, hints)

  inputs = [inputs]
  hints = [hints]
  outputs = [outputs]
  left = [lengths.copy()]
  lengths = [lengths.copy()]

  while True:
    # Create a new empty chunk
    chunk_inputs = jax.tree_map(np.zeros_like, chunk_inputs)
    chunk_hints = jax.tree_map(np.zeros_like, chunk_hints)
    chunk_outputs = jax.tree_map(np.zeros_like, chunk_outputs)
    start_mark = np.zeros((chunk_length, batch_size), dtype=int)
    end_mark = np.zeros((chunk_length, batch_size), dtype=int)

    # Get enough data batches to fill the new chunk
    while np.any(np.sum(left, axis=0) < chunk_length):
      inp, hh, out, ll = _get_batch()
      inputs.append(inp)
      hints.append(hh)
      outputs.append(out)
      left.append(ll.copy())
      lengths.append(ll.copy())

    # Fill the chunk, one batch element at a time
    for i in range(batch_size):
      total, idx = 0, 0
      while total < chunk_length:
        to_add = min(left[idx][i], chunk_length - total)
        if to_add:
          start = lengths[idx][i] - left[idx][i]
          assert start >= 0
          f_io = functools.partial(_copy_io, i=i, start_dest=total,
                                   to_add=to_add)
          chunk_inputs = jax.tree_map(f_io, inputs[idx], chunk_inputs)
          chunk_outputs = jax.tree_map(f_io, outputs[idx], chunk_outputs)
          f_hint = functools.partial(_copy_hint, i=i, start_source=start,
                                     start_dest=total, to_add=to_add)
          chunk_hints = jax.tree_map(f_hint, hints[idx], chunk_hints)
          if start == 0:
            start_mark[total, i] = 1
          total += to_add
          left[idx][i] -= to_add
          assert left[idx][i] >= 0
          if left[idx][i] == 0:
            end_mark[total - 1, i] = 1
        idx += 1
      assert total == chunk_length

    while left and np.all(left[0] == 0):
      inputs.pop(0)
      hints.pop(0)
      outputs.pop(0)
      left.pop(0)
      lengths.pop(0)

    yield samplers.Feedback(
        samplers.FeaturesChunked(chunk_inputs, chunk_hints,
                                 start_mark, end_mark),
        chunk_outputs)


def create_chunked_dataset(folder, algorithm, split, batch_size, chunk_length):
  dataset = tfds.load(f'clrs_dataset/{algorithm}_{split}',
                      data_dir=folder, split=split)
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(lambda d: _preprocess(d, algorithm=algorithm))
  dataset = dataset.as_numpy_iterator()
  return chunkify(dataset, chunk_length), specs.SPECS[algorithm]
