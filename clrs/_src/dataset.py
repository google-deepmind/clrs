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

from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs

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

  def _create_data(self, single_sample):
    algorithm_name = '_'.join(self._builder_config.name.split('_')[:-1])
    num_samples = samplers.CLRS30[self._builder_config.split]['num_samples']
    if self._builder_config.split != 'train':
      # Generate more samples for those algorithms in which the number of
      # signals is small.
      num_samples *= specs.CLRS_30_ALGS_SETTINGS[algorithm_name][
          'num_samples_multiplier']
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
    for i in range(samplers.CLRS30[self._builder_config.split]['num_samples']):
      data = {k: _correct_axis_filtering(v, i, k)
              for k, v in self._instantiated_dataset.items()}
      yield str(i), data


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
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  return (dataset.map(lambda d: _preprocess(d, algorithm=algorithm)),
          specs.SPECS[algorithm])
