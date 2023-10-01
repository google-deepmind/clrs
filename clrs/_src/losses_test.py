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

"""Unit tests for `losses.py`."""

from typing import Generator

from absl.testing import absltest
from absl.testing import parameterized

from clrs._src import dataset
from clrs._src import losses
from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs
import jax
import jax.numpy as jnp
import numpy as np

_Array = np.ndarray
_Location = specs.Location


def _make_sampler(algo: str, nb_nodes: int) -> samplers.Sampler:
  sampler, _ = samplers.build_sampler(
      algo,
      seed=samplers.CLRS30['val']['seed'],
      num_samples=samplers.CLRS30['val']['num_samples'],
      length=nb_nodes,
  )
  return sampler


def _make_iterable_sampler(
    algo: str, batch_size: int,
    nb_nodes: int) -> Generator[samplers.Feedback, None, None]:
  sampler = _make_sampler(algo, nb_nodes)
  while True:
    yield sampler.next(batch_size)


def _as_pred_data(x, nb_nodes, seed, batch_axis):
  """Fake a prediction from a data point."""
  # Permute along batch axis to make the prediction different.
  key = jax.random.PRNGKey(seed)
  data = jax.random.permutation(key, x.data, axis=batch_axis)
  # Extend to one-hot for pointer types.
  if x.type_ == specs.Type.POINTER:
    return jax.nn.one_hot(data, nb_nodes)
  return data


def _mask_datapoint(x, seed, t_axis=None):
  """Add some masking to data."""
  key = jax.random.PRNGKey(seed)
  data = x.data
  if x.type_ == specs.Type.MASK:
    # mask some data at random
    mask_shape = list(data.shape)
    if t_axis is not None:
      mask_shape[t_axis] = 1
    mask = jax.random.uniform(key, tuple(mask_shape)) < 0.2
    data = jnp.where(mask, specs.OutputClass.MASKED, data)
  elif x.type_ in [specs.Type.CATEGORICAL, specs.Type.MASK_ONE]:
    # mask some data at random (all categories together)
    mask_shape = list(data.shape)[:-1]
    if t_axis is not None:
      mask_shape[t_axis] = 1
    mask = jax.random.uniform(key, tuple(mask_shape)) < 0.2
    data = jnp.where(mask[..., None], specs.OutputClass.MASKED, data)
  return probing.DataPoint(name=x.name, location=x.location, type_=x.type_,
                           data=data)


def _rand_diff(seed, shape):
  return 2.0 * jax.random.uniform(jax.random.PRNGKey(seed), shape) - 1.0


def _rand_mask(seed, shape, p=0.5):
  return (jax.random.uniform(jax.random.PRNGKey(seed), shape) > p).astype(float)


def invert(d):
  """Dict of lists -> list of dicts."""
  if d:
    return [dict(zip(d, i)) for i in zip(*d.values())]


def _create_data(algo, nb_nodes):
  batch_size = 8

  ds = _make_iterable_sampler(algo, batch_size, nb_nodes)
  full_sample = next(ds)

  chunk_length = full_sample.features.lengths[0].astype(int)
  chunked_ds = dataset.chunkify(
      _make_iterable_sampler(algo, batch_size, nb_nodes),
      chunk_length)
  chunk_sample = next(chunked_ds)
  return full_sample, chunk_sample


class FullVsChunkLossesTest(parameterized.TestCase):
  """Test that the full and chunked versions of the losses match."""

  # Test two algorithms with fixed-length, covering all data types
  @parameterized.parameters('dfs', 'floyd_warshall')
  def test_output_loss(self, algo):
    nb_nodes = 16
    full_sample, chunk_sample = _create_data(algo, nb_nodes)

    # Calculate output loss.
    for truth_full, truth_chunked in zip(full_sample.outputs,
                                         chunk_sample.outputs):
      chunk_output_loss = losses.output_loss_chunked(
          truth=_mask_datapoint(truth_chunked, seed=0),
          pred=_as_pred_data(truth_chunked, nb_nodes, 0, 1),
          is_last=chunk_sample.features.is_last,
          nb_nodes=nb_nodes,
      )
      full_output_loss = losses.output_loss(
          truth=_mask_datapoint(truth_full, seed=0),
          pred=_as_pred_data(truth_full, nb_nodes, 0, 0),
          nb_nodes=nb_nodes,
      )
      np.testing.assert_allclose(chunk_output_loss, full_output_loss, rtol=1e-4)

  @parameterized.parameters('dfs', 'floyd_warshall')
  def test_hint_loss(self, algo):
    nb_nodes = 16
    full_sample, chunk_sample = _create_data(algo, nb_nodes)
    for truth_full, truth_chunked in zip(full_sample.features.hints,
                                         chunk_sample.features.hints):
      np.testing.assert_array_equal(truth_full.data, truth_chunked.data)
      pred = _as_pred_data(truth_chunked, nb_nodes, 0, 1)
      chunk_hint_loss = losses.hint_loss_chunked(
          truth=_mask_datapoint(truth_chunked, seed=1, t_axis=0),
          pred=pred,
          is_first=chunk_sample.features.is_first,
          nb_nodes=nb_nodes,
      )

      full_preds = list(pred[1:])
      full_hint_loss = losses.hint_loss(
          truth=_mask_datapoint(truth_full, 1, t_axis=0),
          preds=full_preds,
          lengths=full_sample.features.lengths,
          nb_nodes=nb_nodes,
      )
      np.testing.assert_allclose(chunk_hint_loss, full_hint_loss, rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
