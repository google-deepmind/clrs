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
"""Utilities for calculating losses."""

import chex
from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs

import haiku as hk
import jax
import jax.numpy as jnp

_Array = chex.Array
_DataPoint = probing.DataPoint
_Location = specs.Location
_OutputClass = specs.OutputClass
_Trajectories = samplers.Trajectories
_Trajectory = samplers.Trajectory
_Type = specs.Type


def output_loss(truth: _DataPoint, preds: _Trajectory, nb_nodes: int) -> float:
  """Calculates the output loss."""

  pred = preds[truth.name]
  if truth.type_ == _Type.SCALAR:
    total_loss = jnp.mean((pred - truth.data)**2)

  elif truth.type_ == _Type.MASK:
    loss = (
        jnp.maximum(pred, 0) - pred * truth.data +
        jnp.log1p(jnp.exp(-jnp.abs(pred))))
    mask = (truth.data != _OutputClass.MASKED).astype(jnp.float32)
    total_loss = jnp.sum(loss * mask) / jnp.sum(mask)

  elif truth.type_ in [_Type.MASK_ONE, _Type.CATEGORICAL]:
    unmasked_data = truth.data[truth.data == _OutputClass.POSITIVE]
    masked_truth = truth.data * (truth.data != _OutputClass.MASKED).astype(
        jnp.float32)
    total_loss = (-jnp.sum(masked_truth * jax.nn.log_softmax(pred)) /
                  jnp.sum(unmasked_data))

  elif truth.type_ == _Type.POINTER:
    total_loss = (
        jnp.mean(-jnp.sum(
            hk.one_hot(truth.data, nb_nodes) * jax.nn.log_softmax(pred),
            axis=-1)))

  return total_loss


def diff_loss(diff_logits, gt_diffs, lengths, verbose=False):
  """Calculates diff losses."""
  total_loss = 0.
  verbose_loss = dict()
  length = len(gt_diffs)

  for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
    for i in range(length):
      loss = _diff_loss(loc, i, diff_logits, gt_diffs, lengths) / length
      if verbose:
        verbose_loss[loc + '_diff_%d' % i] = loss
      else:
        total_loss += loss

  return verbose_loss if verbose else total_loss


def _diff_loss(loc, i, diff_logits, gt_diffs, lengths) -> float:
  """Diff loss helper."""
  is_not_done = _is_not_done_broadcast(lengths, i, diff_logits[i][loc])
  loss = (
      jnp.maximum(diff_logits[i][loc], 0) -
      diff_logits[i][loc] * gt_diffs[i][loc] +
      jnp.log1p(jnp.exp(-jnp.abs(diff_logits[i][loc]))) * is_not_done)

  return jnp.mean(loss)


def hint_loss(
    truth: _DataPoint,
    preds: _Trajectories,
    gt_diffs: _Trajectories,
    lengths: _Array,
    nb_nodes: int,
    decode_diffs: bool,
    verbose: bool = False,
):
  """Calculates hint losses."""
  total_loss = 0.
  verbose_loss = {}
  length = truth.data.shape[0] - 1

  for i in range(length):
    loss = _hint_loss(
        i,
        truth=truth,
        preds=preds,
        gt_diffs=gt_diffs,
        lengths=lengths,
        nb_nodes=nb_nodes,
        decode_diffs=decode_diffs,
    ) / length
    if verbose:
      verbose_loss[truth.name + '_%d' % i] = loss
    else:
      total_loss += loss

  return verbose_loss if verbose else total_loss


def _hint_loss(
    i: int,
    truth: _DataPoint,
    preds: _Trajectories,
    gt_diffs: _Trajectories,
    lengths: _Array,
    nb_nodes: int,
    decode_diffs: bool,
) -> float:
  """Hint loss helper."""
  pred = preds[i][truth.name]
  is_not_done = _is_not_done_broadcast(lengths, i, truth.data[i + 1])

  if truth.type_ == _Type.SCALAR:
    if decode_diffs:
      total_loss = jnp.mean((pred - truth.data[i + 1])**2 *
                            gt_diffs[i][truth.location] * is_not_done)
    else:
      total_loss = jnp.mean((pred - truth.data[i + 1])**2 * is_not_done)

  elif truth.type_ == _Type.MASK:
    if decode_diffs:
      loss = jnp.mean(
          jnp.maximum(pred, 0) - pred * truth.data[i + 1] +
          jnp.log1p(jnp.exp(-jnp.abs(pred))) * gt_diffs[i][truth.location] *
          is_not_done)
    else:
      loss = jnp.mean(
          jnp.maximum(pred, 0) - pred * truth.data[i + 1] +
          jnp.log1p(jnp.exp(-jnp.abs(pred))) * is_not_done)
    mask = (truth.data != _OutputClass.MASKED).astype(jnp.float32)
    total_loss = jnp.sum(loss * mask) / jnp.sum(mask)

  elif truth.type_ == _Type.MASK_ONE:
    if decode_diffs:
      total_loss = jnp.mean(-jnp.sum(
          truth.data[i + 1] * jax.nn.log_softmax(pred) * is_not_done,
          axis=-1,
          keepdims=True) * gt_diffs[i][truth.location])
    else:
      total_loss = jnp.mean(-jnp.sum(
          truth.data[i + 1] * jax.nn.log_softmax(pred) * is_not_done, axis=-1))

  elif truth.type_ == _Type.CATEGORICAL:
    unmasked_data = truth.data[truth.data == _OutputClass.POSITIVE]
    masked_truth = truth.data * (truth.data != _OutputClass.MASKED).astype(
        jnp.float32)
    if decode_diffs:
      total_loss = jnp.sum(-jnp.sum(
          masked_truth[i + 1] * jax.nn.log_softmax(pred),
          axis=-1,
          keepdims=True) * jnp.expand_dims(gt_diffs[i][truth.location], -1) *
                           is_not_done) / jnp.sum(unmasked_data)
    else:
      total_loss = jnp.sum(
          -jnp.sum(masked_truth[i + 1] * jax.nn.log_softmax(pred), axis=-1) *
          is_not_done) / jnp.sum(unmasked_data)

  elif truth.type_ == _Type.POINTER:
    if decode_diffs:
      total_loss = jnp.mean(-jnp.sum(
          hk.one_hot(truth.data[i + 1], nb_nodes) * jax.nn.log_softmax(pred),
          axis=-1) * gt_diffs[i][truth.location] * is_not_done)
    else:
      total_loss = jnp.mean(-jnp.sum(
          hk.one_hot(truth.data[i + 1], nb_nodes) * jax.nn.log_softmax(pred),
          axis=-1) * is_not_done)

  return total_loss


def _is_not_done_broadcast(lengths, i, tensor):
  is_not_done = (lengths > i + 1) * 1.0
  while len(is_not_done.shape) < len(tensor.shape):
    is_not_done = jnp.expand_dims(is_not_done, -1)
  return is_not_done
