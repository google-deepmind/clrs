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

from typing import Dict, List, Tuple
import chex
from clrs._src import probing
from clrs._src import specs

import haiku as hk
import jax
import jax.numpy as jnp

_Array = chex.Array
_DataPoint = probing.DataPoint
_Location = specs.Location
_OutputClass = specs.OutputClass
_PredTrajectory = Dict[str, _Array]
_PredTrajectories = List[_PredTrajectory]
_Type = specs.Type

EPS = 1e-12


def _expand_to(x: _Array, y: _Array) -> _Array:
  while len(y.shape) > len(x.shape):
    x = jnp.expand_dims(x, -1)
  return x


def _expand_and_broadcast_to(x: _Array, y: _Array) -> _Array:
  return jnp.broadcast_to(_expand_to(x, y), y.shape)


def output_loss_chunked(truth: _DataPoint, pred: _Array,
                        is_last: _Array, nb_nodes: int) -> float:
  """Output loss for time-chunked training."""

  mask = None

  if truth.type_ == _Type.SCALAR:
    loss = (pred - truth.data)**2

  elif truth.type_ == _Type.MASK:
    loss = (
        jnp.maximum(pred, 0) - pred * truth.data +
        jnp.log1p(jnp.exp(-jnp.abs(pred))))
    mask = (truth.data != _OutputClass.MASKED)

  elif truth.type_ in [_Type.MASK_ONE, _Type.CATEGORICAL]:
    mask = jnp.any(truth.data == _OutputClass.POSITIVE, axis=-1)
    masked_truth = truth.data * (truth.data != _OutputClass.MASKED).astype(
        jnp.float32)
    loss = -jnp.sum(masked_truth * jax.nn.log_softmax(pred), axis=-1)

  elif truth.type_ == _Type.POINTER:
    loss = -jnp.sum(
        hk.one_hot(truth.data, nb_nodes) * jax.nn.log_softmax(pred), axis=-1)

  if mask is not None:
    mask = mask * _expand_and_broadcast_to(is_last, loss)
  else:
    mask = _expand_and_broadcast_to(is_last, loss)
  total_mask = jnp.maximum(jnp.sum(mask), EPS)
  return jnp.sum(jnp.where(mask, loss, 0.0)) / total_mask


def output_loss(truth: _DataPoint, pred: _Array, nb_nodes: int) -> float:
  """Output loss for full-sample training."""

  if truth.type_ == _Type.SCALAR:
    total_loss = jnp.mean((pred - truth.data)**2)

  elif truth.type_ == _Type.MASK:
    loss = (
        jnp.maximum(pred, 0) - pred * truth.data +
        jnp.log1p(jnp.exp(-jnp.abs(pred))))
    mask = (truth.data != _OutputClass.MASKED).astype(jnp.float32)
    total_loss = jnp.sum(loss * mask) / jnp.sum(mask)

  elif truth.type_ in [_Type.MASK_ONE, _Type.CATEGORICAL]:
    masked_truth = truth.data * (truth.data != _OutputClass.MASKED).astype(
        jnp.float32)
    total_loss = (-jnp.sum(masked_truth * jax.nn.log_softmax(pred)) /
                  jnp.sum(truth.data == _OutputClass.POSITIVE))

  elif truth.type_ == _Type.POINTER:
    total_loss = (
        jnp.mean(-jnp.sum(
            hk.one_hot(truth.data, nb_nodes) * jax.nn.log_softmax(pred),
            axis=-1)))

  return total_loss


def diff_loss_chunked(diff_logits, gt_diffs, is_first):
  """Diff loss for time-chunked training."""
  total_loss = 0.
  for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
    valid = (1 - _expand_and_broadcast_to(is_first, diff_logits[loc])).astype(
        jnp.float32)
    total_valid = jnp.maximum(jnp.sum(valid), EPS)
    loss = (
        jnp.maximum(diff_logits[loc], 0) -
        diff_logits[loc] * gt_diffs[loc] +
        jnp.log1p(jnp.exp(-jnp.abs(diff_logits[loc]))))
    total_loss += jnp.sum(jnp.where(valid, loss, 0.0)) / total_valid
  return total_loss


def diff_loss(diff_logits, gt_diffs, lengths, verbose=False):
  """Diff loss for full-sample training."""
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
  """Full-sample diff loss helper."""
  is_not_done = _is_not_done_broadcast(lengths, i, diff_logits[i][loc])
  loss = (
      jnp.maximum(diff_logits[i][loc], 0) -
      diff_logits[i][loc] * gt_diffs[i][loc] +
      jnp.log1p(jnp.exp(-jnp.abs(diff_logits[i][loc]))) * is_not_done)

  return jnp.mean(loss)


def hint_loss_chunked(
    truth: _DataPoint,
    pred: _Array,
    gt_diffs: _PredTrajectory,
    is_first: _Array,
    nb_nodes: int,
    decode_diffs: bool,
):
  """Hint loss for time-chunked training."""
  loss, mask = _hint_loss(
      truth_data=truth.data,
      truth_type=truth.type_,
      pred=pred,
      nb_nodes=nb_nodes,
  )

  mask *= (1 - _expand_to(is_first, loss)).astype(jnp.float32)
  if decode_diffs:
    mask *= gt_diffs[truth.location]
  loss = jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), EPS)
  return loss


def hint_loss(
    truth: _DataPoint,
    preds: List[_Array],
    gt_diffs: _PredTrajectories,
    lengths: _Array,
    nb_nodes: int,
    decode_diffs: bool,
    verbose: bool = False,
):
  """Hint loss for full-sample training."""
  total_loss = 0.
  verbose_loss = {}
  length = truth.data.shape[0] - 1

  loss, mask = _hint_loss(
      truth_data=truth.data[1:],
      truth_type=truth.type_,
      pred=jnp.stack(preds),
      nb_nodes=nb_nodes,
  )
  mask *= _is_not_done_broadcast(lengths, jnp.arange(length)[:, None], loss)
  if decode_diffs:
    mask *= jnp.stack([g[truth.location] for g in gt_diffs])
  loss = jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), EPS)
  if verbose:
    verbose_loss['loss_' + truth.name] = loss
  else:
    total_loss += loss

  return verbose_loss if verbose else total_loss


def _hint_loss(
    truth_data: _Array,
    truth_type: str,
    pred: _Array,
    nb_nodes: int,
) -> Tuple[_Array, _Array]:
  """Hint loss helper."""
  mask = None
  if truth_type == _Type.SCALAR:
    loss = (pred - truth_data)**2

  elif truth_type == _Type.MASK:
    loss = (jnp.maximum(pred, 0) - pred * truth_data +
            jnp.log1p(jnp.exp(-jnp.abs(pred))))
    mask = (truth_data != _OutputClass.MASKED).astype(jnp.float32)

  elif truth_type == _Type.MASK_ONE:
    loss = -jnp.sum(truth_data * jax.nn.log_softmax(pred), axis=-1,
                    keepdims=True)

  elif truth_type == _Type.CATEGORICAL:
    loss = -jnp.sum(truth_data * jax.nn.log_softmax(pred), axis=-1)
    mask = jnp.any(truth_data == _OutputClass.POSITIVE, axis=-1).astype(
        jnp.float32)

  elif truth_type == _Type.POINTER:
    loss = -jnp.sum(
        hk.one_hot(truth_data, nb_nodes) * jax.nn.log_softmax(pred),
        axis=-1)

  if mask is None:
    mask = jnp.ones_like(loss)
  return loss, mask


def _is_not_done_broadcast(lengths, i, tensor):
  is_not_done = (lengths > i + 1) * 1.0
  while len(is_not_done.shape) < len(tensor.shape):
    is_not_done = jnp.expand_dims(is_not_done, -1)
  return is_not_done
