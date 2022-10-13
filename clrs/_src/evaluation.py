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

"""Model base classes and utilities."""

from typing import Dict, List, Tuple
import chex
from clrs._src import probing
from clrs._src import specs
import numpy as np


_Array = chex.Array
Result = Dict[str, probing.DataPoint]


def fuse_perm_and_mask(perm: probing.DataPoint,
                       mask: probing.DataPoint) -> probing.DataPoint:
  """Replace permutation pointers active in the mask with self-pointers.

  Args:
    perm: a node permutation_pointer; data shape is expected to be
      [..., N, N], and ideally one-hot over the last two dimensions, although
      this method does not check for one-hotness.
    mask: a mask_one over nodes; data shape is expected to be
      [..., N], and ideally one-hot over the last dimension, although
      this method does not check for one-hotness.
  Returns:
    A node pointer with shape [..., N].
  """
  assert perm.type_ == specs.Type.PERMUTATION_POINTER
  assert perm.location == specs.Location.NODE
  assert mask.name == perm.name + '_mask'
  assert mask.type_ == specs.Type.MASK_ONE
  assert mask.location == specs.Location.NODE
  assert perm.data.shape[-1] == perm.data.shape[-2]
  assert perm.data.shape[:-1] == mask.data.shape
  data = np.where(mask.data > 0.5,
                  np.arange(perm.data.shape[-1]),  # self-pointers
                  np.argmax(perm.data, axis=-1))   # original pointers
  return probing.DataPoint(name=perm.name,
                           type_=specs.Type.POINTER,
                           location=perm.location,
                           data=data)


def _reduce_permutations_tuple(
    targets: Tuple[probing.DataPoint, ...]) -> Tuple[probing.DataPoint, ...]:
  """Reduce node pointer + mask_one permutation to just node pointer."""
  out_targets = []
  n_perms = 0
  i = 0
  while i < len(targets):
    truth = targets[i]
    if truth.type_ != specs.Type.PERMUTATION_POINTER:
      out_targets.append(truth)
      i += 1
      continue
    truth_mask = targets[i + 1]
    out_targets.append(fuse_perm_and_mask(truth, truth_mask))
    i += 2
    n_perms += 1

  assert len(out_targets) == len(targets) - n_perms
  return tuple(out_targets)


def _reduce_permutations_dict(predictions: Result) -> Result:
  """Reduce node pointer + mask_one permutation to just node pointer."""
  out_preds = {}
  n_perms = 0
  for k, pred in predictions.items():
    if (k.endswith('_mask') and k[:-5] in predictions and
        predictions[k[:-5]].type_ == specs.Type.PERMUTATION_POINTER):
      # This mask will be processed with its associated permutation datapoint
      continue
    if pred.type_ != specs.Type.PERMUTATION_POINTER:
      out_preds[k] = pred
      continue
    pred_mask = predictions[k + '_mask']
    out_preds[k] = fuse_perm_and_mask(pred, pred_mask)
    n_perms += 1

  assert len(out_preds) == len(predictions) - n_perms
  return out_preds


def evaluate_hints(
    hints: Tuple[probing.DataPoint, ...],
    lengths: _Array,
    hint_preds: List[Result],
) -> Dict[str, _Array]:
  """Evaluate hint predictions."""
  evals = {}
  hints = _reduce_permutations_tuple(hints)
  hint_preds = [_reduce_permutations_dict(h) for h in hint_preds]
  for truth in hints:
    assert truth.name in hint_preds[0]
    eval_along_time = [_evaluate(truth, p[truth.name],
                                 idx=i+1, lengths=lengths)
                       for (i, p) in enumerate(hint_preds)]
    evals[truth.name] = np.sum(
        [x * np.sum(i+1 < lengths)
         for i, x in enumerate(eval_along_time)]) / np.sum(lengths - 1)
    evals[truth.name + '_along_time'] = np.array(eval_along_time)

  # Unlike outputs, the hints sometimes include scalars, which don't have
  # a meaningful eval score. So we don't compute a global 'hint score' as we
  # do for outputs.
  return evals


def evaluate(
    outputs: Tuple[probing.DataPoint, ...],
    predictions: Result,
) -> Dict[str, float]:
  """Evaluate output predictions."""
  evals = {}
  outputs = _reduce_permutations_tuple(outputs)
  predictions = _reduce_permutations_dict(predictions)
  for truth in outputs:
    assert truth.name in predictions
    pred = predictions[truth.name]
    evals[truth.name] = _evaluate(truth, pred)
  # Return a single scalar score that is the mean of all output scores.
  evals['score'] = sum([v.item() for v in evals.values()]) / len(evals)
  return evals


def _evaluate(truth, pred, idx=None, lengths=None):
  """Evaluate single prediction of hint or output."""
  assert pred.name == truth.name
  assert pred.location == truth.location
  assert pred.type_ == truth.type_

  if truth.type_ not in _EVAL_FN:
    raise ValueError('Invalid type')
  truth_data = truth.data
  pred_data = pred.data
  if idx is not None:
    if np.all(idx >= lengths):
      return 0.
    truth_data = truth_data[idx][idx < lengths]
    pred_data = pred_data[idx < lengths]
  return _EVAL_FN[truth.type_](pred_data, truth_data)


def _eval_one(pred, truth):
  mask = np.all(truth != specs.OutputClass.MASKED, axis=-1)
  return np.sum(
      (np.argmax(pred, -1) == np.argmax(truth, -1)) * mask) / np.sum(mask)


def _mask_fn(pred, truth):
  """Evaluate outputs of type MASK, and account for any class imbalance."""
  mask = (truth != specs.OutputClass.MASKED).astype(np.float32)

  # Use F1 score for the masked outputs to address any imbalance
  tp = np.sum((((pred > 0.5) * (truth > 0.5)) * 1.0) * mask)
  fp = np.sum((((pred > 0.5) * (truth < 0.5)) * 1.0) * mask)
  fn = np.sum((((pred < 0.5) * (truth > 0.5)) * 1.0) * mask)

  # Protect against division by zero
  if tp + fp > 0:
    precision = tp / (tp + fp)
  else:
    precision = np.float32(1.0)
  if tp + fn > 0:
    recall = tp / (tp + fn)
  else:
    recall = np.float32(1.0)

  if precision + recall > 0.0:
    f_1 = 2.0 * precision * recall / (precision + recall)
  else:
    f_1 = np.float32(0.0)

  return f_1

_EVAL_FN = {
    specs.Type.SCALAR:
        lambda pred, truth: np.mean((pred - truth)**2),
    specs.Type.MASK: _mask_fn,
    specs.Type.MASK_ONE:
        _eval_one,
    specs.Type.CATEGORICAL:
        _eval_one,
    specs.Type.POINTER:
        lambda pred, truth: np.mean((pred == truth) * 1.0),
}
