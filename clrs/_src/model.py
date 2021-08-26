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

import abc
from typing import Dict, Optional

from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs
import numpy as np


Result = Dict[str, probing.DataPoint]


class Model(abc.ABC):
  """Abstract base class for CLRS3-B models."""

  def __init__(self, spec: specs.Spec):
    """Set up the problem, prepare to predict on first task."""
    self._spec = spec

  @abc.abstractmethod
  def predict(self, features: samplers.Features) -> Result:
    """Make predictions about the current task."""
    pass

  @abc.abstractmethod
  def feedback(self, feedback: Optional[samplers.Feedback]):
    """Advance to the next task, incorporating any available feedback."""
    pass


def evaluate(
    feedback: samplers.Feedback,
    predictions: Result,
) -> Dict[str, float]:
  """Evaluate predictions."""
  evals = {}
  for truth in feedback.outputs:
    assert truth.name in predictions
    pred = predictions[truth.name]
    assert pred.name == truth.name
    assert pred.location == truth.location
    assert pred.type_ == truth.type_
    mask_name = f'{truth.name}_mask'
    if mask_name in feedback.outputs:
      mask = feedback.outputs[mask_name].data
      evals[truth.name] = np.mean(
          (pred.data[mask].flatten() - truth.data[mask].flatten())**2)
    else:
      if truth.type_ not in _EVAL_FN:
        raise ValueError('Invalid type')
      evals[truth.name] = _EVAL_FN[truth.type_](pred.data, truth.data)

  # Return a single scalar score that is the mean of all output scores.
  evals['score'] = sum([v.item() for v in evals.values()]) / len(evals)
  return evals


def _eval_one(pred, truth):
  mask = np.all(truth != specs.OutputClass.MASKED.value, axis=-1)
  return np.sum(
      (np.argmax(pred, -1) == np.argmax(truth, -1)) * mask) / np.sum(mask)


def _mask_fn(pred, truth):
  mask = (truth != specs.OutputClass.MASKED.value).astype(np.float32)
  return np.sum((((pred > 0.0) == (truth > 0.5)) * 1.0) * mask)/np.sum(mask)

_EVAL_FN = {
    specs.Type.SCALAR:
        lambda pred, truth: np.mean((pred - truth)**2),
    specs.Type.MASK: _mask_fn,
    specs.Type.MASK_ONE:
        _eval_one,
    specs.Type.CATEGORICAL:
        _eval_one,
    specs.Type.POINTER:
        lambda pred, truth: np.mean((pred == truth) * 1.0)
}
