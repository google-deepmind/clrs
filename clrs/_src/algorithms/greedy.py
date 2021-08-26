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

"""Greedy algorithm generators.

Currently implements the following:
- Activity selection (Gavril, 1972)
- Task scheduling (Lawler, 1985)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""
# pylint: disable=invalid-name

from typing import Tuple

import chex
from clrs._src import probing
from clrs._src import specs
import numpy as np


_Array = np.ndarray
_Out = Tuple[_Array, probing.ProbesDict]


def activity_selector(s: _Array, f: _Array) -> _Out:
  """Activity selection (Gavril, 1972)."""

  chex.assert_rank([s, f], 1)
  probes = probing.initialize(specs.SPECS['activity_selector'])

  A_pos = np.arange(s.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A_pos.shape[0],
          's': np.copy(s),
          'f': np.copy(f)
      })

  A = np.zeros(s.shape[0])

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pred_h': probing.array(np.copy(A_pos)),
          'selected_h': np.copy(A),
          'm': probing.mask_one(0, A_pos.shape[0]),
          'k': probing.mask_one(0, A_pos.shape[0])
      })

  ind = np.argsort(f)
  A[ind[0]] = 1
  k = ind[0]

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pred_h': probing.array(np.copy(A_pos)),
          'selected_h': np.copy(A),
          'm': probing.mask_one(ind[0], A_pos.shape[0]),
          'k': probing.mask_one(k, A_pos.shape[0])
      })

  for m in range(1, s.shape[0]):
    if s[ind[m]] >= f[k]:
      A[ind[m]] = 1
      k = ind[m]
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.array(np.copy(A_pos)),
            'selected_h': np.copy(A),
            'm': probing.mask_one(ind[m], A_pos.shape[0]),
            'k': probing.mask_one(k, A_pos.shape[0])
        })

  probing.push(probes, specs.Stage.OUTPUT, next_probe={'selected': np.copy(A)})
  probing.finalize(probes)

  return A, probes


def task_scheduling(d: _Array, w: _Array) -> _Out:
  """Task scheduling (Lawler, 1985)."""

  chex.assert_rank([d, w], 1)
  probes = probing.initialize(specs.SPECS['task_scheduling'])

  A_pos = np.arange(d.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A_pos.shape[0],
          'd': np.copy(d),
          'w': np.copy(w)
      })

  A = np.zeros(d.shape[0])

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pred_h': probing.array(np.copy(A_pos)),
          'selected_h': np.copy(A),
          'i': probing.mask_one(0, A_pos.shape[0]),
          't': 0
      })

  ind = np.argsort(-w)
  A[ind[0]] = 1
  t = 1

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pred_h': probing.array(np.copy(A_pos)),
          'selected_h': np.copy(A),
          'i': probing.mask_one(ind[0], A_pos.shape[0]),
          't': t
      })

  for i in range(1, d.shape[0]):
    if t < d[ind[i]]:
      A[ind[i]] = 1
      t += 1
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.array(np.copy(A_pos)),
            'selected_h': np.copy(A),
            'i': probing.mask_one(ind[i], A_pos.shape[0]),
            't': t
        })

  probing.push(probes, specs.Stage.OUTPUT, next_probe={'selected': np.copy(A)})
  probing.finalize(probes)

  return A, probes
