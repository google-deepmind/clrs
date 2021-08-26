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

"""Searching algorithm generators.

Currently implements the following:
- Minimum
- Binary search
- Quickselect (Hoare, 1961)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""
# pylint: disable=invalid-name


from typing import Tuple, Union

import chex
from clrs._src import probing
from clrs._src import specs
import numpy as np


_Array = np.ndarray
_Numeric = Union[int, float]
_Out = Tuple[int, probing.ProbesDict]


def minimum(A: _Array) -> _Out:
  """Minimum."""

  chex.assert_rank(A, 1)
  probes = probing.initialize(specs.SPECS['minimum'])

  A_pos = np.arange(A.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          'key': np.copy(A)
      })

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pred_h': probing.array(np.copy(A_pos)),
          'min_h': probing.mask_one(0, A.shape[0]),
          'i': probing.mask_one(0, A.shape[0])
      })

  min_ = 0
  for i in range(1, A.shape[0]):
    if A[min_] > A[i]:
      min_ = i

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.array(np.copy(A_pos)),
            'min_h': probing.mask_one(min_, A.shape[0]),
            'i': probing.mask_one(i, A.shape[0])
        })

  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={'min': probing.mask_one(min_, A.shape[0])})

  probing.finalize(probes)

  return min_, probes


def binary_search(x: _Numeric, A: _Array) -> _Out:
  """Binary search."""

  chex.assert_rank(A, 1)
  probes = probing.initialize(specs.SPECS['binary_search'])

  T_pos = np.arange(A.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(T_pos) * 1.0 / A.shape[0],
          'key': np.copy(A),
          'target': x
      })

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pred_h': probing.array(np.copy(T_pos)),
          'low': probing.mask_one(0, A.shape[0]),
          'high': probing.mask_one(A.shape[0] - 1, A.shape[0]),
          'mid': probing.mask_one((A.shape[0] - 1) // 2, A.shape[0]),
      })

  low = 0
  high = A.shape[0] - 1  # make sure return is always in array
  while low < high:
    mid = (low + high) // 2
    if x <= A[mid]:
      high = mid
    else:
      low = mid + 1

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.array(np.copy(T_pos)),
            'low': probing.mask_one(low, A.shape[0]),
            'high': probing.mask_one(high, A.shape[0]),
            'mid': probing.mask_one((low + high) // 2, A.shape[0]),
        })

  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={'return': probing.mask_one(high, A.shape[0])})

  probing.finalize(probes)

  return high, probes


def quickselect(
    A: _Array,
    A_pos=None,
    p=None,
    r=None,
    i=None,
    probes=None,
) -> _Out:
  """Quickselect (Hoare, 1961)."""

  chex.assert_rank(A, 1)

  def partition(A, A_pos, p, r, target, probes):
    x = A[r]
    i = p - 1
    for j in range(p, r):
      if A[j] <= x:
        i += 1
        tmp = A[i]
        A[i] = A[j]
        A[j] = tmp
        tmp = A_pos[i]
        A_pos[i] = A_pos[j]
        A_pos[j] = tmp

      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'pred_h': probing.array(np.copy(A_pos)),
              'p': probing.mask_one(A_pos[p], A.shape[0]),
              'r': probing.mask_one(A_pos[r], A.shape[0]),
              'i': probing.mask_one(A_pos[i + 1], A.shape[0]),
              'j': probing.mask_one(A_pos[j], A.shape[0]),
              'i_rank': (i + 1) * 1.0 / A.shape[0],
              'target': target * 1.0 / A.shape[0]
          })

    tmp = A[i + 1]
    A[i + 1] = A[r]
    A[r] = tmp
    tmp = A_pos[i + 1]
    A_pos[i + 1] = A_pos[r]
    A_pos[r] = tmp

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.array(np.copy(A_pos)),
            'p': probing.mask_one(A_pos[p], A.shape[0]),
            'r': probing.mask_one(A_pos[r], A.shape[0]),
            'i': probing.mask_one(A_pos[i + 1], A.shape[0]),
            'j': probing.mask_one(A_pos[r], A.shape[0]),
            'i_rank': (i + 1 - p) * 1.0 / A.shape[0],
            'target': target * 1.0 / A.shape[0]
        })

    return i + 1

  if A_pos is None:
    A_pos = np.arange(A.shape[0])
  if p is None:
    p = 0
  if r is None:
    r = len(A) - 1
  if i is None:
    i = len(A) // 2
  if probes is None:
    probes = probing.initialize(specs.SPECS['quickselect'])
    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'key': np.copy(A)
        })

  q = partition(A, A_pos, p, r, i, probes)
  k = q - p
  if i == k:
    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={'median': probing.mask_one(A_pos[q], A.shape[0])})
    probing.finalize(probes)
    return A[q], probes
  elif i < k:
    return quickselect(A, A_pos, p, q - 1, i, probes)
  else:
    return quickselect(A, A_pos, q + 1, r, i - k - 1, probes)
