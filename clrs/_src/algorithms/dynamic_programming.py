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

"""Dynamic programming algorithm generators.

Currently implements the following:
- Matrix-chain multiplication
- Longest common subsequence
- Optimal binary search tree (Aho et al., 1974)

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


def matrix_chain_order(p: _Array) -> _Out:
  """Matrix-chain multiplication."""

  chex.assert_rank(p, 1)
  probes = probing.initialize(specs.SPECS['matrix_chain_order'])

  A_pos = np.arange(p.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A_pos.shape[0],
          'p': np.copy(p)
      })

  m = np.zeros((p.shape[0], p.shape[0]))
  s = np.zeros((p.shape[0], p.shape[0]))
  msk = np.zeros((p.shape[0], p.shape[0]))
  for i in range(1, p.shape[0]):
    m[i, i] = 0
    msk[i, i] = 1
  while True:
    prev_m = np.copy(m)
    prev_msk = np.copy(msk)
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.array(np.copy(A_pos)),
            'm': np.copy(prev_m),
            's_h': np.copy(s),
            'msk': np.copy(msk)
        })
    for i in range(1, p.shape[0]):
      for j in range(i + 1, p.shape[0]):
        flag = prev_msk[i, j]
        for k in range(i, j):
          if prev_msk[i, k] == 1 and prev_msk[k + 1, j] == 1:
            msk[i, j] = 1
            q = prev_m[i, k] + prev_m[k + 1, j] + p[i - 1] * p[k] * p[j]
            if flag == 0 or q < m[i, j]:
              m[i, j] = q
              s[i, j] = k
              flag = 1
    if np.all(prev_m == m):
      break

  probing.push(probes, specs.Stage.OUTPUT, next_probe={'s': np.copy(s)})
  probing.finalize(probes)

  return s[1:, 1:], probes


def lcs_length(x: _Array, y: _Array) -> _Out:
  """Longest common subsequence."""
  chex.assert_rank([x, y], 1)
  probes = probing.initialize(specs.SPECS['lcs_length'])

  x_pos = np.arange(x.shape[0])
  y_pos = np.arange(y.shape[0])
  b = np.zeros((x.shape[0], y.shape[0]))
  c = np.zeros((x.shape[0], y.shape[0]))

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'string': probing.strings_id(x_pos, y_pos),
          'pos': probing.strings_pos(x_pos, y_pos),
          'key': probing.array_cat(np.concatenate([np.copy(x), np.copy(y)]), 4)
      })

  for i in range(x.shape[0]):
    if x[i] == y[0]:
      c[i, 0] = 1
      b[i, 0] = 0
    elif i > 0 and c[i - 1, 0] == 1:
      c[i, 0] = 1
      b[i, 0] = 1
    else:
      c[i, 0] = 0
      b[i, 0] = 1
  for j in range(y.shape[0]):
    if x[0] == y[j]:
      c[0, j] = 1
      b[0, j] = 0
    elif j > 0 and c[0, j - 1] == 1:
      c[0, j] = 1
      b[0, j] = 2
    else:
      c[0, j] = 0
      b[0, j] = 1

  while True:
    prev_c = np.copy(c)

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.strings_pred(x_pos, y_pos),
            'b_h': probing.strings_pair_cat(np.copy(b), 3),
            'c': probing.strings_pair(prev_c)
        })

    for i in range(1, x.shape[0]):
      for j in range(1, y.shape[0]):
        if x[i] == y[j]:
          c[i, j] = prev_c[i - 1, j - 1] + 1
          b[i, j] = 0
        elif prev_c[i - 1, j] >= prev_c[i, j - 1]:
          c[i, j] = prev_c[i - 1, j]
          b[i, j] = 1
        else:
          c[i, j] = prev_c[i, j - 1]
          b[i, j] = 2
    if np.all(prev_c == c):
      break

  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={'b': probing.strings_pair_cat(np.copy(b), 3)})
  probing.finalize(probes)

  return b, probes


def optimal_bst(p: _Array, q: _Array) -> _Out:
  """Optimal binary search tree (Aho et al., 1974)."""

  chex.assert_rank([p, q], 1)
  probes = probing.initialize(specs.SPECS['optimal_bst'])

  A_pos = np.arange(q.shape[0])
  p_cpy = np.zeros(q.shape[0])
  p_cpy[:-1] = np.copy(p)

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / q.shape[0],
          'p': np.copy(p_cpy),
          'q': np.copy(q)
      })

  e = np.zeros((q.shape[0], q.shape[0]))
  w = np.zeros((q.shape[0], q.shape[0]))
  root = np.zeros((q.shape[0], q.shape[0]))
  msks = np.zeros((q.shape[0], q.shape[0]))

  for i in range(q.shape[0]):
    e[i, i] = q[i]
    w[i, i] = q[i]
    msks[i, i] = 1

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pred_h': probing.array(np.copy(A_pos)),
          'root_h': np.copy(root),
          'e': np.copy(e),
          'w': np.copy(w),
          'msk': np.copy(msks)
      })

  for l in range(1, p.shape[0] + 1):
    for i in range(p.shape[0] - l + 1):
      j = i + l
      e[i, j] = 1e9
      w[i, j] = w[i, j - 1] + p[j - 1] + q[j]
      for r in range(i, j):
        t = e[i, r] + e[r + 1, j] + w[i, j]
        if t < e[i, j]:
          e[i, j] = t
          root[i, j] = r
      msks[i, j] = 1
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.array(np.copy(A_pos)),
            'root_h': np.copy(root),
            'e': np.copy(e),
            'w': np.copy(w),
            'msk': np.copy(msks)
        })

  probing.push(probes, specs.Stage.OUTPUT, next_probe={'root': np.copy(root)})
  probing.finalize(probes)

  return root, probes
