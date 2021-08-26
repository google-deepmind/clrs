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

"""Strings algorithm generators.

Currently implements the following:
- Naive string matching
- Knuth-Morris-Pratt string matching (Knuth et al., 1977)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""
# pylint: disable=invalid-name


from typing import Tuple

import chex
from clrs._src import probing
from clrs._src import specs
import numpy as np


_Array = np.ndarray
_Out = Tuple[int, probing.ProbesDict]

_ALPHABET_SIZE = 4


def naive_string_matcher(T: _Array, P: _Array) -> _Out:
  """Naive string matching."""

  chex.assert_rank([T, P], 1)
  probes = probing.initialize(specs.SPECS['naive_string_matcher'])

  T_pos = np.arange(T.shape[0])
  P_pos = np.arange(P.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'string':
              probing.strings_id(T_pos, P_pos),
          'pos':
              probing.strings_pos(T_pos, P_pos),
          'key':
              probing.array_cat(
                  np.concatenate([np.copy(T), np.copy(P)]), _ALPHABET_SIZE),
      })

  s = 0
  while s <= T.shape[0] - P.shape[0]:
    i = s
    j = 0

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.strings_pred(T_pos, P_pos),
            's': probing.mask_one(s, T.shape[0] + P.shape[0]),
            'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
            'j': probing.mask_one(T.shape[0] + j, T.shape[0] + P.shape[0])
        })

    while True:
      if T[i] != P[j]:
        break
      elif j == P.shape[0] - 1:
        probing.push(
            probes,
            specs.Stage.OUTPUT,
            next_probe={'match': probing.mask_one(s, T.shape[0] + P.shape[0])})
        probing.finalize(probes)
        return s, probes
      else:
        i += 1
        j += 1
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'pred_h': probing.strings_pred(T_pos, P_pos),
                's': probing.mask_one(s, T.shape[0] + P.shape[0]),
                'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
                'j': probing.mask_one(T.shape[0] + j, T.shape[0] + P.shape[0])
            })

    s += 1

  # By convention, set probe to head of needle if no match is found
  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={
          'match': probing.mask_one(T.shape[0], T.shape[0] + P.shape[0])
      })
  return T.shape[0], probes


def kmp_matcher(T: _Array, P: _Array) -> _Out:
  """Knuth-Morris-Pratt string matching (Knuth et al., 1977)."""

  chex.assert_rank([T, P], 1)
  probes = probing.initialize(specs.SPECS['kmp_matcher'])

  T_pos = np.arange(T.shape[0])
  P_pos = np.arange(P.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'string':
              probing.strings_id(T_pos, P_pos),
          'pos':
              probing.strings_pos(T_pos, P_pos),
          'key':
              probing.array_cat(
                  np.concatenate([np.copy(T), np.copy(P)]), _ALPHABET_SIZE),
      })

  pi = np.arange(P.shape[0])
  k = 0

  # Cover the edge case where |P| = 1, and the first half is not executed.
  delta = 1 if P.shape[0] > 1 else 0

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pred_h': probing.strings_pred(T_pos, P_pos),
          'pi': probing.strings_pi(T_pos, P_pos, pi),
          'k': probing.mask_one(T.shape[0], T.shape[0] + P.shape[0]),
          'q': probing.mask_one(T.shape[0] + delta, T.shape[0] + P.shape[0]),
          's': probing.mask_one(0, T.shape[0] + P.shape[0]),
          'i': probing.mask_one(0, T.shape[0] + P.shape[0]),
          'phase': 0
      })

  for q in range(1, P.shape[0]):
    while k != pi[k] and P[k] != P[q]:
      k = pi[k]
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'pred_h': probing.strings_pred(T_pos, P_pos),
              'pi': probing.strings_pi(T_pos, P_pos, pi),
              'k': probing.mask_one(T.shape[0] + k, T.shape[0] + P.shape[0]),
              'q': probing.mask_one(T.shape[0] + q, T.shape[0] + P.shape[0]),
              's': probing.mask_one(0, T.shape[0] + P.shape[0]),
              'i': probing.mask_one(0, T.shape[0] + P.shape[0]),
              'phase': 0
          })
    if P[k] == P[q]:
      k += 1
    pi[q] = k
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.strings_pred(T_pos, P_pos),
            'pi': probing.strings_pi(T_pos, P_pos, pi),
            'k': probing.mask_one(T.shape[0] + k, T.shape[0] + P.shape[0]),
            'q': probing.mask_one(T.shape[0] + q, T.shape[0] + P.shape[0]),
            's': probing.mask_one(0, T.shape[0] + P.shape[0]),
            'i': probing.mask_one(0, T.shape[0] + P.shape[0]),
            'phase': 0
        })
  q = 0
  s = 0
  for i in range(T.shape[0]):
    if i >= P.shape[0]:
      s += 1
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.strings_pred(T_pos, P_pos),
            'pi': probing.strings_pi(T_pos, P_pos, pi),
            'k': probing.mask_one(T.shape[0] + k, T.shape[0] + P.shape[0]),
            'q': probing.mask_one(T.shape[0] + q, T.shape[0] + P.shape[0]),
            's': probing.mask_one(s, T.shape[0] + P.shape[0]),
            'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
            'phase': 1
        })
    while q != pi[q] and P[q] != T[i]:
      q = pi[q]
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'pred_h': probing.strings_pred(T_pos, P_pos),
              'pi': probing.strings_pi(T_pos, P_pos, pi),
              'k': probing.mask_one(T.shape[0] + k, T.shape[0] + P.shape[0]),
              'q': probing.mask_one(T.shape[0] + q, T.shape[0] + P.shape[0]),
              's': probing.mask_one(s, T.shape[0] + P.shape[0]),
              'i': probing.mask_one(i, T.shape[0] + P.shape[0]),
              'phase': 1
          })
    if P[q] == T[i]:
      if q == P.shape[0] - 1:
        probing.push(
            probes,
            specs.Stage.OUTPUT,
            next_probe={'match': probing.mask_one(s, T.shape[0] + P.shape[0])})
        probing.finalize(probes)
        return s, probes
      q += 1

  # By convention, set probe to head of needle if no match is found
  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={
          'match': probing.mask_one(T.shape[0], T.shape[0] + P.shape[0])
      })
  probing.finalize(probes)

  return T.shape[0], probes
