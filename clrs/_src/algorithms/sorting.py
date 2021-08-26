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

"""Sorting algorithm generators.

Currently implements the following:
- Insertion sort
- Bubble sort
- Heapsort (Williams, 1964)
- Quicksort (Hoare, 1962)

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


def insertion_sort(A: _Array) -> _Out:
  """Insertion sort."""

  chex.assert_rank(A, 1)
  probes = probing.initialize(specs.SPECS['insertion_sort'])

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
          'i': probing.mask_one(0, A.shape[0]),
          'j': probing.mask_one(0, A.shape[0])
      })

  for j in range(1, A.shape[0]):
    key = A[j]
    # Insert A[j] into the sorted sequence A[1 .. j - 1]
    i = j - 1
    while i >= 0 and A[i] > key:
      A[i + 1] = A[i]
      A_pos[i + 1] = A_pos[i]
      i -= 1
    A[i + 1] = key
    stor_pos = A_pos[i + 1]
    A_pos[i + 1] = j

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.array(np.copy(A_pos)),
            'i': probing.mask_one(stor_pos, np.copy(A.shape[0])),
            'j': probing.mask_one(j, np.copy(A.shape[0]))
        })

  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={'pred': probing.array(np.copy(A_pos))})

  probing.finalize(probes)

  return A, probes


def bubble_sort(A: _Array) -> _Out:
  """Bubble sort."""

  chex.assert_rank(A, 1)
  probes = probing.initialize(specs.SPECS['bubble_sort'])

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
          'i': probing.mask_one(0, A.shape[0]),
          'j': probing.mask_one(0, A.shape[0])
      })

  for i in range(A.shape[0] - 1):
    for j in reversed(range(i + 1, A.shape[0])):
      if A[j] < A[j - 1]:
        tmp = A[j]
        A[j] = A[j - 1]
        A[j - 1] = tmp

        tmp = A_pos[j]
        A_pos[j] = A_pos[j - 1]
        A_pos[j - 1] = tmp

      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'pred_h': probing.array(np.copy(A_pos)),
              'i': probing.mask_one(A_pos[i], np.copy(A.shape[0])),
              'j': probing.mask_one(A_pos[j], np.copy(A.shape[0]))
          })

  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={'pred': probing.array(np.copy(A_pos))},
  )

  probing.finalize(probes)

  return A, probes


def heapsort(A: _Array) -> _Out:
  """Heapsort (Williams, 1964)."""

  chex.assert_rank(A, 1)
  probes = probing.initialize(specs.SPECS['heapsort'])

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
          'parent': probing.heap(np.copy(A_pos), A.shape[0]),
          'i': probing.mask_one(A.shape[0] - 1, A.shape[0]),
          'j': probing.mask_one(A.shape[0] - 1, A.shape[0]),
          'largest': probing.mask_one(A.shape[0] - 1, A.shape[0]),
          'heap_size': probing.mask_one(A.shape[0] - 1, A.shape[0]),
          'phase': probing.mask_one(0, 3)
      })

  def max_heapify(A, i, heap_size, ind, phase):
    l = 2 * i + 1
    r = 2 * i + 2
    if l < heap_size and A[l] > A[i]:
      largest = l
    else:
      largest = i
    if r < heap_size and A[r] > A[largest]:
      largest = r
    if largest != i:
      tmp = A[i]
      A[i] = A[largest]
      A[largest] = tmp

      tmp = A_pos[i]
      A_pos[i] = A_pos[largest]
      A_pos[largest] = tmp

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.array(np.copy(A_pos)),
            'parent': probing.heap(np.copy(A_pos), heap_size),
            'i': probing.mask_one(A_pos[ind], A.shape[0]),
            'j': probing.mask_one(A_pos[i], A.shape[0]),
            'largest': probing.mask_one(A_pos[largest], A.shape[0]),
            'heap_size': probing.mask_one(A_pos[heap_size - 1], A.shape[0]),
            'phase': probing.mask_one(phase, 3)
        })

    if largest != i:
      max_heapify(A, largest, heap_size, ind, phase)

  def build_max_heap(A):
    for i in reversed(range(A.shape[0])):
      max_heapify(A, i, A.shape[0], i, 0)

  build_max_heap(A)
  heap_size = A.shape[0]
  for i in reversed(range(1, A.shape[0])):
    tmp = A[0]
    A[0] = A[i]
    A[i] = tmp

    tmp = A_pos[0]
    A_pos[0] = A_pos[i]
    A_pos[i] = tmp

    heap_size -= 1

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.array(np.copy(A_pos)),
            'parent': probing.heap(np.copy(A_pos), heap_size),
            'i': probing.mask_one(A_pos[0], A.shape[0]),
            'j': probing.mask_one(A_pos[i], A.shape[0]),
            'largest': probing.mask_one(0, A.shape[0]),  # Consider masking
            'heap_size': probing.mask_one(A_pos[heap_size - 1], A.shape[0]),
            'phase': probing.mask_one(1, 3)
        })

    max_heapify(A, 0, heap_size, i, 2)  # reduce heap_size!

  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={'pred': probing.array(np.copy(A_pos))},
  )

  probing.finalize(probes)

  return A, probes


def quicksort(A: _Array, A_pos=None, p=None, r=None, probes=None) -> _Out:
  """Quicksort (Hoare, 1962)."""

  chex.assert_rank(A, 1)

  def partition(A, A_pos, p, r, probes):
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
              'j': probing.mask_one(A_pos[j], A.shape[0])
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
            'j': probing.mask_one(A_pos[r], A.shape[0])
        })

    return i + 1

  if A_pos is None:
    A_pos = np.arange(A.shape[0])
  if p is None:
    p = 0
  if r is None:
    r = len(A) - 1
  if probes is None:
    probes = probing.initialize(specs.SPECS['quicksort'])
    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'key': np.copy(A)
        })

  if p < r:
    q = partition(A, A_pos, p, r, probes)
    quicksort(A, A_pos, p, q - 1, probes)
    quicksort(A, A_pos, q + 1, r, probes)

  if p == 0 and r == len(A) - 1:
    probing.push(
        probes,
        specs.Stage.OUTPUT,
        next_probe={'pred': probing.array(np.copy(A_pos))},
    )
    probing.finalize(probes)

  return A, probes
