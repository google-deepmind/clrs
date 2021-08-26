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

"""Geometry algorithm generators.

Currently implements the following:
- Segment intersection
- Graham scan convex hull (Graham, 1972)
- Jarvis' march convex hull (Jarvis, 1973)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""
# pylint: disable=invalid-name

import math
from typing import Any, Tuple

import chex
from clrs._src import probing
from clrs._src import specs
import numpy as np


_Array = np.ndarray
_Out = Tuple[Any, probing.ProbesDict]


def segments_intersect(xs: _Array, ys: _Array) -> _Out:
  """Segment intersection."""

  assert xs.shape == (4,)
  assert ys.shape == (4,)
  probes = probing.initialize(specs.SPECS['segments_intersect'])

  A_pos = np.arange(xs.shape[0])
  dirs = np.zeros(xs.shape[0])
  on_seg = np.zeros(xs.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A_pos.shape[0],
          'x': np.copy(xs),
          'y': np.copy(ys)
      })

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'i': probing.mask_one(0, xs.shape[0]),
          'j': probing.mask_one(0, xs.shape[0]),
          'k': probing.mask_one(0, xs.shape[0]),
          'dir': np.copy(dirs),
          'on_seg': np.copy(on_seg)
      })

  def cross_product(x1, y1, x2, y2):
    return x1 * y2 - x2 * y1

  def direction(xs, ys, i, j, k):
    return cross_product(xs[k] - xs[i], ys[k] - ys[i], xs[j] - xs[i],
                         ys[j] - ys[i])

  def on_segment(xs, ys, i, j, k):
    if min(xs[i], xs[j]) <= xs[k] and xs[k] <= max(xs[i], xs[j]):
      if min(ys[i], ys[j]) <= ys[k] and ys[k] <= max(ys[i], ys[j]):
        return 1
    return 0

  dirs[0] = direction(xs, ys, 2, 3, 0)
  on_seg[0] = on_segment(xs, ys, 2, 3, 0)

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'i': probing.mask_one(2, xs.shape[0]),
          'j': probing.mask_one(3, xs.shape[0]),
          'k': probing.mask_one(0, xs.shape[0]),
          'dir': np.copy(dirs),
          'on_seg': np.copy(on_seg)
      })

  dirs[1] = direction(xs, ys, 2, 3, 1)
  on_seg[1] = on_segment(xs, ys, 2, 3, 1)

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'i': probing.mask_one(2, xs.shape[0]),
          'j': probing.mask_one(3, xs.shape[0]),
          'k': probing.mask_one(1, xs.shape[0]),
          'dir': np.copy(dirs),
          'on_seg': np.copy(on_seg)
      })

  dirs[2] = direction(xs, ys, 0, 1, 2)
  on_seg[2] = on_segment(xs, ys, 0, 1, 2)

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'i': probing.mask_one(0, xs.shape[0]),
          'j': probing.mask_one(1, xs.shape[0]),
          'k': probing.mask_one(2, xs.shape[0]),
          'dir': np.copy(dirs),
          'on_seg': np.copy(on_seg)
      })

  dirs[3] = direction(xs, ys, 0, 1, 3)
  on_seg[3] = on_segment(xs, ys, 0, 1, 3)

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'i': probing.mask_one(0, xs.shape[0]),
          'j': probing.mask_one(1, xs.shape[0]),
          'k': probing.mask_one(3, xs.shape[0]),
          'dir': np.copy(dirs),
          'on_seg': np.copy(on_seg)
      })

  ret = 0

  if ((dirs[0] > 0 and dirs[1] < 0) or
      (dirs[0] < 0 and dirs[1] > 0)) and ((dirs[2] > 0 and dirs[3] < 0) or
                                          (dirs[2] < 0 and dirs[3] > 0)):
    ret = 1
  elif dirs[0] == 0 and on_seg[0]:
    ret = 1
  elif dirs[1] == 0 and on_seg[1]:
    ret = 1
  elif dirs[2] == 0 and on_seg[2]:
    ret = 1
  elif dirs[3] == 0 and on_seg[3]:
    ret = 1

  probing.push(probes, specs.Stage.OUTPUT, next_probe={'intersect': ret})
  probing.finalize(probes)

  return ret, probes


def graham_scan(xs: _Array, ys: _Array) -> _Out:
  """Graham scan convex hull (Graham, 1972)."""

  chex.assert_rank([xs, ys], 1)
  probes = probing.initialize(specs.SPECS['graham_scan'])

  A_pos = np.arange(xs.shape[0])
  in_hull = np.zeros(xs.shape[0])
  stack_prev = np.arange(xs.shape[0])
  atans = np.zeros(xs.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A_pos.shape[0],
          'x': np.copy(xs),
          'y': np.copy(ys)
      })

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'best': probing.mask_one(0, xs.shape[0]),
          'atans': np.copy(atans),
          'in_hull_h': np.copy(in_hull),
          'stack_prev': np.copy(stack_prev),
          'last_stack': probing.mask_one(0, xs.shape[0]),
          'i': probing.mask_one(0, xs.shape[0]),
          'phase': probing.mask_one(0, 5)
      })

  def counter_clockwise(xs, ys, i, j, k):
    return ((xs[j] - xs[i]) * (ys[k] - ys[i]) - (ys[j] - ys[i]) *
            (xs[k] - xs[i])) <= 0

  best = 0
  for i in range(xs.shape[0]):
    if ys[i] < ys[best] or (ys[i] == ys[best] and xs[i] < xs[best]):
      best = i

  in_hull[best] = 1
  last_stack = best

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'best': probing.mask_one(best, xs.shape[0]),
          'atans': np.copy(atans),
          'in_hull_h': np.copy(in_hull),
          'stack_prev': np.copy(stack_prev),
          'last_stack': probing.mask_one(last_stack, xs.shape[0]),
          'i': probing.mask_one(best, xs.shape[0]),
          'phase': probing.mask_one(1, 5)
      })

  for i in range(xs.shape[0]):
    if i != best:
      atans[i] = math.atan2(ys[i] - ys[best], xs[i] - xs[best])
  atans[best] = -123456789
  ind = np.argsort(atans)
  atans[best] = 0

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'best': probing.mask_one(best, xs.shape[0]),
          'atans': np.copy(atans),
          'in_hull_h': np.copy(in_hull),
          'stack_prev': np.copy(stack_prev),
          'last_stack': probing.mask_one(last_stack, xs.shape[0]),
          'i': probing.mask_one(best, xs.shape[0]),
          'phase': probing.mask_one(2, 5)
      })

  for i in range(1, xs.shape[0]):
    if i >= 3:
      while counter_clockwise(xs, ys, stack_prev[last_stack], last_stack,
                              ind[i]):
        prev_last = last_stack
        last_stack = stack_prev[last_stack]
        stack_prev[prev_last] = prev_last
        in_hull[prev_last] = 0
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'best': probing.mask_one(best, xs.shape[0]),
                'atans': np.copy(atans),
                'in_hull_h': np.copy(in_hull),
                'stack_prev': np.copy(stack_prev),
                'last_stack': probing.mask_one(last_stack, xs.shape[0]),
                'i': probing.mask_one(A_pos[ind[i]], xs.shape[0]),
                'phase': probing.mask_one(3, 5)
            })

    in_hull[ind[i]] = 1
    stack_prev[ind[i]] = last_stack
    last_stack = ind[i]

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'best': probing.mask_one(best, xs.shape[0]),
            'atans': np.copy(atans),
            'in_hull_h': np.copy(in_hull),
            'stack_prev': np.copy(stack_prev),
            'last_stack': probing.mask_one(last_stack, xs.shape[0]),
            'i': probing.mask_one(A_pos[ind[i]], xs.shape[0]),
            'phase': probing.mask_one(4, 5)
        })

  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={'in_hull': np.copy(in_hull)},
  )
  probing.finalize(probes)

  return in_hull, probes


def jarvis_march(xs: _Array, ys: _Array) -> _Out:
  """Jarvis' march convex hull (Jarvis, 1973)."""

  chex.assert_rank([xs, ys], 1)
  probes = probing.initialize(specs.SPECS['jarvis_march'])

  A_pos = np.arange(xs.shape[0])
  in_hull = np.zeros(xs.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A_pos.shape[0],
          'x': np.copy(xs),
          'y': np.copy(ys)
      })

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pred_h': probing.array(np.copy(A_pos)),
          'in_hull_h': np.copy(in_hull),
          'best': probing.mask_one(0, xs.shape[0]),
          'last_point': probing.mask_one(0, xs.shape[0]),
          'endpoint': probing.mask_one(0, xs.shape[0]),
          'i': probing.mask_one(0, xs.shape[0]),
          'phase': probing.mask_one(0, 2)
      })

  def counter_clockwise(xs, ys, i, j, k):
    if (k == i) or (k == j):
      return False
    return ((xs[j] - xs[i]) * (ys[k] - ys[i]) - (ys[j] - ys[i]) *
            (xs[k] - xs[i])) <= 0

  best = 0
  for i in range(xs.shape[0]):
    if ys[i] < ys[best] or (ys[i] == ys[best] and xs[i] < xs[best]):
      best = i

  in_hull[best] = 1
  last_point = best
  endpoint = 0

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pred_h': probing.array(np.copy(A_pos)),
          'in_hull_h': np.copy(in_hull),
          'best': probing.mask_one(best, xs.shape[0]),
          'last_point': probing.mask_one(last_point, xs.shape[0]),
          'endpoint': probing.mask_one(endpoint, xs.shape[0]),
          'i': probing.mask_one(0, xs.shape[0]),
          'phase': probing.mask_one(1, 2)
      })

  while True:
    for i in range(xs.shape[0]):
      if endpoint == last_point or counter_clockwise(xs, ys, last_point,
                                                     endpoint, i):
        endpoint = i
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'pred_h': probing.array(np.copy(A_pos)),
              'in_hull_h': np.copy(in_hull),
              'best': probing.mask_one(best, xs.shape[0]),
              'last_point': probing.mask_one(last_point, xs.shape[0]),
              'endpoint': probing.mask_one(endpoint, xs.shape[0]),
              'i': probing.mask_one(i, xs.shape[0]),
              'phase': probing.mask_one(1, 2)
          })
    if in_hull[endpoint] > 0:
      break
    in_hull[endpoint] = 1
    last_point = endpoint
    endpoint = 0
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pred_h': probing.array(np.copy(A_pos)),
            'in_hull_h': np.copy(in_hull),
            'best': probing.mask_one(best, xs.shape[0]),
            'last_point': probing.mask_one(last_point, xs.shape[0]),
            'endpoint': probing.mask_one(endpoint, xs.shape[0]),
            'i': probing.mask_one(0, xs.shape[0]),
            'phase': probing.mask_one(1, 2)
        })

  probing.push(
      probes, specs.Stage.OUTPUT, next_probe={'in_hull': np.copy(in_hull)})
  probing.finalize(probes)

  return in_hull, probes
