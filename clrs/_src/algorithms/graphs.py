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

"""Graph algorithm generators.

Currently implements the following:
- Depth-first search (Moore, 1959)
- Breadth-first search (Moore, 1959)
- Topological sorting (Knuth, 1973)
- Articulation points
- Bridges
- Kosaraju's strongly-connected components (Aho et al., 1974)
- Kruskal's minimum spanning tree (Kruskal, 1956)
- Prim's minimum spanning tree (Prim, 1957)
- Bellman-Ford's single-source shortest path (Bellman, 1958)
- Dijkstra's single-source shortest path (Dijkstra, 1959)
- DAG shortest path
- Floyd-Warshall's all-pairs shortest paths (Floyd, 1962)
- Edmonds-Karp bipartite matching (Edmund & Karp, 1972)

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
_OutputClass = specs.OutputClass


def dfs(A: _Array) -> _Out:
  """Depth-first search (Moore, 1959)."""

  chex.assert_rank(A, 2)
  probes = probing.initialize(specs.SPECS['dfs'])

  A_pos = np.arange(A.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          'A': np.copy(A),
          'adj': probing.graph(np.copy(A))
      })

  color = np.zeros(A.shape[0], dtype=np.int32)
  pi = np.arange(A.shape[0])
  d = np.zeros(A.shape[0])
  f = np.zeros(A.shape[0])
  s_prev = np.arange(A.shape[0])
  time = 0
  for s in range(A.shape[0]):
    if color[s] == 0:
      s_last = s
      u = s
      v = s
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'pi_h': np.copy(pi),
              'color': probing.array_cat(color, 3),
              'd': np.copy(d),
              'f': np.copy(f),
              's_prev': np.copy(s_prev),
              's': probing.mask_one(s, A.shape[0]),
              'u': probing.mask_one(u, A.shape[0]),
              'v': probing.mask_one(v, A.shape[0]),
              's_last': probing.mask_one(s_last, A.shape[0]),
              'time': time
          })
      while True:
        if color[u] == 0 or d[u] == 0.0:
          time += 0.01
          d[u] = time
          color[u] = 1
          probing.push(
              probes,
              specs.Stage.HINT,
              next_probe={
                  'pi_h': np.copy(pi),
                  'color': probing.array_cat(color, 3),
                  'd': np.copy(d),
                  'f': np.copy(f),
                  's_prev': np.copy(s_prev),
                  's': probing.mask_one(s, A.shape[0]),
                  'u': probing.mask_one(u, A.shape[0]),
                  'v': probing.mask_one(v, A.shape[0]),
                  's_last': probing.mask_one(s_last, A.shape[0]),
                  'time': time
              })

        for v in range(A.shape[0]):
          if A[u, v] != 0:
            if color[v] == 0:
              pi[v] = u
              color[v] = 1
              s_prev[v] = s_last
              s_last = v

              probing.push(
                  probes,
                  specs.Stage.HINT,
                  next_probe={
                      'pi_h': np.copy(pi),
                      'color': probing.array_cat(color, 3),
                      'd': np.copy(d),
                      'f': np.copy(f),
                      's_prev': np.copy(s_prev),
                      's': probing.mask_one(s, A.shape[0]),
                      'u': probing.mask_one(u, A.shape[0]),
                      'v': probing.mask_one(v, A.shape[0]),
                      's_last': probing.mask_one(s_last, A.shape[0]),
                      'time': time
                  })
              break

        if s_last == u:
          color[u] = 2
          time += 0.01
          f[u] = time

          probing.push(
              probes,
              specs.Stage.HINT,
              next_probe={
                  'pi_h': np.copy(pi),
                  'color': probing.array_cat(color, 3),
                  'd': np.copy(d),
                  'f': np.copy(f),
                  's_prev': np.copy(s_prev),
                  's': probing.mask_one(s, A.shape[0]),
                  'u': probing.mask_one(u, A.shape[0]),
                  'v': probing.mask_one(v, A.shape[0]),
                  's_last': probing.mask_one(s_last, A.shape[0]),
                  'time': time
              })

          if s_prev[u] == u:
            assert s_prev[s_last] == s_last
            break
          pr = s_prev[s_last]
          s_prev[s_last] = s_last
          s_last = pr

        u = s_last

  probing.push(probes, specs.Stage.OUTPUT, next_probe={'pi': np.copy(pi)})
  probing.finalize(probes)

  return pi, probes


def bfs(A: _Array, s: int) -> _Out:
  """Breadth-first search (Moore, 1959)."""

  chex.assert_rank(A, 2)
  probes = probing.initialize(specs.SPECS['bfs'])

  A_pos = np.arange(A.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          's': probing.mask_one(s, A.shape[0]),
          'A': np.copy(A),
          'adj': probing.graph(np.copy(A))
      })

  reach = np.zeros(A.shape[0])
  pi = np.arange(A.shape[0])
  reach[s] = 1
  while True:
    prev_reach = np.copy(reach)
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'reach_h': np.copy(prev_reach),
            'pi_h': np.copy(pi)
        })
    for i in range(A.shape[0]):
      for j in range(A.shape[0]):
        if A[i, j] > 0 and prev_reach[i] == 1:
          if pi[j] == j and j != s:
            pi[j] = i
          reach[j] = 1
    if np.all(reach == prev_reach):
      break

  probing.push(probes, specs.Stage.OUTPUT, next_probe={'pi': np.copy(pi)})
  probing.finalize(probes)

  return pi, probes


def topological_sort(A: _Array) -> _Out:
  """Topological sorting (Knuth, 1973)."""

  chex.assert_rank(A, 2)
  probes = probing.initialize(specs.SPECS['topological_sort'])

  A_pos = np.arange(A.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          'A': np.copy(A),
          'adj': probing.graph(np.copy(A))
      })

  color = np.zeros(A.shape[0], dtype=np.int32)
  topo = np.arange(A.shape[0])
  s_prev = np.arange(A.shape[0])
  topo_head = 0
  for s in range(A.shape[0]):
    if color[s] == 0:
      s_last = s
      u = s
      v = s
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'topo_h': np.copy(topo),
              'topo_head_h': probing.mask_one(topo_head, A.shape[0]),
              'color': probing.array_cat(color, 3),
              's_prev': np.copy(s_prev),
              's': probing.mask_one(s, A.shape[0]),
              'u': probing.mask_one(u, A.shape[0]),
              'v': probing.mask_one(v, A.shape[0]),
              's_last': probing.mask_one(s_last, A.shape[0])
          })
      while True:
        if color[u] == 0:
          color[u] = 1
          probing.push(
              probes,
              specs.Stage.HINT,
              next_probe={
                  'topo_h': np.copy(topo),
                  'topo_head_h': probing.mask_one(topo_head, A.shape[0]),
                  'color': probing.array_cat(color, 3),
                  's_prev': np.copy(s_prev),
                  's': probing.mask_one(s, A.shape[0]),
                  'u': probing.mask_one(u, A.shape[0]),
                  'v': probing.mask_one(v, A.shape[0]),
                  's_last': probing.mask_one(s_last, A.shape[0])
              })

        for v in range(A.shape[0]):
          if A[u, v] != 0:
            if color[v] == 0:
              color[v] = 1
              s_prev[v] = s_last
              s_last = v

              probing.push(
                  probes,
                  specs.Stage.HINT,
                  next_probe={
                      'topo_h': np.copy(topo),
                      'topo_head_h': probing.mask_one(topo_head, A.shape[0]),
                      'color': probing.array_cat(color, 3),
                      's_prev': np.copy(s_prev),
                      's': probing.mask_one(s, A.shape[0]),
                      'u': probing.mask_one(u, A.shape[0]),
                      'v': probing.mask_one(v, A.shape[0]),
                      's_last': probing.mask_one(s_last, A.shape[0])
                  })
              break

        if s_last == u:
          color[u] = 2

          if color[topo_head] == 2:
            topo[u] = topo_head
          topo_head = u

          probing.push(
              probes,
              specs.Stage.HINT,
              next_probe={
                  'topo_h': np.copy(topo),
                  'topo_head_h': probing.mask_one(topo_head, A.shape[0]),
                  'color': probing.array_cat(color, 3),
                  's_prev': np.copy(s_prev),
                  's': probing.mask_one(s, A.shape[0]),
                  'u': probing.mask_one(u, A.shape[0]),
                  'v': probing.mask_one(v, A.shape[0]),
                  's_last': probing.mask_one(s_last, A.shape[0])
              })

          if s_prev[u] == u:
            assert s_prev[s_last] == s_last
            break
          pr = s_prev[s_last]
          s_prev[s_last] = s_last
          s_last = pr

        u = s_last

  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={
          'topo': np.copy(topo),
          'topo_head': probing.mask_one(topo_head, A.shape[0])
      })
  probing.finalize(probes)

  return topo, probes


def articulation_points(A: _Array) -> _Out:
  """Articulation points."""

  chex.assert_rank(A, 2)
  probes = probing.initialize(specs.SPECS['articulation_points'])

  A_pos = np.arange(A.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          'A': np.copy(A),
          'adj': probing.graph(np.copy(A))
      })

  color = np.zeros(A.shape[0], dtype=np.int32)
  pi = np.arange(A.shape[0])
  d = np.zeros(A.shape[0])
  f = np.zeros(A.shape[0])
  s_prev = np.arange(A.shape[0])
  time = 0

  low = np.zeros(A.shape[0])
  child_cnt = np.zeros(A.shape[0])
  is_cut = np.zeros(A.shape[0])

  for s in range(A.shape[0]):
    if color[s] == 0:
      s_last = s
      u = s
      v = s
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'is_cut_h': np.copy(is_cut),
              'pi_h': np.copy(pi),
              'color': probing.array_cat(color, 3),
              'd': np.copy(d),
              'f': np.copy(f),
              'low': np.copy(low),
              'child_cnt': np.copy(child_cnt),
              's_prev': np.copy(s_prev),
              's': probing.mask_one(s, A.shape[0]),
              'u': probing.mask_one(u, A.shape[0]),
              'v': probing.mask_one(v, A.shape[0]),
              's_last': probing.mask_one(s_last, A.shape[0]),
              'time': time
          })
      while True:
        if color[u] == 0 or d[u] == 0.0:
          time += 0.01
          d[u] = time
          low[u] = time
          color[u] = 1
          probing.push(
              probes,
              specs.Stage.HINT,
              next_probe={
                  'is_cut_h': np.copy(is_cut),
                  'pi_h': np.copy(pi),
                  'color': probing.array_cat(color, 3),
                  'd': np.copy(d),
                  'f': np.copy(f),
                  'low': np.copy(low),
                  'child_cnt': np.copy(child_cnt),
                  's_prev': np.copy(s_prev),
                  's': probing.mask_one(s, A.shape[0]),
                  'u': probing.mask_one(u, A.shape[0]),
                  'v': probing.mask_one(v, A.shape[0]),
                  's_last': probing.mask_one(s_last, A.shape[0]),
                  'time': time
              })

        for v in range(A.shape[0]):
          if A[u, v] != 0:
            if color[v] == 0:
              pi[v] = u
              color[v] = 1
              s_prev[v] = s_last
              s_last = v
              child_cnt[u] += 0.01

              probing.push(
                  probes,
                  specs.Stage.HINT,
                  next_probe={
                      'is_cut_h': np.copy(is_cut),
                      'pi_h': np.copy(pi),
                      'color': probing.array_cat(color, 3),
                      'd': np.copy(d),
                      'f': np.copy(f),
                      'low': np.copy(low),
                      'child_cnt': np.copy(child_cnt),
                      's_prev': np.copy(s_prev),
                      's': probing.mask_one(s, A.shape[0]),
                      'u': probing.mask_one(u, A.shape[0]),
                      'v': probing.mask_one(v, A.shape[0]),
                      's_last': probing.mask_one(s_last, A.shape[0]),
                      'time': time
                  })
              break
            elif v != pi[u]:
              low[u] = min(low[u], d[v])
              probing.push(
                  probes,
                  specs.Stage.HINT,
                  next_probe={
                      'is_cut_h': np.copy(is_cut),
                      'pi_h': np.copy(pi),
                      'color': probing.array_cat(color, 3),
                      'd': np.copy(d),
                      'f': np.copy(f),
                      'low': np.copy(low),
                      'child_cnt': np.copy(child_cnt),
                      's_prev': np.copy(s_prev),
                      's': probing.mask_one(s, A.shape[0]),
                      'u': probing.mask_one(u, A.shape[0]),
                      'v': probing.mask_one(v, A.shape[0]),
                      's_last': probing.mask_one(s_last, A.shape[0]),
                      'time': time
                  })

        if s_last == u:
          color[u] = 2
          time += 0.01
          f[u] = time

          for v in range(A.shape[0]):
            if pi[v] == u:
              low[u] = min(low[u], low[v])
              if pi[u] != u and low[v] >= d[u]:
                is_cut[u] = 1
          if pi[u] == u and child_cnt[u] > 0.01:
            is_cut[u] = 1

          probing.push(
              probes,
              specs.Stage.HINT,
              next_probe={
                  'is_cut_h': np.copy(is_cut),
                  'pi_h': np.copy(pi),
                  'color': probing.array_cat(color, 3),
                  'd': np.copy(d),
                  'f': np.copy(f),
                  'low': np.copy(low),
                  'child_cnt': np.copy(child_cnt),
                  's_prev': np.copy(s_prev),
                  's': probing.mask_one(s, A.shape[0]),
                  'u': probing.mask_one(u, A.shape[0]),
                  'v': probing.mask_one(v, A.shape[0]),
                  's_last': probing.mask_one(s_last, A.shape[0]),
                  'time': time
              })

          if s_prev[u] == u:
            assert s_prev[s_last] == s_last
            break
          pr = s_prev[s_last]
          s_prev[s_last] = s_last
          s_last = pr

        u = s_last

  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={'is_cut': np.copy(is_cut)},
  )
  probing.finalize(probes)

  return is_cut, probes


def bridges(A: _Array) -> _Out:
  """Bridges."""

  chex.assert_rank(A, 2)
  probes = probing.initialize(specs.SPECS['bridges'])

  A_pos = np.arange(A.shape[0])
  adj = probing.graph(np.copy(A))

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          'A': np.copy(A),
          'adj': adj
      })

  color = np.zeros(A.shape[0], dtype=np.int32)
  pi = np.arange(A.shape[0])
  d = np.zeros(A.shape[0])
  f = np.zeros(A.shape[0])
  s_prev = np.arange(A.shape[0])
  time = 0

  low = np.zeros(A.shape[0])
  is_bridge = (
      np.zeros((A.shape[0], A.shape[0])) + _OutputClass.MASKED.value + adj)

  for s in range(A.shape[0]):
    if color[s] == 0:
      s_last = s
      u = s
      v = s
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'is_bridge_h': np.copy(is_bridge),
              'pi_h': np.copy(pi),
              'color': probing.array_cat(color, 3),
              'd': np.copy(d),
              'f': np.copy(f),
              'low': np.copy(low),
              's_prev': np.copy(s_prev),
              's': probing.mask_one(s, A.shape[0]),
              'u': probing.mask_one(u, A.shape[0]),
              'v': probing.mask_one(v, A.shape[0]),
              's_last': probing.mask_one(s_last, A.shape[0]),
              'time': time
          })
      while True:
        if color[u] == 0 or d[u] == 0.0:
          time += 0.01
          d[u] = time
          low[u] = time
          color[u] = 1
          probing.push(
              probes,
              specs.Stage.HINT,
              next_probe={
                  'is_bridge_h': np.copy(is_bridge),
                  'pi_h': np.copy(pi),
                  'color': probing.array_cat(color, 3),
                  'd': np.copy(d),
                  'f': np.copy(f),
                  'low': np.copy(low),
                  's_prev': np.copy(s_prev),
                  's': probing.mask_one(s, A.shape[0]),
                  'u': probing.mask_one(u, A.shape[0]),
                  'v': probing.mask_one(v, A.shape[0]),
                  's_last': probing.mask_one(s_last, A.shape[0]),
                  'time': time
              })

        for v in range(A.shape[0]):
          if A[u, v] != 0:
            if color[v] == 0:
              pi[v] = u
              color[v] = 1
              s_prev[v] = s_last
              s_last = v

              probing.push(
                  probes,
                  specs.Stage.HINT,
                  next_probe={
                      'is_bridge_h': np.copy(is_bridge),
                      'pi_h': np.copy(pi),
                      'color': probing.array_cat(color, 3),
                      'd': np.copy(d),
                      'f': np.copy(f),
                      'low': np.copy(low),
                      's_prev': np.copy(s_prev),
                      's': probing.mask_one(s, A.shape[0]),
                      'u': probing.mask_one(u, A.shape[0]),
                      'v': probing.mask_one(v, A.shape[0]),
                      's_last': probing.mask_one(s_last, A.shape[0]),
                      'time': time
                  })
              break
            elif v != pi[u]:
              low[u] = min(low[u], d[v])
              probing.push(
                  probes,
                  specs.Stage.HINT,
                  next_probe={
                      'is_bridge_h': np.copy(is_bridge),
                      'pi_h': np.copy(pi),
                      'color': probing.array_cat(color, 3),
                      'd': np.copy(d),
                      'f': np.copy(f),
                      'low': np.copy(low),
                      's_prev': np.copy(s_prev),
                      's': probing.mask_one(s, A.shape[0]),
                      'u': probing.mask_one(u, A.shape[0]),
                      'v': probing.mask_one(v, A.shape[0]),
                      's_last': probing.mask_one(s_last, A.shape[0]),
                      'time': time
                  })

        if s_last == u:
          color[u] = 2
          time += 0.01
          f[u] = time

          for v in range(A.shape[0]):
            if pi[v] == u:
              low[u] = min(low[u], low[v])
              if low[v] > d[u]:
                is_bridge[u, v] = 1
                is_bridge[v, u] = 1

          probing.push(
              probes,
              specs.Stage.HINT,
              next_probe={
                  'is_bridge_h': np.copy(is_bridge),
                  'pi_h': np.copy(pi),
                  'color': probing.array_cat(color, 3),
                  'd': np.copy(d),
                  'f': np.copy(f),
                  'low': np.copy(low),
                  's_prev': np.copy(s_prev),
                  's': probing.mask_one(s, A.shape[0]),
                  'u': probing.mask_one(u, A.shape[0]),
                  'v': probing.mask_one(v, A.shape[0]),
                  's_last': probing.mask_one(s_last, A.shape[0]),
                  'time': time
              })

          if s_prev[u] == u:
            assert s_prev[s_last] == s_last
            break
          pr = s_prev[s_last]
          s_prev[s_last] = s_last
          s_last = pr

        u = s_last

  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={'is_bridge': np.copy(is_bridge)},
  )
  probing.finalize(probes)

  return is_bridge, probes


def strongly_connected_components(A: _Array) -> _Out:
  """Kosaraju's strongly-connected components (Aho et al., 1974)."""

  chex.assert_rank(A, 2)
  probes = probing.initialize(
      specs.SPECS['strongly_connected_components'])

  A_pos = np.arange(A.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          'A': np.copy(A),
          'adj': probing.graph(np.copy(A))
      })

  scc_id = np.arange(A.shape[0])
  color = np.zeros(A.shape[0], dtype=np.int32)
  d = np.zeros(A.shape[0])
  f = np.zeros(A.shape[0])
  s_prev = np.arange(A.shape[0])
  time = 0
  A_t = np.transpose(A)

  for s in range(A.shape[0]):
    if color[s] == 0:
      s_last = s
      u = s
      v = s
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'scc_id_h': np.copy(scc_id),
              'A_t': probing.graph(np.copy(A_t)),
              'color': probing.array_cat(color, 3),
              'd': np.copy(d),
              'f': np.copy(f),
              's_prev': np.copy(s_prev),
              's': probing.mask_one(s, A.shape[0]),
              'u': probing.mask_one(u, A.shape[0]),
              'v': probing.mask_one(v, A.shape[0]),
              's_last': probing.mask_one(s_last, A.shape[0]),
              'time': time,
              'phase': 0
          })
      while True:
        if color[u] == 0 or d[u] == 0.0:
          time += 0.01
          d[u] = time
          color[u] = 1
          probing.push(
              probes,
              specs.Stage.HINT,
              next_probe={
                  'scc_id_h': np.copy(scc_id),
                  'A_t': probing.graph(np.copy(A_t)),
                  'color': probing.array_cat(color, 3),
                  'd': np.copy(d),
                  'f': np.copy(f),
                  's_prev': np.copy(s_prev),
                  's': probing.mask_one(s, A.shape[0]),
                  'u': probing.mask_one(u, A.shape[0]),
                  'v': probing.mask_one(v, A.shape[0]),
                  's_last': probing.mask_one(s_last, A.shape[0]),
                  'time': time,
                  'phase': 0
              })
        for v in range(A.shape[0]):
          if A[u, v] != 0:
            if color[v] == 0:
              color[v] = 1
              s_prev[v] = s_last
              s_last = v
              probing.push(
                  probes,
                  specs.Stage.HINT,
                  next_probe={
                      'scc_id_h': np.copy(scc_id),
                      'A_t': probing.graph(np.copy(A_t)),
                      'color': probing.array_cat(color, 3),
                      'd': np.copy(d),
                      'f': np.copy(f),
                      's_prev': np.copy(s_prev),
                      's': probing.mask_one(s, A.shape[0]),
                      'u': probing.mask_one(u, A.shape[0]),
                      'v': probing.mask_one(v, A.shape[0]),
                      's_last': probing.mask_one(s_last, A.shape[0]),
                      'time': time,
                      'phase': 0
                  })
              break

        if s_last == u:
          color[u] = 2
          time += 0.01
          f[u] = time

          probing.push(
              probes,
              specs.Stage.HINT,
              next_probe={
                  'scc_id_h': np.copy(scc_id),
                  'A_t': probing.graph(np.copy(A_t)),
                  'color': probing.array_cat(color, 3),
                  'd': np.copy(d),
                  'f': np.copy(f),
                  's_prev': np.copy(s_prev),
                  's': probing.mask_one(s, A.shape[0]),
                  'u': probing.mask_one(u, A.shape[0]),
                  'v': probing.mask_one(v, A.shape[0]),
                  's_last': probing.mask_one(s_last, A.shape[0]),
                  'time': time,
                  'phase': 0
              })

          if s_prev[u] == u:
            assert s_prev[s_last] == s_last
            break
          pr = s_prev[s_last]
          s_prev[s_last] = s_last
          s_last = pr

        u = s_last

  color = np.zeros(A.shape[0], dtype=np.int32)
  s_prev = np.arange(A.shape[0])

  for s in np.argsort(-f):
    if color[s] == 0:
      s_last = s
      u = s
      v = s
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'scc_id_h': np.copy(scc_id),
              'A_t': probing.graph(np.copy(A_t)),
              'color': probing.array_cat(color, 3),
              'd': np.copy(d),
              'f': np.copy(f),
              's_prev': np.copy(s_prev),
              's': probing.mask_one(s, A.shape[0]),
              'u': probing.mask_one(u, A.shape[0]),
              'v': probing.mask_one(v, A.shape[0]),
              's_last': probing.mask_one(s_last, A.shape[0]),
              'time': time,
              'phase': 1
          })
      while True:
        scc_id[u] = s
        if color[u] == 0 or d[u] == 0.0:
          time += 0.01
          d[u] = time
          color[u] = 1
          probing.push(
              probes,
              specs.Stage.HINT,
              next_probe={
                  'scc_id_h': np.copy(scc_id),
                  'A_t': probing.graph(np.copy(A_t)),
                  'color': probing.array_cat(color, 3),
                  'd': np.copy(d),
                  'f': np.copy(f),
                  's_prev': np.copy(s_prev),
                  's': probing.mask_one(s, A.shape[0]),
                  'u': probing.mask_one(u, A.shape[0]),
                  'v': probing.mask_one(v, A.shape[0]),
                  's_last': probing.mask_one(s_last, A.shape[0]),
                  'time': time,
                  'phase': 1
              })
        for v in range(A.shape[0]):
          if A_t[u, v] != 0:
            if color[v] == 0:
              color[v] = 1
              s_prev[v] = s_last
              s_last = v
              probing.push(
                  probes,
                  specs.Stage.HINT,
                  next_probe={
                      'scc_id_h': np.copy(scc_id),
                      'A_t': probing.graph(np.copy(A_t)),
                      'color': probing.array_cat(color, 3),
                      'd': np.copy(d),
                      'f': np.copy(f),
                      's_prev': np.copy(s_prev),
                      's': probing.mask_one(s, A.shape[0]),
                      'u': probing.mask_one(u, A.shape[0]),
                      'v': probing.mask_one(v, A.shape[0]),
                      's_last': probing.mask_one(s_last, A.shape[0]),
                      'time': time,
                      'phase': 1
                  })
              break

        if s_last == u:
          color[u] = 2
          time += 0.01
          f[u] = time

          probing.push(
              probes,
              specs.Stage.HINT,
              next_probe={
                  'scc_id_h': np.copy(scc_id),
                  'A_t': probing.graph(np.copy(A_t)),
                  'color': probing.array_cat(color, 3),
                  'd': np.copy(d),
                  'f': np.copy(f),
                  's_prev': np.copy(s_prev),
                  's': probing.mask_one(s, A.shape[0]),
                  'u': probing.mask_one(u, A.shape[0]),
                  'v': probing.mask_one(v, A.shape[0]),
                  's_last': probing.mask_one(s_last, A.shape[0]),
                  'time': time,
                  'phase': 1
              })

          if s_prev[u] == u:
            assert s_prev[s_last] == s_last
            break
          pr = s_prev[s_last]
          s_prev[s_last] = s_last
          s_last = pr

        u = s_last

  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={'scc_id': np.copy(scc_id)},
  )
  probing.finalize(probes)

  return scc_id, probes


def mst_kruskal(A: _Array) -> _Out:
  """Kruskal's minimum spanning tree (Kruskal, 1956)."""

  chex.assert_rank(A, 2)
  probes = probing.initialize(specs.SPECS['mst_kruskal'])

  A_pos = np.arange(A.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          'A': np.copy(A),
          'adj': probing.graph(np.copy(A))
      })

  pi = np.arange(A.shape[0])

  def mst_union(u, v, in_mst, probes):
    root_u = u
    root_v = v

    mask_u = np.zeros(in_mst.shape[0])
    mask_v = np.zeros(in_mst.shape[0])

    mask_u[u] = 1
    mask_v[v] = 1

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'in_mst_h': np.copy(in_mst),
            'pi': np.copy(pi),
            'u': probing.mask_one(u, A.shape[0]),
            'v': probing.mask_one(v, A.shape[0]),
            'root_u': probing.mask_one(root_u, A.shape[0]),
            'root_v': probing.mask_one(root_v, A.shape[0]),
            'mask_u': np.copy(mask_u),
            'mask_v': np.copy(mask_v),
            'phase': probing.mask_one(1, 3)
        })

    while pi[root_u] != root_u:
      root_u = pi[root_u]
      for i in range(mask_u.shape[0]):
        if mask_u[i] == 1:
          pi[i] = root_u
      mask_u[root_u] = 1
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'in_mst_h': np.copy(in_mst),
              'pi': np.copy(pi),
              'u': probing.mask_one(u, A.shape[0]),
              'v': probing.mask_one(v, A.shape[0]),
              'root_u': probing.mask_one(root_u, A.shape[0]),
              'root_v': probing.mask_one(root_v, A.shape[0]),
              'mask_u': np.copy(mask_u),
              'mask_v': np.copy(mask_v),
              'phase': probing.mask_one(1, 3)
          })

    while pi[root_v] != root_v:
      root_v = pi[root_v]
      for i in range(mask_v.shape[0]):
        if mask_v[i] == 1:
          pi[i] = root_v
      mask_v[root_v] = 1
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'in_mst_h': np.copy(in_mst),
              'pi': np.copy(pi),
              'u': probing.mask_one(u, A.shape[0]),
              'v': probing.mask_one(v, A.shape[0]),
              'root_u': probing.mask_one(root_u, A.shape[0]),
              'root_v': probing.mask_one(root_v, A.shape[0]),
              'mask_u': np.copy(mask_u),
              'mask_v': np.copy(mask_v),
              'phase': probing.mask_one(2, 3)
          })

    if root_u < root_v:
      in_mst[u, v] = 1
      in_mst[v, u] = 1
      pi[root_u] = root_v
    elif root_u > root_v:
      in_mst[u, v] = 1
      in_mst[v, u] = 1
      pi[root_v] = root_u
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'in_mst_h': np.copy(in_mst),
            'pi': np.copy(pi),
            'u': probing.mask_one(u, A.shape[0]),
            'v': probing.mask_one(v, A.shape[0]),
            'root_u': probing.mask_one(root_u, A.shape[0]),
            'root_v': probing.mask_one(root_v, A.shape[0]),
            'mask_u': np.copy(mask_u),
            'mask_v': np.copy(mask_v),
            'phase': probing.mask_one(0, 3)
        })

  in_mst = np.zeros((A.shape[0], A.shape[0]))

  # Prep to sort edge array
  lx = []
  ly = []
  wts = []
  for i in range(A.shape[0]):
    for j in range(i + 1, A.shape[0]):
      if A[i, j] > 0:
        lx.append(i)
        ly.append(j)
        wts.append(A[i, j])

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'in_mst_h': np.copy(in_mst),
          'pi': np.copy(pi),
          'u': probing.mask_one(0, A.shape[0]),
          'v': probing.mask_one(0, A.shape[0]),
          'root_u': probing.mask_one(0, A.shape[0]),
          'root_v': probing.mask_one(0, A.shape[0]),
          'mask_u': np.zeros(A.shape[0]),
          'mask_v': np.zeros(A.shape[0]),
          'phase': probing.mask_one(0, 3)
      })
  for ind in np.argsort(wts):
    u = lx[ind]
    v = ly[ind]
    mst_union(u, v, in_mst, probes)

  probing.push(
      probes,
      specs.Stage.OUTPUT,
      next_probe={'in_mst': np.copy(in_mst)},
  )
  probing.finalize(probes)

  return in_mst, probes


def mst_prim(A: _Array, s: int) -> _Out:
  """Prim's minimum spanning tree (Prim, 1957)."""

  chex.assert_rank(A, 2)
  probes = probing.initialize(specs.SPECS['mst_prim'])

  A_pos = np.arange(A.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          's': probing.mask_one(s, A.shape[0]),
          'A': np.copy(A),
          'adj': probing.graph(np.copy(A))
      })

  key = np.zeros(A.shape[0])
  mark = np.zeros(A.shape[0])
  in_queue = np.zeros(A.shape[0])
  pi = np.arange(A.shape[0])
  key[s] = 0
  in_queue[s] = 1

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pi_h': np.copy(pi),
          'key': np.copy(key),
          'mark': np.copy(mark),
          'in_queue': np.copy(in_queue),
          'u': probing.mask_one(s, A.shape[0])
      })

  for _ in range(A.shape[0]):
    u = np.argsort(key + (1.0 - in_queue) * 1e9)[0]  # drop-in for extract-min
    if in_queue[u] == 0:
      break
    mark[u] = 1
    in_queue[u] = 0
    for v in range(A.shape[0]):
      if A[u, v] != 0:
        if mark[v] == 0 and (in_queue[v] == 0 or A[u, v] < key[v]):
          pi[v] = u
          key[v] = A[u, v]
          in_queue[v] = 1

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pi_h': np.copy(pi),
            'key': np.copy(key),
            'mark': np.copy(mark),
            'in_queue': np.copy(in_queue),
            'u': probing.mask_one(u, A.shape[0])
        })

  probing.push(probes, specs.Stage.OUTPUT, next_probe={'pi': np.copy(pi)})
  probing.finalize(probes)

  return pi, probes


def bellman_ford(A: _Array, s: int) -> _Out:
  """Bellman-Ford's single-source shortest path (Bellman, 1958)."""

  chex.assert_rank(A, 2)
  probes = probing.initialize(specs.SPECS['bellman_ford'])

  A_pos = np.arange(A.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          's': probing.mask_one(s, A.shape[0]),
          'A': np.copy(A),
          'adj': probing.graph(np.copy(A))
      })

  d = np.zeros(A.shape[0])
  pi = np.arange(A.shape[0])
  msk = np.zeros(A.shape[0])
  d[s] = 0
  msk[s] = 1
  while True:
    prev_d = np.copy(d)
    prev_msk = np.copy(msk)
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pi_h': np.copy(pi),
            'd': np.copy(prev_d),
            'msk': np.copy(prev_msk)
        })
    for u in range(A.shape[0]):
      for v in range(A.shape[0]):
        if prev_msk[u] == 1 and A[u, v] != 0:
          if msk[v] == 0 or prev_d[u] + A[u, v] < d[v]:
            d[v] = prev_d[u] + A[u, v]
            pi[v] = u
          msk[v] = 1
    if np.all(d == prev_d):
      break

  probing.push(probes, specs.Stage.OUTPUT, next_probe={'pi': np.copy(pi)})
  probing.finalize(probes)

  return pi, probes


def dijkstra(A: _Array, s: int) -> _Out:
  """Dijkstra's single-source shortest path (Dijkstra, 1959)."""

  chex.assert_rank(A, 2)
  probes = probing.initialize(specs.SPECS['dijkstra'])

  A_pos = np.arange(A.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          's': probing.mask_one(s, A.shape[0]),
          'A': np.copy(A),
          'adj': probing.graph(np.copy(A))
      })

  d = np.zeros(A.shape[0])
  mark = np.zeros(A.shape[0])
  in_queue = np.zeros(A.shape[0])
  pi = np.arange(A.shape[0])
  d[s] = 0
  in_queue[s] = 1

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pi_h': np.copy(pi),
          'd': np.copy(d),
          'mark': np.copy(mark),
          'in_queue': np.copy(in_queue),
          'u': probing.mask_one(s, A.shape[0])
      })

  for _ in range(A.shape[0]):
    u = np.argsort(d + (1.0 - in_queue) * 1e9)[0]  # drop-in for extract-min
    if in_queue[u] == 0:
      break
    mark[u] = 1
    in_queue[u] = 0
    for v in range(A.shape[0]):
      if A[u, v] != 0:
        if mark[v] == 0 and (in_queue[v] == 0 or d[u] + A[u, v] < d[v]):
          pi[v] = u
          d[v] = d[u] + A[u, v]
          in_queue[v] = 1

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pi_h': np.copy(pi),
            'd': np.copy(d),
            'mark': np.copy(mark),
            'in_queue': np.copy(in_queue),
            'u': probing.mask_one(u, A.shape[0])
        })

  probing.push(probes, specs.Stage.OUTPUT, next_probe={'pi': np.copy(pi)})
  probing.finalize(probes)

  return pi, probes


def dag_shortest_paths(A: _Array, s: int) -> _Out:
  """DAG shortest path."""

  chex.assert_rank(A, 2)
  probes = probing.initialize(specs.SPECS['dag_shortest_paths'])

  A_pos = np.arange(A.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          's': probing.mask_one(s, A.shape[0]),
          'A': np.copy(A),
          'adj': probing.graph(np.copy(A))
      })

  pi = np.arange(A.shape[0])
  d = np.zeros(A.shape[0])
  mark = np.zeros(A.shape[0])
  color = np.zeros(A.shape[0], dtype=np.int32)
  topo = np.arange(A.shape[0])
  s_prev = np.arange(A.shape[0])
  topo_head = 0
  s_last = s
  u = s
  v = s
  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pi_h': np.copy(pi),
          'd': np.copy(d),
          'mark': np.copy(mark),
          'topo_h': np.copy(topo),
          'topo_head_h': probing.mask_one(topo_head, A.shape[0]),
          'color': probing.array_cat(color, 3),
          's_prev': np.copy(s_prev),
          's': probing.mask_one(s, A.shape[0]),
          'u': probing.mask_one(u, A.shape[0]),
          'v': probing.mask_one(v, A.shape[0]),
          's_last': probing.mask_one(s_last, A.shape[0]),
          'phase': 0
      })
  while True:
    if color[u] == 0:
      color[u] = 1
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'pi_h': np.copy(pi),
              'd': np.copy(d),
              'mark': np.copy(mark),
              'topo_h': np.copy(topo),
              'topo_head_h': probing.mask_one(topo_head, A.shape[0]),
              'color': probing.array_cat(color, 3),
              's_prev': np.copy(s_prev),
              's': probing.mask_one(s, A.shape[0]),
              'u': probing.mask_one(u, A.shape[0]),
              'v': probing.mask_one(v, A.shape[0]),
              's_last': probing.mask_one(s_last, A.shape[0]),
              'phase': 0
          })

    for v in range(A.shape[0]):
      if A[u, v] != 0:
        if color[v] == 0:
          color[v] = 1
          s_prev[v] = s_last
          s_last = v

          probing.push(
              probes,
              specs.Stage.HINT,
              next_probe={
                  'pi_h': np.copy(pi),
                  'd': np.copy(d),
                  'mark': np.copy(mark),
                  'topo_h': np.copy(topo),
                  'topo_head_h': probing.mask_one(topo_head, A.shape[0]),
                  'color': probing.array_cat(color, 3),
                  's_prev': np.copy(s_prev),
                  's': probing.mask_one(s, A.shape[0]),
                  'u': probing.mask_one(u, A.shape[0]),
                  'v': probing.mask_one(v, A.shape[0]),
                  's_last': probing.mask_one(s_last, A.shape[0]),
                  'phase': 0
              })
          break

    if s_last == u:
      color[u] = 2

      if color[topo_head] == 2:
        topo[u] = topo_head
      topo_head = u

      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'pi_h': np.copy(pi),
              'd': np.copy(d),
              'mark': np.copy(mark),
              'topo_h': np.copy(topo),
              'topo_head_h': probing.mask_one(topo_head, A.shape[0]),
              'color': probing.array_cat(color, 3),
              's_prev': np.copy(s_prev),
              's': probing.mask_one(s, A.shape[0]),
              'u': probing.mask_one(u, A.shape[0]),
              'v': probing.mask_one(v, A.shape[0]),
              's_last': probing.mask_one(s_last, A.shape[0]),
              'phase': 0
          })

      if s_prev[u] == u:
        assert s_prev[s_last] == s_last
        break
      pr = s_prev[s_last]
      s_prev[s_last] = s_last
      s_last = pr

    u = s_last

  assert topo_head == s
  d[topo_head] = 0
  mark[topo_head] = 1

  while topo[topo_head] != topo_head:
    i = topo_head
    mark[topo_head] = 1

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pi_h': np.copy(pi),
            'd': np.copy(d),
            'mark': np.copy(mark),
            'topo_h': np.copy(topo),
            'topo_head_h': probing.mask_one(topo_head, A.shape[0]),
            'color': probing.array_cat(color, 3),
            's_prev': np.copy(s_prev),
            's': probing.mask_one(s, A.shape[0]),
            'u': probing.mask_one(u, A.shape[0]),
            'v': probing.mask_one(v, A.shape[0]),
            's_last': probing.mask_one(s_last, A.shape[0]),
            'phase': 1
        })

    for j in range(A.shape[0]):
      if A[i, j] != 0.0:
        if mark[j] == 0 or d[i] + A[i, j] < d[j]:
          d[j] = d[i] + A[i, j]
          pi[j] = i
          mark[j] = 1

    topo_head = topo[topo_head]

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pi_h': np.copy(pi),
          'd': np.copy(d),
          'mark': np.copy(mark),
          'topo_h': np.copy(topo),
          'topo_head_h': probing.mask_one(topo_head, A.shape[0]),
          'color': probing.array_cat(color, 3),
          's_prev': np.copy(s_prev),
          's': probing.mask_one(s, A.shape[0]),
          'u': probing.mask_one(u, A.shape[0]),
          'v': probing.mask_one(v, A.shape[0]),
          's_last': probing.mask_one(s_last, A.shape[0]),
          'phase': 1
      })

  probing.push(probes, specs.Stage.OUTPUT, next_probe={'pi': np.copy(pi)})
  probing.finalize(probes)

  return pi, probes


def floyd_warshall(A: _Array) -> _Out:
  """Floyd-Warshall's all-pairs shortest paths (Floyd, 1962)."""

  chex.assert_rank(A, 2)
  probes = probing.initialize(specs.SPECS['floyd_warshall'])

  A_pos = np.arange(A.shape[0])

  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) / A.shape[0],
          'A': np.copy(A),
          'adj': probing.graph(np.copy(A))
      })

  D = np.copy(A)
  Pi = np.zeros((A.shape[0], A.shape[0]))
  msk = probing.graph(np.copy(A))

  for i in range(A.shape[0]):
    for j in range(A.shape[0]):
      Pi[i, j] = i

  for k in range(A.shape[0]):
    prev_D = np.copy(D)
    prev_msk = np.copy(msk)

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'Pi_h': np.copy(Pi),
            'D': np.copy(prev_D),
            'msk': np.copy(prev_msk),
            'k': probing.mask_one(k, A.shape[0])
        })

    for i in range(A.shape[0]):
      for j in range(A.shape[0]):
        if prev_msk[i, k] > 0 and prev_msk[k, j] > 0:
          if msk[i, j] == 0 or prev_D[i, k] + prev_D[k, j] < D[i, j]:
            D[i, j] = prev_D[i, k] + prev_D[k, j]
            Pi[i, j] = Pi[k, j]
          else:
            D[i, j] = prev_D[i, j]
          msk[i, j] = 1

  probing.push(probes, specs.Stage.OUTPUT, next_probe={'Pi': np.copy(Pi)})
  probing.finalize(probes)

  return Pi, probes


def bipartite_matching(A: _Array, n: int, m: int, s: int, t: int) -> _Out:
  """Edmonds-Karp bipartite matching (Edmund & Karp, 1972)."""

  chex.assert_rank(A, 2)
  assert A.shape[0] == n + m + 2  # add source and sink vertices
  assert s == 0 and t == n + m + 1  # ensure for consistency

  probes = probing.initialize(specs.SPECS['bipartite_matching'])

  A_pos = np.arange(A.shape[0])

  adj = probing.graph(np.copy(A))
  probing.push(
      probes,
      specs.Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          'A': np.copy(A),
          'adj': adj,
          's': probing.mask_one(s, A.shape[0]),
          't': probing.mask_one(t, A.shape[0])
      })
  in_matching = (
      np.zeros((A.shape[0], A.shape[1])) + _OutputClass.MASKED.value + adj
      + adj.T)
  u = t
  while True:
    mask = np.zeros(A.shape[0])
    d = np.zeros(A.shape[0])
    pi = np.arange(A.shape[0])
    d[s] = 0
    mask[s] = 1
    while True:
      prev_d = np.copy(d)
      prev_mask = np.copy(mask)
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'in_matching_h': np.copy(in_matching),
              'A_h': np.copy(A),
              'adj_h': probing.graph(np.copy(A)),
              'd': np.copy(prev_d),
              'msk': np.copy(prev_mask),
              'pi': np.copy(pi),
              'u': probing.mask_one(u, A.shape[0]),
              'phase': 0
          })
      for u in range(A.shape[0]):
        for v in range(A.shape[0]):
          if A[u, v] != 0:
            if prev_mask[u] == 1 and (
                mask[v] == 0 or prev_d[u] + A[u, v] < d[v]):
              d[v] = prev_d[u] + A[u, v]
              pi[v] = u
              mask[v] = 1
      if np.all(d == prev_d):
        probing.push(
            probes,
            specs.Stage.OUTPUT,
            next_probe={'in_matching': np.copy(in_matching)},
        )
        probing.finalize(probes)
        return in_matching, probes
      elif pi[t] != t:
        break
    u = t
    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'in_matching_h': np.copy(in_matching),
            'A_h': np.copy(A),
            'adj_h': probing.graph(np.copy(A)),
            'd': np.copy(prev_d),
            'msk': np.copy(prev_mask),
            'pi': np.copy(pi),
            'u': probing.mask_one(u, A.shape[0]),
            'phase': 1
        })
    while pi[u] != u:
      if pi[u] < u:
        in_matching[pi[u], u] = 1
      else:
        in_matching[u, pi[u]] = 0
      A[pi[u], u] = 0
      A[u, pi[u]] = 1
      u = pi[u]
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'in_matching_h': np.copy(in_matching),
              'A_h': np.copy(A),
              'adj_h': probing.graph(np.copy(A)),
              'd': np.copy(prev_d),
              'msk': np.copy(prev_mask),
              'pi': np.copy(pi),
              'u': probing.mask_one(u, A.shape[0]),
              'phase': 1
          })
