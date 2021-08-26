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

"""Unit tests for `graphs.py`."""
# pylint: disable=invalid-name

from absl.testing import absltest

from clrs._src.algorithms import graphs
import numpy as np


# Unweighted graphs.

DAG = np.array([
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
])

DIRECTED = np.array([
    [0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1],
])

UNDIRECTED = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0],
])

ANOTHER_UNDIRECTED = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
])


# Weighted graphs.

X = np.iinfo(np.int32).max  # not connected

WEIGHTED_DAG = np.array([
    [X, 9, 3, X, X],
    [X, X, 6, X, 2],
    [X, X, X, 1, X],
    [X, X, X, X, 2],
    [X, X, X, X, X],
])

WEIGHTED_DIRECTED = np.array([
    [X, 9, 3, X, X],
    [X, X, 6, X, 2],
    [X, 2, X, 1, X],
    [X, X, 2, X, 2],
    [X, X, X, X, X],
])

WEIGHTED_UNDIRECTED = np.array([
    [X, 2, 3, X, X],
    [2, X, 1, 3, 2],
    [3, 1, X, X, 1],
    [X, 3, X, X, 5],
    [X, 2, 1, 5, X],
])


# Bipartite graphs.

BIPARTITE = np.array([
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

BIPARTITE_2 = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
])


class GraphsTest(absltest.TestCase):

  def test_dfs(self):
    expected_directed = np.array([0, 0, 2, 4, 1, 2])
    out, _ = graphs.dfs(DIRECTED)
    np.testing.assert_array_equal(expected_directed, out)

    expected_undirected = np.array([0, 0, 1, 2, 3])
    out, _ = graphs.dfs(UNDIRECTED)
    np.testing.assert_array_equal(expected_undirected, out)

  def test_bfs(self):
    expected_directed = np.array([0, 0, 2, 0, 1, 5])
    out, _ = graphs.bfs(DIRECTED, 0)
    np.testing.assert_array_equal(expected_directed, out)

    expected_undirected = np.array([0, 0, 1, 1, 0])
    out, _ = graphs.bfs(UNDIRECTED, 0)
    np.testing.assert_array_equal(expected_undirected, out)

  def test_topological_sort(self):
    expected_dag = np.array([3, 4, 0, 1, 4])
    out, _ = graphs.topological_sort(DAG)
    np.testing.assert_array_equal(expected_dag, out)

  def test_articulation_points(self):
    expected = np.array([1, 0, 0, 1, 0])
    out, _ = graphs.articulation_points(ANOTHER_UNDIRECTED)
    np.testing.assert_array_equal(expected, out)

  def test_bridges(self):
    expected = np.array([
        [0, 0, 0, 1, -1],
        [0, 0, 0, -1, -1],
        [0, 0, 0, -1, -1],
        [1, -1, -1, 0, 1],
        [-1, -1, -1, 1, 0],
    ])
    out, _ = graphs.bridges(ANOTHER_UNDIRECTED)
    np.testing.assert_array_equal(expected, out)

  def test_strongly_connected_components(self):
    expected_directed = np.array([0, 1, 2, 1, 1, 5])
    out, _ = graphs.strongly_connected_components(DIRECTED)
    np.testing.assert_array_equal(expected_directed, out)

    expected_undirected = np.array([0, 0, 0, 0, 0])
    out, _ = graphs.strongly_connected_components(UNDIRECTED)
    np.testing.assert_array_equal(expected_undirected, out)

  def test_mst_kruskal(self):
    expected = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ])
    out, _ = graphs.mst_kruskal(WEIGHTED_UNDIRECTED)
    np.testing.assert_array_equal(expected, out)

  def test_mst_prim(self):
    expected = np.array([0, 0, 1, 1, 2])
    out, _ = graphs.mst_prim(WEIGHTED_UNDIRECTED, 0)
    np.testing.assert_array_equal(expected, out)

  def test_bellman_ford(self):
    expected = np.array([0, 2, 0, 2, 3])
    out, _ = graphs.bellman_ford(WEIGHTED_DIRECTED, 0)
    np.testing.assert_array_equal(expected, out)

  def test_dag_shortest_paths(self):
    expected = np.array([0, 0, 0, 2, 3])
    out, _ = graphs.bellman_ford(WEIGHTED_DAG, 0)
    np.testing.assert_array_equal(expected, out)

  def test_dijkstra(self):
    expected = np.array([0, 2, 0, 2, 3])
    out, _ = graphs.dijkstra(WEIGHTED_DIRECTED, 0)
    np.testing.assert_array_equal(expected, out)

  def test_floyd_warshall(self):
    expected = np.array([0, 2, 0, 2, 3])
    out, _ = graphs.floyd_warshall(WEIGHTED_DIRECTED)
    np.testing.assert_array_equal(expected, out[0])

  def test_bipartite_matching(self):
    expected = np.array([
        [1, 1, 1, 1, 0, 0, -1, -1, -1, -1, -1],
        [0, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1],
        [0, -1, 1, -1, -1, -1, 0, -1, 1, -1, -1],
        [0, -1, -1, 1, -1, -1, -1, 1, 0, 0, -1],
        [0, -1, -1, -1, 1, -1, -1, -1, 0, -1, -1],
        [0, -1, -1, -1, -1, 1, -1, -1, 0, -1, -1],
        [-1, 0, 0, -1, -1, -1, 1, -1, -1, -1, 1],
        [-1, -1, -1, 0, -1, -1, -1, 1, -1, -1, 1],
        [-1, -1, 0, 0, 0, 0, -1, -1, 1, -1, 1],
        [-1, -1, -1, 0, -1, -1, -1, -1, -1, 1, 0],
        [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1],
    ])
    out, _ = graphs.bipartite_matching(BIPARTITE, 5, 4, 0, 10)
    np.testing.assert_array_equal(expected, out)

    expected_2 = np.array([
        [1, 1, 1, 1, -1, -1, -1, -1],
        [0, 1, -1, -1, 0, 1, -1, -1],
        [0, -1, 1, -1, 1, -1, 0, -1],
        [0, -1, -1, 1, -1, -1, 1, -1],
        [-1, 0, 0, -1, 1, -1, -1, 1],
        [-1, 0, -1, -1, -1, 1, -1, 1],
        [-1, -1, 0, 0, -1, -1, 1, 1],
        [-1, -1, -1, -1, 0, 0, 0, 1],
    ])
    out_2, _ = graphs.bipartite_matching(BIPARTITE_2, 3, 3, 0, 7)
    np.testing.assert_array_equal(expected_2, out_2)

if __name__ == "__main__":
  absltest.main()
