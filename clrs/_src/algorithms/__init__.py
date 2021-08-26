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

"""CLRS algorithm implementations."""

# pylint:disable=g-bad-import-order

from clrs._src.algorithms.divide_and_conquer import find_maximum_subarray
from clrs._src.algorithms.divide_and_conquer import find_maximum_subarray_kadane

from clrs._src.algorithms.dynamic_programming import matrix_chain_order
from clrs._src.algorithms.dynamic_programming import lcs_length
from clrs._src.algorithms.dynamic_programming import optimal_bst

from clrs._src.algorithms.geometry import segments_intersect
from clrs._src.algorithms.geometry import graham_scan
from clrs._src.algorithms.geometry import jarvis_march

from clrs._src.algorithms.graphs import dfs
from clrs._src.algorithms.graphs import bfs
from clrs._src.algorithms.graphs import topological_sort
from clrs._src.algorithms.graphs import articulation_points
from clrs._src.algorithms.graphs import bridges
from clrs._src.algorithms.graphs import strongly_connected_components
from clrs._src.algorithms.graphs import mst_kruskal
from clrs._src.algorithms.graphs import mst_prim
from clrs._src.algorithms.graphs import bellman_ford
from clrs._src.algorithms.graphs import dijkstra
from clrs._src.algorithms.graphs import dag_shortest_paths
from clrs._src.algorithms.graphs import floyd_warshall
from clrs._src.algorithms.graphs import bipartite_matching

from clrs._src.algorithms.greedy import activity_selector
from clrs._src.algorithms.greedy import task_scheduling

from clrs._src.algorithms.searching import minimum
from clrs._src.algorithms.searching import binary_search
from clrs._src.algorithms.searching import quickselect

from clrs._src.algorithms.sorting import insertion_sort
from clrs._src.algorithms.sorting import bubble_sort
from clrs._src.algorithms.sorting import heapsort
from clrs._src.algorithms.sorting import quicksort

from clrs._src.algorithms.strings import naive_string_matcher
from clrs._src.algorithms.strings import kmp_matcher
