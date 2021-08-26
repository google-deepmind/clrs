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

"""Sampling utilities."""

import abc
import collections
import types

from typing import Any, Callable, List, Optional, Tuple

from clrs._src import algorithms
from clrs._src import probing
from clrs._src import specs
import numpy as np


_Array = np.ndarray
_DataPoint = probing.DataPoint
Trajectory = List[_DataPoint]
Trajectories = List[Trajectory]


Algorithm = Callable[..., Any]
Features = collections.namedtuple('Features', ['inputs', 'hints', 'lengths'])
Feedback = collections.namedtuple('Feedback', ['features', 'outputs'])

# CLRS-21 baseline spec.
CLRS21 = types.MappingProxyType({
    'train': {
        'num_samples': 1000,
        'length': 16,
        'seed': 1,
    },
    'val': {
        'num_samples': 32,
        'length': 16,
        'seed': 2,
    },
    'test': {
        'num_samples': 32,
        'length': 64,
        'seed': 3,
    },
})


class Sampler(abc.ABC):
  """Sampler abstract base class."""

  def __init__(
      self,
      algorithm: Algorithm,
      spec: specs.Spec,
      num_samples: int,
      *args,
      seed: Optional[int] = None,
      **kwargs,
  ):
    """Initializes a `Sampler`.

    Args:
      algorithm: The algorithm to sample from
      spec: The algorithm spec.
      num_samples: Number of algorithm unrolls to sample.
      *args: Algorithm args.
      seed: RNG seed.
      **kwargs: Algorithm kwargs.
    """

    # Use `RandomState` to ensure deterministic sampling across Numpy versions.
    self._rng = np.random.RandomState(seed)
    self._num_samples = num_samples

    inputs = []
    outputs = []
    hints = []

    for _ in range(num_samples):
      data = self._sample_data(*args, **kwargs)
      _, probes = algorithm(*data)
      inp, outp, hint = probing.split_stages(probes, spec)
      inputs.append(inp)
      outputs.append(outp)
      hints.append(hint)

    # Batch and pad trajectories to max(T).
    self._inputs = _batch_io(inputs)
    self._outputs = _batch_io(outputs)
    self._hints, self._lengths = _batch_hints(hints)

  def next(self, batch_size: Optional[int] = None) -> Feedback:
    """Subsamples trajectories from the pre-generated dataset.

    Args:
      batch_size: Optional batch size. If `None`, returns entire dataset.

    Returns:
      Subsampled trajectories.
    """
    if batch_size:
      if batch_size > self._num_samples:
        raise ValueError(
            f'Batch size {batch_size} > dataset size {self._num_samples}.')

      # Returns a fixed-size random batch.
      indices = np.random.choice(self._num_samples, (batch_size,), replace=True)
      inputs = _subsample_data(self._inputs, indices, axis=0)
      outputs = _subsample_data(self._outputs, indices, axis=0)
      hints = _subsample_data(self._hints, indices, axis=1)
      lengths = self._lengths[indices]

    else:
      # Returns the full dataset.
      inputs = self._inputs
      hints = self._hints
      lengths = self._lengths
      outputs = self._outputs

    return Feedback(Features(inputs, hints, lengths), outputs)

  @abc.abstractmethod
  def _sample_data(self, length: int, *args, **kwargs) -> List[_Array]:
    pass

  def _random_sequence(self, length, low=0.0, high=1.0):
    """Random sequence."""
    return self._rng.uniform(low=low, high=high, size=(length,))

  def _random_string(self, length, chars=4):
    """Random string."""
    return self._rng.randint(0, high=chars, size=(length,))

  def _random_er_graph(self, nb_nodes, p=0.5, directed=False, acyclic=False,
                       weighted=False, low=0.0, high=1.0):
    """Random Erdos-Renyi graph."""

    mat = self._rng.binomial(1, p, size=(nb_nodes, nb_nodes))
    if not directed:
      mat *= np.transpose(mat)
    elif acyclic:
      mat = np.triu(mat, k=1)
      p = self._rng.permutation(nb_nodes)  # To allow nontrivial solutions
      mat = mat[p, :][:, p]
    if weighted:
      weights = self._rng.uniform(low=low, high=high, size=(nb_nodes, nb_nodes))
      mat = mat.astype(float) * weights
    return mat

  def _random_bipartite_graph(self, n, m, p=0.25):
    """Random bipartite graph-based flow network."""
    nb_nodes = n + m + 2
    s = 0
    t = n + m + 1
    mat = np.zeros((nb_nodes, nb_nodes))
    mat[s, 1:n+1] = 1.0  # supersource
    mat[n+1:n+m+1, t] = 1.0  # supersink
    mat[1:n+1, n+1:n+m+1] = self._rng.binomial(1, p, size=(n, m))
    return mat


def clrs21_train(name: str) -> Tuple[Sampler, specs.Spec]:
  """Builds a CLRS-21 training sampler for algorithm specified by `name`."""
  if name not in specs.CLRS_21_ALGS:
    raise NotImplementedError(f'Algorithm {name} not supported in CLRS-21.')
  sampler = build_sampler(
      name,
      seed=CLRS21['train']['seed'],
      num_samples=CLRS21['train']['num_samples'],
      length=CLRS21['train']['length'],
  )
  return sampler


def clrs21_val(name: str) -> Tuple[Sampler, specs.Spec]:
  """Builds a CLRS-21 validation sampler for algorithm specified by `name`."""
  if name not in specs.CLRS_21_ALGS:
    raise NotImplementedError(f'Algorithm {name} not supported in CLRS-21.')
  sampler = build_sampler(
      name,
      seed=CLRS21['val']['seed'],
      num_samples=CLRS21['val']['num_samples'],
      length=CLRS21['val']['length'],
  )
  return sampler


def clrs21_test(name: str) -> Tuple[Sampler, specs.Spec]:
  """Builds a CLRS-21 testing sampler for algorithm specified by `name`."""
  if name not in specs.CLRS_21_ALGS:
    raise NotImplementedError(f'Algorithm {name} not supported in CLRS-21.')
  sampler = build_sampler(
      name,
      seed=CLRS21['test']['seed'],
      num_samples=CLRS21['test']['num_samples'],
      length=CLRS21['test']['length'],
  )
  return sampler


def build_sampler(
    name: str,
    num_samples: int,
    *args,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[Sampler, specs.Spec]:
  """Builds a sampler. See `Sampler` documentation."""

  if name not in specs.SPECS or name not in SAMPLERS:
    raise NotImplementedError(f'No implementation of algorithm {name}.')
  spec = specs.SPECS[name]
  algorithm = getattr(algorithms, name)
  sampler = SAMPLERS[name](
      algorithm, spec, num_samples, seed=seed, *args, **kwargs)
  return sampler, spec


class SortingSampler(Sampler):
  """Sorting sampler. Generates a random sequence of U[0, 1]."""

  def _sample_data(
      self,
      length: int,
      low: float = 0.,
      high: float = 1.,
  ):
    arr = self._random_sequence(length=length, low=low, high=high)
    return [arr]


class SearchSampler(Sampler):
  """Search sampler. Generates a random sequence and target (of U[0, 1])."""

  def _sample_data(
      self,
      length: int,
      low: float = 0.,
      high: float = 1.,
  ):
    arr = self._random_sequence(length=length, low=low, high=high)
    x = self._rng.uniform(low=low, high=high)
    return [x, arr]


class MaxSubarraySampler(Sampler):
  """Maximum subarray sampler. Generates a random sequence of U[-1, 1]."""

  def _sample_data(
      self,
      length: int,
      low: float = -1.,
      high: float = 1.,
  ):
    arr = self._random_sequence(length=length, low=low, high=high)
    return [arr]


class LCSSampler(Sampler):
  """Longest Common Subsequence sampler. Generates two random ATCG strings."""

  def _sample_data(
      self,
      length: int,
      length_2: Optional[int] = None,
      chars: int = 4,
  ):
    if length_2 is None:
      # Assume provided length is total length.
      length_2 = length // 2
      length -= length_2
    a = self._random_string(length=length, chars=chars)
    b = self._random_string(length=length_2, chars=chars)
    return [a, b]


class OptimalBSTSampler(Sampler):
  """Optimal BST sampler. Samples array of probabilities, splits it into two."""

  def _sample_data(
      self,
      length: int,
  ):
    tot_length = length + (length + 1)
    arr = self._random_sequence(length=tot_length, low=0.0, high=1.0)
    arr /= np.sum(arr)
    p = arr[:length]
    q = arr[length:]
    return [p, q]


class ActivitySampler(Sampler):
  """Activity sampler. Samples start and finish times from U[0, 1]."""

  def _sample_data(
      self,
      length: int,
      low: float = 0.,
      high: float = 1.,
  ):
    arr_1 = self._random_sequence(length=length, low=low, high=high)
    arr_2 = self._random_sequence(length=length, low=low, high=high)
    return [np.minimum(arr_1, arr_2), np.maximum(arr_1, arr_2)]


class TaskSampler(Sampler):
  """Task sampler. Samples deadlines (integers) and values (U[0, 1])."""

  def _sample_data(
      self,
      length: int,
      max_deadline: Optional[int] = None,
      low: float = 0.,
      high: float = 1.,
  ):
    if max_deadline is None:
      max_deadline = length
    d = self._random_string(length=length, chars=max_deadline) + 1
    w = self._random_sequence(length=length, low=low, high=high)
    return [d, w]


class DfsSampler(Sampler):
  """DFS sampler."""

  def _sample_data(
      self,
      length: int,
      p: float = 0.5,
  ):
    graph = self._random_er_graph(
        nb_nodes=length, p=p, directed=True, acyclic=False, weighted=False)
    return [graph]


class BfsSampler(Sampler):
  """BFS sampler."""

  def _sample_data(
      self,
      length: int,
      p: float = 0.5,
  ):
    graph = self._random_er_graph(
        nb_nodes=length, p=p, directed=False, acyclic=False, weighted=False)
    source_node = self._rng.choice(length)
    return [graph, source_node]


class TopoSampler(Sampler):
  """Topological Sorting sampler."""

  def _sample_data(
      self,
      length: int,
      p: float = 0.5,
  ):
    graph = self._random_er_graph(
        nb_nodes=length, p=p, directed=True, acyclic=True, weighted=False)
    return [graph]


class ArticulationSampler(Sampler):
  """Articulation Point sampler."""

  def _sample_data(
      self,
      length: int,
      p: float = 0.3,
  ):
    graph = self._random_er_graph(
        nb_nodes=length, p=p, directed=False, acyclic=False, weighted=False)
    return [graph]


class MSTSampler(Sampler):
  """MST sampler."""

  def _sample_data(
      self,
      length: int,
      p: float = 0.5,
      low: float = 0.,
      high: float = 1.,
  ):
    graph = self._random_er_graph(
        nb_nodes=length,
        p=p,
        directed=False,
        acyclic=False,
        weighted=True,
        low=low,
        high=high)
    return [graph]


class BellmanFordSampler(Sampler):
  """Bellman-Ford sampler."""

  def _sample_data(
      self,
      length: int,
      p: float = 0.5,
      low: float = 0.,
      high: float = 1.,
  ):
    graph = self._random_er_graph(
        nb_nodes=length,
        p=p,
        directed=False,
        acyclic=False,
        weighted=True,
        low=low,
        high=high)
    source_node = self._rng.choice(length)
    return [graph, source_node]


class DAGPathSampler(Sampler):
  """Sampler for DAG shortest paths."""

  def _sample_data(
      self,
      length: int,
      p: float = 0.5,
      low: float = 0.,
      high: float = 1.,
  ):
    graph = self._random_er_graph(
        nb_nodes=length,
        p=p,
        directed=True,
        acyclic=True,
        weighted=True,
        low=low,
        high=high)
    source_node = self._rng.choice(length)
    return [graph, source_node]


class BipartiteSampler(Sampler):
  """Sampler for bipartite matching-based flow networks."""

  def _sample_data(
      self,
      length: int,
      length_2: Optional[int] = None,
      p: float = 0.3,
  ):
    if length_2 is None:
      # Assume provided length is total length.
      length_2 = length // 2
      length -= length_2
    graph = self._random_bipartite_graph(n=length, m=length_2, p=p)
    return [graph, length, length_2, 0, length + length_2 + 1]


class MatcherSampler(Sampler):
  """String matching sampler; embeds needle in a random haystack."""

  def _sample_data(
      self,
      length: int,
      length_needle: Optional[int] = None,
      chars: int = 4,
  ):
    if length_needle is None:
      if length < 4:
        length_needle = 1
      else:
        length_needle = length // 4
    needle = self._random_string(length=length_needle, chars=chars)
    haystack = self._random_string(length=length, chars=chars)
    embed_pos = self._rng.choice(length - length_needle)
    haystack[embed_pos:embed_pos + length_needle] = needle
    return [haystack, needle]


class SegmentsSampler(Sampler):
  """Two-segment sampler of points from (U[0, 1], U[0, 1])."""

  def _sample_data(self, length: int, low: float = 0., high: float = 1.):
    del length  # There are exactly four endpoints.
    xs = self._random_sequence(length=4, low=low, high=high)
    ys = self._random_sequence(length=4, low=low, high=high)
    return [xs, ys]


class ConvexHullSampler(Sampler):
  """Convex hull sampler of points from (U[0, 1], U[0, 1])."""

  def _sample_data(self, length: int, low: float = 0., high: float = 1.):
    xs = self._random_sequence(length=length, low=low, high=high)
    ys = self._random_sequence(length=length, low=low, high=high)
    return [xs, ys]


SAMPLERS = {
    'insertion_sort': SortingSampler,
    'bubble_sort': SortingSampler,
    'heapsort': SortingSampler,
    'quicksort': SortingSampler,
    'quickselect': SortingSampler,
    'minimum': SortingSampler,
    'binary_search': SearchSampler,
    'find_maximum_subarray': MaxSubarraySampler,
    'find_maximum_subarray_kadane': MaxSubarraySampler,
    'matrix_chain_order': SortingSampler,
    'lcs_length': LCSSampler,
    'optimal_bst': OptimalBSTSampler,
    'activity_selector': ActivitySampler,
    'task_scheduling': TaskSampler,
    'dfs': DfsSampler,
    'topological_sort': TopoSampler,
    'strongly_connected_components': DfsSampler,
    'articulation_points': ArticulationSampler,
    'bridges': ArticulationSampler,
    'bfs': BfsSampler,
    'mst_kruskal': MSTSampler,
    'mst_prim': BellmanFordSampler,
    'bellman_ford': BellmanFordSampler,
    'dag_shortest_paths': DAGPathSampler,
    'dijkstra': BellmanFordSampler,
    'floyd_warshall': MSTSampler,
    'bipartite_matching': BipartiteSampler,
    'naive_string_matcher': MatcherSampler,
    'kmp_matcher': MatcherSampler,
    'segments_intersect': SegmentsSampler,
    'graham_scan': ConvexHullSampler,
    'jarvis_march': ConvexHullSampler,
}


def _batch_io(traj_io: Trajectories) -> Trajectory:
  """Batches a trajectory of input/output samples along the time axis per probe.

  Args:
    traj_io:  An i/o trajectory of `DataPoint`s indexed by time then probe.

  Returns:
    A |num probes| list of `DataPoint`s with the time axis stacked into `data`.
  """

  assert traj_io  # non-empty
  for sample_io in traj_io:
    for dp in sample_io:
      assert dp.data.shape[0] == 1  # batching axis

  batched_traj = traj_io[0]  # construct batched trajectory in-place
  for cur_sample in traj_io[1:]:
    for i in range(len(batched_traj)):
      # Validate that each trajectory contains the same probes.
      assert batched_traj[i].name == cur_sample[i].name

      # Concatenate each probe along the trajectory/time axis.
      batched_traj[i] = probing.DataPoint(
          name=batched_traj[i].name,
          location=batched_traj[i].location,
          type_=batched_traj[i].type_,
          data=np.concatenate([batched_traj[i].data, cur_sample[i].data],
                              axis=0))

  return batched_traj


def _batch_hints(traj_hints: Trajectories) -> Tuple[Trajectory, List[int]]:
  """Batches a trajectory of hints samples along the time axis per probe.

  Unlike i/o, hints have a variable-length time dimension. Before batching, each
  trajectory is padded to the maximum trajectory length.

  Args:
    traj_hints: A hint trajectory of `DataPoints`s indexed by time then probe

  Returns:
    A |num probes| list of `DataPoint`s with the time axis stacked into `data`,
    and a |sample| list containing the length of each trajectory.
  """

  max_steps = 0
  assert traj_hints  # non-empty
  for sample_hint in traj_hints:
    for dp in sample_hint:
      assert dp.data.shape[1] == 1  # batching axis
      if dp.data.shape[0] > max_steps:
        max_steps = dp.data.shape[0]

  batched_traj = traj_hints[0]  # construct batched trajectory in-place
  hint_lengths = np.zeros(len(traj_hints))
  for i in range(len(traj_hints[0])):
    hint_i = traj_hints[0][i]
    assert batched_traj[i].name == hint_i.name
    batched_traj[i] = probing.DataPoint(
        name=batched_traj[i].name,
        location=batched_traj[i].location,
        type_=batched_traj[i].type_,
        data=np.zeros((max_steps,) + hint_i.data.shape[1:]))
    batched_traj[i].data[:hint_i.data.shape[0]] = hint_i.data
    if i > 0:
      assert hint_lengths[0] == hint_i.data.shape[0]
    else:
      hint_lengths[0] = hint_i.data.shape[0]

  for hint_ind, cur_hint in enumerate(traj_hints[1:], start=1):
    for i in range(len(cur_hint)):
      assert batched_traj[i].name == cur_hint[i].name

      # Extend the previously built stacked hint with new all-zero data point.
      batched_traj[i] = probing.DataPoint(
          name=batched_traj[i].name,
          location=batched_traj[i].location,
          type_=batched_traj[i].type_,
          data=np.concatenate(
              [batched_traj[i].data,
               np.zeros((max_steps,) + cur_hint[i].data.shape[1:])], axis=1))

      # Once extended, populate it only up to the current hint's length.
      # The -1: indexes the last timestep, but keeps axis (present in cur_hint).
      batched_traj[i].data[:cur_hint[i].data.shape[0], -1:] = cur_hint[i].data

      if i > 0:
        assert hint_lengths[hint_ind] == cur_hint[i].data.shape[0]
      else:
        hint_lengths[hint_ind] = cur_hint[i].data.shape[0]

  return batched_traj, hint_lengths


def _subsample_data(
    trajectory: Trajectory,
    idx: List[int],
    axis: int = 0,
) -> Trajectory:
  """New `Trajectory` where each `DataPoint`'s data is subsampled along axis."""
  sampled_traj = []
  for dp in trajectory:
    sampled_data = np.take(dp.data, idx, axis=axis)
    sampled_traj.append(
        probing.DataPoint(dp.name, dp.location, dp.type_, sampled_data))
  return sampled_traj
