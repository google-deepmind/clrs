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
import copy
import inspect
import itertools
import types
from typing import Any, Callable, List, Optional, Tuple
from absl import logging
from clrs._src import algorithms
from clrs._src import probing
from clrs._src import specs
import jax
import numpy as np


_Array = np.ndarray
_DataPoint = probing.DataPoint
Trajectory = List[_DataPoint]
Trajectories = List[Trajectory]


Algorithm = Callable[..., Any]
Features = collections.namedtuple('Features', ['inputs', 'hints', 'lengths'])
FeaturesChunked = collections.namedtuple(
    'Features', ['inputs', 'hints', 'is_first', 'is_last'])
Feedback = collections.namedtuple('Feedback', ['features', 'outputs'])

# CLRS-30 baseline spec.
CLRS30 = types.MappingProxyType({
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
  CAN_TRUNCATE_INPUT_DATA = None

  def __init__(
      self,
      algorithm: Algorithm,
      spec: specs.Spec,
      num_samples: int,
      *args,
      seed: Optional[int] = None,
      track_max_steps: bool = True,
      truncate_decimals: int | None = None,
      **kwargs,
  ):
    """Initializes a `Sampler`.

    Args:
      algorithm: The algorithm to sample from
      spec: The algorithm spec.
      num_samples: Number of algorithm unrolls to sample. If positive, all the
        samples will be generated in the constructor, and at each call of the
        `next` method a batch will be randomly selected among them. If -1,
        samples are generated on the fly with each call to `next`.
      *args: Algorithm args.
      seed: RNG seed.
      track_max_steps: if True and sampling on the fly (`num_samples`==-1), we
        keep track of the maximum unroll length so far to pad batched samples to
        that length. This is desirable when batches are used in compiled
        functions that need recompilation every time the batch size changes.
        Also, we get an initial value for max_steps by generating 1000 samples,
        which will slow down initialization. If uniform shape of the batches is
        not a concern, set `track_max_steps` to False.
      truncate_decimals: If not None, the sampler will truncate the input data
        of the algorithm.
      **kwargs: Algorithm kwargs.
    """

    # Use `RandomState` to ensure deterministic sampling across Numpy versions.
    self._rng = np.random.RandomState(seed)
    self._spec = spec
    self._num_samples = num_samples
    self._algorithm = algorithm
    self._args = args
    self._kwargs = kwargs
    self._track_max_steps = track_max_steps
    self._truncate_decimals = truncate_decimals

    if num_samples < 0:
      logging.log_first_n(
          logging.WARNING, 'Sampling dataset on-the-fly, unlimited samples.', 1
      )
      if track_max_steps:
        # Get an initial estimate of max hint length
        self.max_steps = -1
        for _ in range(1000):
          data = self._sample_data(*args, **kwargs)
          data = self._trunc_array(data)
          _, probes = algorithm(*data)
          _, _, hint = probing.split_stages(probes, spec)
          for dp in hint:
            assert dp.data.shape[1] == 1  # batching axis
            if dp.data.shape[0] > self.max_steps:
              self.max_steps = dp.data.shape[0]
    else:
      logging.info('Creating a dataset with %i samples.', num_samples)
      (self._inputs, self._outputs, self._hints,
       self._lengths) = self._make_batch(num_samples, spec, 0, algorithm, *args,
                                         **kwargs)

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    # Check that the subclass has overridden CAN_TRUNCATE_INPUT_DATA
    if getattr(cls, 'CAN_TRUNCATE_INPUT_DATA', None) is None:
      raise NotImplementedError(
          f'{cls.__name__} must define class attribute'
          " 'CAN_TRUNCATE_INPUT_DATA'."
      )

  def _make_batch(self, num_samples: int, spec: specs.Spec, min_length: int,
                  algorithm: Algorithm, *args, **kwargs):
    """Generate a batch of data."""
    inputs = []
    outputs = []
    hints = []

    for _ in range(num_samples):
      data = self._sample_data(*args, **kwargs)
      data = self._trunc_array(data)
      _, probes = algorithm(*data)
      inp, outp, hint = probing.split_stages(probes, spec)
      inputs.append(inp)
      outputs.append(outp)
      hints.append(hint)
      if len(hints) % 1000 == 0:
        logging.info('%i samples created', len(hints))

    # Batch and pad trajectories to max(T).
    inputs = _batch_io(inputs)
    outputs = _batch_io(outputs)
    hints, lengths = _batch_hints(hints, min_length)
    return inputs, outputs, hints, lengths

  def next(self, batch_size: Optional[int] = None) -> Feedback:
    """Subsamples trajectories from the pre-generated dataset.

    Args:
      batch_size: Optional batch size. If `None`, returns entire dataset.

    Returns:
      Subsampled trajectories.
    """
    if batch_size:
      if self._num_samples < 0:  # generate on the fly
        min_length = self.max_steps if self._track_max_steps else 0
        inputs, outputs, hints, lengths = self._make_batch(
            batch_size,
            self._spec,
            min_length,
            self._algorithm,
            *self._args,
            **self._kwargs,
        )
        if self._track_max_steps and hints[0].data.shape[0] > self.max_steps:
          logging.warning('Increasing hint lengh from %i to %i',
                          self.max_steps, hints[0].data.shape[0])
          self.max_steps = hints[0].data.shape[0]
      else:
        if batch_size > self._num_samples:
          raise ValueError(
              f'Batch size {batch_size} > dataset size {self._num_samples}.')

        # Returns a fixed-size random batch.
        indices = self._rng.choice(self._num_samples, (batch_size,),
                                   replace=True)
        inputs = _subsample_data(self._inputs, indices, axis=0)
        outputs = _subsample_data(self._outputs, indices, axis=0)
        hints = _subsample_data(self._hints, indices, axis=1)
        lengths = self._lengths[indices]

    else:
      # Returns the full dataset.
      assert self._num_samples >= 0
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
      if not directed:
        weights *= np.transpose(weights)
        weights = np.sqrt(weights + 1e-3)  # Add epsilon to protect underflow
      mat = mat.astype(float) * weights
    return mat

  def _random_community_graph(self, nb_nodes, k=4, p=0.5, eps=0.01,
                              directed=False, acyclic=False, weighted=False,
                              low=0.0, high=1.0):
    """Random perturbed k-community graph."""
    mat = np.zeros((nb_nodes, nb_nodes))
    if k > nb_nodes:
      raise ValueError(f'Cannot generate graph of too many ({k}) communities.')
    los, his = [], []
    lo = 0
    for i in range(k):
      if i == k - 1:
        hi = nb_nodes
      else:
        hi = lo + nb_nodes // k
      mat[lo:hi, lo:hi] = self._random_er_graph(
          hi - lo, p=p, directed=directed,
          acyclic=acyclic, weighted=weighted,
          low=low, high=high)
      los.append(lo)
      his.append(hi)
      lo = hi
    toggle = self._random_er_graph(nb_nodes, p=eps, directed=directed,
                                   acyclic=acyclic, weighted=weighted,
                                   low=low, high=high)

    # Prohibit closing new cycles
    for i in range(k):
      for j in range(i):
        toggle[los[i]:his[i], los[j]:his[j]] *= 0

    mat = np.where(toggle > 0.0, (1.0 - (mat > 0.0)) * toggle, mat)
    p = self._rng.permutation(nb_nodes)  # To allow nontrivial solutions
    mat = mat[p, :][:, p]
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

  def _trunc_array(self, data: Any) -> List[_Array]:
    """Truncates the data if needed."""
    data = copy.deepcopy(data)

    if not self._truncate_decimals:
      return data

    for index in range(len(data)):
      input_data = data[index]
      if not (_is_float_array(input_data) or isinstance(input_data, float)):
        continue

      data[index] = np.trunc(input_data * 10**self._truncate_decimals) / (
          10**self._truncate_decimals
      )

      if isinstance(input_data, float):
        data[index] = float(data[index])

    return data


def _is_float_array(data: Any) -> bool:
  """Checks if the given data is a float numpy array."""
  if isinstance(data, np.ndarray):
    return issubclass(data.dtype.type, np.floating)
  return False


def build_sampler(
    name: str,
    num_samples: int,
    *args,
    seed: Optional[int] = None,
    track_max_steps: bool = True,
    truncate_decimals: int | None = None,
    **kwargs,
) -> Tuple[Sampler, specs.Spec]:
  """Builds a sampler. See `Sampler` documentation."""

  if name not in specs.SPECS or name not in SAMPLERS:
    raise NotImplementedError(f'No implementation of algorithm {name}.')
  spec = specs.SPECS[name]
  algorithm = getattr(algorithms, name)
  sampler_class = SAMPLERS[name]
  # Ignore kwargs not accepted by the sampler.
  sampler_args = inspect.signature(sampler_class._sample_data).parameters  # pylint:disable=protected-access
  clean_kwargs = {k: kwargs[k] for k in kwargs if k in sampler_args}
  if set(clean_kwargs) != set(kwargs):
    logging.warning('Ignoring kwargs %s when building sampler class %s',
                    set(kwargs).difference(clean_kwargs), sampler_class)
  sampler = sampler_class(
      algorithm,
      spec,
      num_samples,
      seed=seed,
      track_max_steps=track_max_steps,
      truncate_decimals=truncate_decimals,
      *args,
      **clean_kwargs,
  )
  return sampler, spec


class SortingSampler(Sampler):
  """Sorting sampler. Generates a random sequence of U[0, 1]."""
  CAN_TRUNCATE_INPUT_DATA = True

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
  CAN_TRUNCATE_INPUT_DATA = True

  def _sample_data(
      self,
      length: int,
      low: float = 0.,
      high: float = 1.,
  ):
    arr = self._random_sequence(length=length, low=low, high=high)
    arr.sort()
    x = self._rng.uniform(low=low, high=high)
    return [x, arr]


class MaxSubarraySampler(Sampler):
  """Maximum subarray sampler. Generates a random sequence of U[-1, 1]."""
  CAN_TRUNCATE_INPUT_DATA = True

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
  CAN_TRUNCATE_INPUT_DATA = False

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
  CAN_TRUNCATE_INPUT_DATA = True

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
  CAN_TRUNCATE_INPUT_DATA = True

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
  CAN_TRUNCATE_INPUT_DATA = True

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
  CAN_TRUNCATE_INPUT_DATA = False

  def _sample_data(
      self,
      length: int,
      p: Tuple[float, ...] = (0.5,),
  ):
    graph = self._random_er_graph(
        nb_nodes=length, p=self._rng.choice(p),
        directed=True, acyclic=False, weighted=False)
    return [graph]


class BfsSampler(Sampler):
  """BFS sampler."""
  CAN_TRUNCATE_INPUT_DATA = False

  def _sample_data(
      self,
      length: int,
      p: Tuple[float, ...] = (0.5,),
  ):
    graph = self._random_er_graph(
        nb_nodes=length, p=self._rng.choice(p),
        directed=False, acyclic=False, weighted=False)
    source_node = self._rng.choice(length)
    return [graph, source_node]


class TopoSampler(Sampler):
  """Topological Sorting sampler."""
  CAN_TRUNCATE_INPUT_DATA = False

  def _sample_data(
      self,
      length: int,
      p: Tuple[float, ...] = (0.5,),
  ):
    graph = self._random_er_graph(
        nb_nodes=length, p=self._rng.choice(p),
        directed=True, acyclic=True, weighted=False)
    return [graph]


class ArticulationSampler(Sampler):
  """Articulation Point sampler."""
  CAN_TRUNCATE_INPUT_DATA = False

  def _sample_data(
      self,
      length: int,
      p: Tuple[float, ...] = (0.2,),
  ):
    graph = self._random_er_graph(
        nb_nodes=length, p=self._rng.choice(p), directed=False,
        acyclic=False, weighted=False)
    return [graph]


class MSTSampler(Sampler):
  """MST sampler for Kruskal's algorithm."""
  CAN_TRUNCATE_INPUT_DATA = True

  def _sample_data(
      self,
      length: int,
      p: Tuple[float, ...] = (0.2,),  # lower p to account for class imbalance
      low: float = 0.,
      high: float = 1.,
  ):
    graph = self._random_er_graph(
        nb_nodes=length,
        p=self._rng.choice(p),
        directed=False,
        acyclic=False,
        weighted=True,
        low=low,
        high=high)
    return [graph]


class BellmanFordSampler(Sampler):
  """Bellman-Ford sampler."""
  CAN_TRUNCATE_INPUT_DATA = True

  def _sample_data(
      self,
      length: int,
      p: Tuple[float, ...] = (0.5,),
      low: float = 0.,
      high: float = 1.,
  ):
    graph = self._random_er_graph(
        nb_nodes=length,
        p=self._rng.choice(p),
        directed=False,
        acyclic=False,
        weighted=True,
        low=low,
        high=high)
    source_node = self._rng.choice(length)
    return [graph, source_node]


class DAGPathSampler(Sampler):
  """Sampler for DAG shortest paths."""
  CAN_TRUNCATE_INPUT_DATA = True

  def _sample_data(
      self,
      length: int,
      p: Tuple[float, ...] = (0.5,),
      low: float = 0.,
      high: float = 1.,
  ):
    graph = self._random_er_graph(
        nb_nodes=length,
        p=self._rng.choice(p),
        directed=True,
        acyclic=True,
        weighted=True,
        low=low,
        high=high)
    source_node = self._rng.choice(length)
    return [graph, source_node]


class FloydWarshallSampler(Sampler):
  """Sampler for all-pairs shortest paths."""
  CAN_TRUNCATE_INPUT_DATA = True

  def _sample_data(
      self,
      length: int,
      p: Tuple[float, ...] = (0.5,),
      low: float = 0.,
      high: float = 1.,
  ):
    graph = self._random_er_graph(
        nb_nodes=length,
        p=self._rng.choice(p),
        directed=False,
        acyclic=False,
        weighted=True,
        low=low,
        high=high)
    return [graph]


class SccSampler(Sampler):
  """Sampler for strongly connected component (SCC) tasks."""
  CAN_TRUNCATE_INPUT_DATA = False

  def _sample_data(
      self,
      length: int,
      k: int = 4,
      p: Tuple[float, ...] = (0.5,),
      eps: float = 0.01,
  ):
    graph = self._random_community_graph(
        nb_nodes=length, k=k, p=self._rng.choice(p), eps=eps,
        directed=True, acyclic=False, weighted=False)
    return [graph]


class BipartiteSampler(Sampler):
  """Sampler for bipartite matching-based flow networks."""
  CAN_TRUNCATE_INPUT_DATA = False

  def _sample_data(
      self,
      length: int,
      length_2: Optional[int] = None,
      p: Tuple[float, ...] = (0.3,),
  ):
    if length_2 is None:
      # Assume provided length is total length.
      length_2 = length // 2
      length -= length_2
    graph = self._random_bipartite_graph(n=length, m=length_2,
                                         p=self._rng.choice(p))
    return [graph, length, length_2, 0, length + length_2 + 1]


class MatcherSampler(Sampler):
  """String matching sampler; embeds needle in a random haystack."""
  CAN_TRUNCATE_INPUT_DATA = False

  def _sample_data(
      self,
      length: int,  # length of haystack + needle, i.e., total number of nodes
      length_needle: Optional[int] = None,
      chars: int = 4,
  ):
    if length_needle is None:
      if length < 5:
        length_needle = 1
      else:
        length_needle = length // 5
    elif length_needle < 0:  # randomize needle length
      length_needle = self._rng.randint(1, high=1 - length_needle)
    length_haystack = length - length_needle
    needle = self._random_string(length=length_needle, chars=chars)
    haystack = self._random_string(length=length_haystack, chars=chars)
    embed_pos = self._rng.choice(length_haystack - length_needle)
    haystack[embed_pos:embed_pos + length_needle] = needle
    return [haystack, needle]


class SegmentsSampler(Sampler):
  """Two-segment sampler of points from (U[0, 1], U[0, 1])."""
  CAN_TRUNCATE_INPUT_DATA = True

  def _sample_data(self, length: int, low: float = 0., high: float = 1.):
    del length  # There are exactly four endpoints.

    # Quick CCW check (ignoring collinearity) for rejection sampling
    def ccw(x_a, y_a, x_b, y_b, x_c, y_c):
      return (y_c - y_a) * (x_b - x_a) > (y_b - y_a) * (x_c - x_a)
    def intersect(xs, ys):
      return ccw(xs[0], ys[0], xs[2], ys[2], xs[3], ys[3]) != ccw(
          xs[1], ys[1], xs[2], ys[2], xs[3], ys[3]) and ccw(
              xs[0], ys[0], xs[1], ys[1], xs[2], ys[2]) != ccw(
                  xs[0], ys[0], xs[1], ys[1], xs[3], ys[3])

    # Decide (with uniform probability) should this sample intersect
    coin_flip = self._rng.binomial(1, 0.5)

    xs = self._random_sequence(length=4, low=low, high=high)
    ys = self._random_sequence(length=4, low=low, high=high)

    while intersect(xs, ys) != coin_flip:
      xs = self._random_sequence(length=4, low=low, high=high)
      ys = self._random_sequence(length=4, low=low, high=high)

    return [xs, ys]


def _cross2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """Computes the cross product of two 2D vectors.

  Args:
    x: The first 2D vector.
    y: The second 2D vector.

  Returns:
    The cross product of the two vectors.
  """
  return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


def _is_collinear(
    point_1: np.ndarray,
    point_2: np.ndarray,
    point_3: np.ndarray,
    eps: float,
) -> bool:
  """Checks if three points are collinear.

  Args:
    point_1: The first point.
    point_2: The second point.
    point_3: The third point.
    eps: The tolerance for collinearity.

  Returns:
    True if the three points are collinear, False otherwise.

  Raises:
    ValueError: If any of the points is not a 2D vector.
  """
  for point in [point_1, point_2, point_3]:
    if point.shape != (2,):
      raise ValueError(f'Point {point} is not a 2D vector.')

  # Vectors from p1
  v_1 = point_2 - point_1
  v_2 = point_3 - point_1

  cross_val = _cross2d(v_1, v_2)

  return bool(abs(cross_val) < eps)


class ConvexHullSampler(Sampler):
  """Convex hull sampler of points over a disk of radius r."""
  CAN_TRUNCATE_INPUT_DATA = True

  def _sample_data(
      self,
      length: int,
      origin_x: float = 0.0,
      origin_y: float = 0.0,
      radius: float = 2.0,
      collinearity_resampling_attempts: int = 1000,
      collineararity_eps: float = 1e-12,
  ):
    """Samples a convex hull of points over a disk of radius r.

    Args:
      length: The number of points to sample.
      origin_x: The x-coordinate of the origin of the disk.
      origin_y: The y-coordinate of the origin of the disk.
      radius: The radius of the disk.
      collinearity_resampling_attempts: The number of times to resample if
        collinear points are found.
      collineararity_eps: The tolerance for collinearity.

    Returns:
      A list of the sampled points.

    Raises:
      RuntimeError: If it could not sample stable points within the specified
        number of attempts.
    """
    for _ in range(collinearity_resampling_attempts):
      thetas = self._random_sequence(
          length=length,
          low=0.0,
          high=2.0 * np.pi,
      )
      rs = radius * np.sqrt(
          self._random_sequence(length=length, low=0.0, high=1.0)
      )

      xs = rs * np.cos(thetas) + origin_x
      ys = rs * np.sin(thetas) + origin_y

      # Sampler._make_batch may do truncation of the input data after
      # calling _sample_data.
      # Truncation can lead to collinearity of points, which in turn leads to
      # numerous correct traces in the Graham scan algorithm. To prevent this,
      # we check for collinearity and resample if collinear points are found.
      xs = self._trunc_array(xs)
      ys = self._trunc_array(ys)

      collinear_found = False
      points = np.stack([xs, ys], axis=1)
      for point_1, point_2, point_3 in itertools.combinations(points, 3):
        if _is_collinear(point_1, point_2, point_3, collineararity_eps):
          collinear_found = True
          break

      if collinear_found:
        continue

      return [xs, ys]

    raise RuntimeError(
        f'Could not sample {length} stable points within {10000} tries.'
    )


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
    'strongly_connected_components': SccSampler,
    'articulation_points': ArticulationSampler,
    'bridges': ArticulationSampler,
    'bfs': BfsSampler,
    'mst_kruskal': MSTSampler,
    'mst_prim': BellmanFordSampler,
    'bellman_ford': BellmanFordSampler,
    'dag_shortest_paths': DAGPathSampler,
    'dijkstra': BellmanFordSampler,
    'floyd_warshall': FloydWarshallSampler,
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
    for i, dp in enumerate(sample_io):
      assert dp.data.shape[0] == 1  # batching axis
      assert traj_io[0][i].name == dp.name

  return jax.tree_util.tree_map(lambda *x: np.concatenate(x), *traj_io)


def _batch_hints(
    traj_hints: Trajectories, min_steps: int) -> Tuple[Trajectory, List[int]]:
  """Batches a trajectory of hints samples along the time axis per probe.

  Unlike i/o, hints have a variable-length time dimension. Before batching, each
  trajectory is padded to the maximum trajectory length.

  Args:
    traj_hints: A hint trajectory of `DataPoints`s indexed by time then probe
    min_steps: Hints will be padded at least to this length - if any hint is
      longer than this, the greater length will be used.

  Returns:
    A |num probes| list of `DataPoint`s with the time axis stacked into `data`,
    and a |sample| list containing the length of each trajectory.
  """

  max_steps = min_steps
  assert traj_hints  # non-empty
  for sample_hint in traj_hints:
    for dp in sample_hint:
      assert dp.data.shape[1] == 1  # batching axis
      if dp.data.shape[0] > max_steps:
        max_steps = dp.data.shape[0]
  time_and_batch = (max_steps, len(traj_hints))

  # Create zero-filled space for the batched hints, then copy each hint
  # up to the corresponding length.
  batched_traj = jax.tree_util.tree_map(
      lambda x: np.zeros(time_and_batch + x.shape[2:]),
      traj_hints[0])
  hint_lengths = np.zeros(len(traj_hints))

  for sample_idx, cur_sample in enumerate(traj_hints):
    for i in range(len(cur_sample)):
      assert batched_traj[i].name == cur_sample[i].name
      cur_data = cur_sample[i].data
      cur_length = cur_data.shape[0]
      batched_traj[i].data[:cur_length, sample_idx:sample_idx+1] = cur_data
      if i > 0:
        assert hint_lengths[sample_idx] == cur_length
      else:
        hint_lengths[sample_idx] = cur_length
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


def _preprocess_permutations(probes, enforce_permutations):
  """Replace should-be permutations with proper permutation pointer + mask."""
  output = []
  for x in probes:
    if x.type_ != specs.Type.SHOULD_BE_PERMUTATION:
      output.append(x)
      continue
    assert x.location == specs.Location.NODE
    if enforce_permutations:
      new_x, mask = probing.predecessor_to_cyclic_predecessor_and_first(x.data)
      output.append(
          probing.DataPoint(
              name=x.name,
              location=x.location,
              type_=specs.Type.PERMUTATION_POINTER,
              data=new_x))
      output.append(
          probing.DataPoint(
              name=x.name + '_mask',
              location=x.location,
              type_=specs.Type.MASK_ONE,
              data=mask))
    else:
      output.append(probing.DataPoint(name=x.name, location=x.location,
                                      type_=specs.Type.POINTER, data=x.data))
  return output


def process_permutations(spec, sample_iterator, enforce_permutations):
  """Replace should-be permutations with proper permutation pointer + mask."""
  def _iterate():
    while True:
      feedback = next(sample_iterator)
      features = feedback.features
      inputs = _preprocess_permutations(features.inputs, enforce_permutations)
      hints = _preprocess_permutations(features.hints, enforce_permutations)
      outputs = _preprocess_permutations(feedback.outputs, enforce_permutations)
      features = features._replace(inputs=tuple(inputs),
                                   hints=tuple(hints))
      feedback = feedback._replace(features=features,
                                   outputs=outputs)
      yield feedback

  new_spec = {}
  for k in spec:
    if (spec[k][1] == specs.Location.NODE and
        spec[k][2] == specs.Type.SHOULD_BE_PERMUTATION):
      if enforce_permutations:
        new_spec[k] = (spec[k][0], spec[k][1], specs.Type.PERMUTATION_POINTER)
        new_spec[k + '_mask'] = (spec[k][0], spec[k][1], specs.Type.MASK_ONE)
      else:
        new_spec[k] = (spec[k][0], spec[k][1], specs.Type.POINTER)
    else:
      new_spec[k] = spec[k]

  return new_spec, _iterate()


def process_pred_as_input(spec, sample_iterator):
  """Move pred_h hint to pred input."""
  def _iterate():
    while True:
      feedback = next(sample_iterator)
      features = feedback.features
      pred_h = [h for h in features.hints if h.name == 'pred_h']
      if pred_h:
        assert len(pred_h) == 1
        pred_h = pred_h[0]
        hints = [h for h in features.hints if h.name != 'pred_h']
        for i in range(len(features.lengths)):
          assert np.sum(np.abs(pred_h.data[1:int(features.lengths[i]), i] -
                               pred_h.data[0, i])) == 0.0
        inputs = tuple(features.inputs) + (
            probing.DataPoint(name='pred', location=pred_h.location,
                              type_=pred_h.type_, data=pred_h.data[0]),)
        features = features._replace(inputs=tuple(inputs),
                                     hints=tuple(hints))
        feedback = feedback._replace(features=features)
      yield feedback

  new_spec = {}
  for k in spec:
    if k == 'pred_h':
      assert spec[k] == (specs.Stage.HINT, specs.Location.NODE,
                         specs.Type.POINTER)
      new_spec['pred'] = (specs.Stage.INPUT, specs.Location.NODE,
                          specs.Type.POINTER)
    else:
      new_spec[k] = spec[k]

  return new_spec, _iterate()


def process_random_pos(sample_iterator, rng):
  """Randomize the `pos` input from a sampler.

  The `pos` input is, by default, a scalar uniformly spaced between 0 and 1
  across the nodes. The exception are string algorithms (naive_string_matcher,
  kmp_string_matcher and lcs_length), where the `pos` sequence is split into
  needle and haystack (or first and second string, for lcs_length). Here
  we replace the uniformly spaced `pos` with an ordered sequence of random
  scalars, or, for string algorithms, two ordered sequences of random scalars.

  Args:
    sample_iterator: An iterator producing samples with non-random `pos` inputs.
    rng: Numpy random generator
  Returns:
    An iterator returning the samples with randomized `pos` inputs.
  """
  def _iterate():
    while True:
      feedback = next(sample_iterator)
      inputs = feedback.features.inputs
      pos, = [x for x in inputs if x.name == 'pos']
      batch_size, num_nodes = pos.data.shape
      unsorted = rng.uniform(size=(batch_size, num_nodes))
      new_pos = []
      for i in range(batch_size):  # we check one example at a time.
        # We find if there are splits in the pos sequence, marked by zeros.
        # We know there will always be at least 1 zero, if there's no split.
        split, = np.where(pos.data[i] == 0)
        split = np.concatenate([split, [num_nodes]])
        # We construct the randomized pos by sorting the random values in each
        # split and concatenating them.
        new_pos.append(
            np.concatenate([np.sort(unsorted[i, split[j]:split[j+1]])
                            for j in range(len(split) - 1)]))
      pos.data = np.array(new_pos)
      inputs = [(pos if x.name == 'pos' else x) for x in inputs]
      features = feedback.features._replace(inputs=inputs)
      feedback = feedback._replace(features=features)
      yield feedback

  return _iterate()
