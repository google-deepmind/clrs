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

"""Probing utilities.

The dataflow for an algorithm is represented by `(stage, loc, type, data)`
"probes" that are valid under that algorithm's spec (see `specs.py`).

When constructing probes, it is convenient to represent these fields in a nested
format (`ProbesDict`) to facilate efficient contest-based look-up.

"""

import functools
from typing import Dict, List, Tuple, Union

import attr
from clrs._src import specs
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


_Location = specs.Location
_Stage = specs.Stage
_Type = specs.Type
_OutputClass = specs.OutputClass

_Array = np.ndarray
_Data = Union[_Array, List[_Array]]
_DataOrType = Union[_Data, str]

ProbesDict = Dict[
    str, Dict[str, Dict[str, Dict[str, _DataOrType]]]]


def _convert_to_str(element):
  if isinstance(element, tf.Tensor):
    return element.numpy().decode('utf-8')
  elif isinstance(element, (np.ndarray, bytes)):
    return element.decode('utf-8')
  else:
    return element


# First anotation makes this object jax.jit/pmap friendly, second one makes this
# tf.data.Datasets friendly.
@jax.tree_util.register_pytree_node_class
@attr.define
class DataPoint:
  """Describes a data point."""

  _name: str
  _location: str
  _type_: str
  data: _Array

  @property
  def name(self):
    return _convert_to_str(self._name)

  @property
  def location(self):
    return _convert_to_str(self._location)

  @property
  def type_(self):
    return _convert_to_str(self._type_)

  def __repr__(self):
    s = f'DataPoint(name="{self.name}",\tlocation={self.location},\t'
    return s + f'type={self.type_},\tdata=Array{self.data.shape})'

  def tree_flatten(self):
    data = (self.data,)
    meta = (self.name, self.location, self.type_)
    return data, meta

  @classmethod
  def tree_unflatten(cls, meta, data):
    name, location, type_ = meta
    subdata, = data
    return DataPoint(name, location, type_, subdata)


class ProbeError(Exception):
  pass


def initialize(spec: specs.Spec) -> ProbesDict:
  """Initializes an empty `ProbesDict` corresponding with the provided spec."""
  probes = dict()
  for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
    probes[stage] = {}
    for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
      probes[stage][loc] = {}

  for name in spec:
    stage, loc, t = spec[name]
    probes[stage][loc][name] = {}
    probes[stage][loc][name]['data'] = []
    probes[stage][loc][name]['type_'] = t
  # Pytype thinks initialize() returns a ProbesDict with a str for all final
  # values instead of _DataOrType.
  return probes  # pytype: disable=bad-return-type


def push(probes: ProbesDict, stage: str, next_probe):
  """Pushes a probe into an existing `ProbesDict`."""
  for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
    for name in probes[stage][loc]:
      if name not in next_probe:
        raise ProbeError(f'Missing probe for {name}.')
      if isinstance(probes[stage][loc][name]['data'], _Array):
        raise ProbeError('Attemping to push to finalized `ProbesDict`.')
      # Pytype thinks initialize() returns a ProbesDict with a str for all final
      # values instead of _DataOrType.
      probes[stage][loc][name]['data'].append(next_probe[name])  # pytype: disable=attribute-error


def finalize(probes: ProbesDict):
  """Finalizes a `ProbesDict` by stacking/squeezing `data` field."""
  for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
    for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
      for name in probes[stage][loc]:
        if isinstance(probes[stage][loc][name]['data'], _Array):
          raise ProbeError('Attemping to re-finalize a finalized `ProbesDict`.')
        if stage == _Stage.HINT:
          # Hints are provided for each timestep. Stack them here.
          probes[stage][loc][name]['data'] = np.stack(
              probes[stage][loc][name]['data'])
        else:
          # Only one instance of input/output exist. Remove leading axis.
          probes[stage][loc][name]['data'] = np.squeeze(
              np.array(probes[stage][loc][name]['data']))


def split_stages(
    probes: ProbesDict,
    spec: specs.Spec,
) -> Tuple[List[DataPoint], List[DataPoint], List[DataPoint]]:
  """Splits contents of `ProbesDict` into `DataPoint`s by stage."""

  inputs = []
  outputs = []
  hints = []

  for name in spec:
    stage, loc, t = spec[name]

    if stage not in probes:
      raise ProbeError(f'Missing stage {stage}.')
    if loc not in probes[stage]:
      raise ProbeError(f'Missing location {loc}.')
    if name not in probes[stage][loc]:
      raise ProbeError(f'Missing probe {name}.')
    if 'type_' not in probes[stage][loc][name]:
      raise ProbeError(f'Probe {name} missing attribute `type_`.')
    if 'data' not in probes[stage][loc][name]:
      raise ProbeError(f'Probe {name} missing attribute `data`.')
    if t != probes[stage][loc][name]['type_']:
      raise ProbeError(f'Probe {name} of incorrect type {t}.')

    data = probes[stage][loc][name]['data']
    if not isinstance(probes[stage][loc][name]['data'], _Array):
      raise ProbeError((f'Invalid `data` for probe "{name}". ' +
                        'Did you forget to call `probing.finalize`?'))

    if t in [_Type.MASK, _Type.MASK_ONE, _Type.CATEGORICAL]:
      # pytype: disable=attribute-error
      if not ((data == 0) | (data == 1) | (data == -1)).all():
        raise ProbeError(f'0|1|-1 `data` for probe "{name}"')
      # pytype: enable=attribute-error
      if t in [_Type.MASK_ONE, _Type.CATEGORICAL
              ] and not np.all(np.sum(np.abs(data), -1) == 1):
        raise ProbeError(f'Expected one-hot `data` for probe "{name}"')

    dim_to_expand = 1 if stage == _Stage.HINT else 0
    data_point = DataPoint(name=name, location=loc, type_=t,
                           data=np.expand_dims(data, dim_to_expand))

    if stage == _Stage.INPUT:
      inputs.append(data_point)
    elif stage == _Stage.OUTPUT:
      outputs.append(data_point)
    else:
      hints.append(data_point)

  return inputs, outputs, hints


# pylint: disable=invalid-name


def array(A_pos: np.ndarray) -> np.ndarray:
  """Constructs an `array` probe."""
  probe = np.arange(A_pos.shape[0])
  for i in range(1, A_pos.shape[0]):
    probe[A_pos[i]] = A_pos[i - 1]
  return probe


def array_cat(A: np.ndarray, n: int) -> np.ndarray:
  """Constructs an `array_cat` probe."""
  assert n > 0
  probe = np.zeros((A.shape[0], n))
  for i in range(A.shape[0]):
    probe[i, A[i]] = 1
  return probe


def heap(A_pos: np.ndarray, heap_size: int) -> np.ndarray:
  """Constructs a `heap` probe."""
  assert heap_size > 0
  probe = np.arange(A_pos.shape[0])
  for i in range(1, heap_size):
    probe[A_pos[i]] = A_pos[(i - 1) // 2]
  return probe


def graph(A: np.ndarray) -> np.ndarray:
  """Constructs a `graph` probe."""
  probe = (A != 0) * 1.0
  probe = ((A + np.eye(A.shape[0])) != 0) * 1.0
  return probe


def mask_one(i: int, n: int) -> np.ndarray:
  """Constructs a `mask_one` probe."""
  assert n > i
  probe = np.zeros(n)
  probe[i] = 1
  return probe


def strings_id(T_pos: np.ndarray, P_pos: np.ndarray) -> np.ndarray:
  """Constructs a `strings_id` probe."""
  probe_T = np.zeros(T_pos.shape[0])
  probe_P = np.ones(P_pos.shape[0])
  return np.concatenate([probe_T, probe_P])


def strings_pair(pair_probe: np.ndarray) -> np.ndarray:
  """Constructs a `strings_pair` probe."""
  n = pair_probe.shape[0]
  m = pair_probe.shape[1]
  probe_ret = np.zeros((n + m, n + m))
  for i in range(0, n):
    for j in range(0, m):
      probe_ret[i, j + n] = pair_probe[i, j]
  return probe_ret


def strings_pair_cat(pair_probe: np.ndarray, nb_classes: int) -> np.ndarray:
  """Constructs a `strings_pair_cat` probe."""
  assert nb_classes > 0
  n = pair_probe.shape[0]
  m = pair_probe.shape[1]

  # Add an extra class for 'this cell left blank.'
  probe_ret = np.zeros((n + m, n + m, nb_classes + 1))
  for i in range(0, n):
    for j in range(0, m):
      probe_ret[i, j + n, int(pair_probe[i, j])] = _OutputClass.POSITIVE

  # Fill the blank cells.
  for i_1 in range(0, n):
    for i_2 in range(0, n):
      probe_ret[i_1, i_2, nb_classes] = _OutputClass.MASKED
  for j_1 in range(0, m):
    for x in range(0, n + m):
      probe_ret[j_1 + n, x, nb_classes] = _OutputClass.MASKED
  return probe_ret


def strings_pi(T_pos: np.ndarray, P_pos: np.ndarray,
               pi: np.ndarray) -> np.ndarray:
  """Constructs a `strings_pi` probe."""
  probe = np.arange(T_pos.shape[0] + P_pos.shape[0])
  for j in range(P_pos.shape[0]):
    probe[T_pos.shape[0] + P_pos[j]] = T_pos.shape[0] + pi[P_pos[j]]
  return probe


def strings_pos(T_pos: np.ndarray, P_pos: np.ndarray) -> np.ndarray:
  """Constructs a `strings_pos` probe."""
  probe_T = np.copy(T_pos) * 1.0 / T_pos.shape[0]
  probe_P = np.copy(P_pos) * 1.0 / P_pos.shape[0]
  return np.concatenate([probe_T, probe_P])


def strings_pred(T_pos: np.ndarray, P_pos: np.ndarray) -> np.ndarray:
  """Constructs a `strings_pred` probe."""
  probe = np.arange(T_pos.shape[0] + P_pos.shape[0])
  for i in range(1, T_pos.shape[0]):
    probe[T_pos[i]] = T_pos[i - 1]
  for j in range(1, P_pos.shape[0]):
    probe[T_pos.shape[0] + P_pos[j]] = T_pos.shape[0] + P_pos[j - 1]
  return probe


@functools.partial(jnp.vectorize, signature='(n)->(n,n),(n)')
def predecessor_to_cyclic_predecessor_and_first(
    pointers: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Converts predecessor pointers to cyclic predecessor + first node mask.

  This function assumes that the pointers represent a linear order of the nodes
  (akin to a linked list), where each node points to its predecessor and the
  first node points to itself. It returns the same pointers, except that
  the first node points to the last, and a mask_one marking the first node.

  Example:
  ```
  pointers = [2, 1, 1]
  P = [[0, 0, 1],
       [1, 0, 0],
       [0, 1, 0]],
  M = [0, 1, 0]
  ```

  Args:
    pointers: array of shape [N] containing pointers. The pointers are assumed
      to describe a linear order such that `pointers[i]` is the predecessor
      of node `i`.

  Returns:
    Permutation pointers `P` of shape [N] and one-hot vector `M` of shape [N].
  """
  nb_nodes = pointers.shape[-1]
  pointers_one_hot = jax.nn.one_hot(pointers, nb_nodes)
  # Find the index of the last node: it's the node that no other node points to.
  last = pointers_one_hot.sum(-2).argmin()
  # Find the first node: should be the only one pointing to itself.
  first = pointers_one_hot.diagonal().argmax()
  mask = jax.nn.one_hot(first, nb_nodes)
  pointers_one_hot += mask[..., None] * jax.nn.one_hot(last, nb_nodes)
  pointers_one_hot -= mask[..., None] * mask
  return pointers_one_hot, mask
