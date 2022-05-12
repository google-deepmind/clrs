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
"""Encoder utilities."""

import functools
import chex
from clrs._src import probing
from clrs._src import specs
import haiku as hk
import jax.numpy as jnp

_Array = chex.Array
_DataPoint = probing.DataPoint
_Location = specs.Location
_Spec = specs.Spec
_Type = specs.Type


def construct_encoders(loc: str, t: str, hidden_dim: int, name: str):
  """Constructs encoders."""
  linear = functools.partial(hk.Linear, name=f'{name}_enc_linear')
  encoders = [linear(hidden_dim)]
  if loc == _Location.EDGE and t == _Type.POINTER:
    # Edge pointers need two-way encoders.
    encoders.append(linear(hidden_dim))

  return encoders


def preprocess(dp: _DataPoint, nb_nodes: int) -> _Array:
  """Pre-process data point."""
  if dp.type_ == _Type.POINTER:
    data = hk.one_hot(dp.data, nb_nodes)
  else:
    data = dp.data.astype(jnp.float32)

  return data


def accum_adj_mat(dp: _DataPoint, data: _Array, adj_mat: _Array) -> _Array:
  """Accumulates adjacency matrix."""
  if dp.location == _Location.NODE and dp.type_ == _Type.POINTER:
    adj_mat += ((data + jnp.transpose(data, (0, 2, 1))) > 0.0)
  elif dp.location == _Location.EDGE and dp.type_ == _Type.MASK:
    adj_mat += ((data + jnp.transpose(data, (0, 2, 1))) > 0.0)

  return (adj_mat > 0.).astype('float32')


def accum_edge_fts(encoders, dp: _DataPoint, data: _Array,
                   edge_fts: _Array) -> _Array:
  """Encodes and accumulates edge features."""
  encoding = _encode_inputs(encoders, dp, data)

  if dp.location == _Location.NODE and dp.type_ == _Type.POINTER:
    edge_fts += encoding

  elif dp.location == _Location.EDGE:
    if dp.type_ == _Type.POINTER:
      # Aggregate pointer contributions across sender and receiver nodes.
      encoding_2 = encoders[1](jnp.expand_dims(data, -1))
      edge_fts += jnp.mean(encoding, axis=1) + jnp.mean(encoding_2, axis=2)
    else:
      edge_fts += encoding

  return edge_fts


def accum_node_fts(encoders, dp: _DataPoint, data: _Array,
                   node_fts: _Array) -> _Array:
  """Encodes and accumulates node features."""
  encoding = _encode_inputs(encoders, dp, data)

  if ((dp.location == _Location.NODE and dp.type_ != _Type.POINTER) or
      (dp.location == _Location.GRAPH and dp.type_ == _Type.POINTER)):
    node_fts += encoding

  return node_fts


def accum_graph_fts(encoders, dp: _DataPoint, data: _Array,
                    graph_fts: _Array) -> _Array:
  """Encodes and accumulates graph features."""
  encoding = _encode_inputs(encoders, dp, data)

  if dp.location == _Location.GRAPH and dp.type_ != _Type.POINTER:
    graph_fts += encoding

  return graph_fts


def _encode_inputs(encoders, dp: _DataPoint, data: _Array) -> _Array:
  if dp.type_ == _Type.CATEGORICAL:
    encoding = encoders[0](data)
  else:
    encoding = encoders[0](jnp.expand_dims(data, -1))
  return encoding
