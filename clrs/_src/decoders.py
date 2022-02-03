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
"""decoders utilities."""

from typing import Dict
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

_BIG_NUMBER = 1e5


def construct_decoders(loc: str, t: str, hidden_dim: int, nb_dims: int):
  """Constructs decoders."""
  if loc == _Location.NODE:
    # Node decoders.
    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
      decoders = (hk.Linear(1),)
    elif t == _Type.CATEGORICAL:
      decoders = (hk.Linear(nb_dims),)
    elif t == _Type.POINTER:
      decoders = (hk.Linear(hidden_dim), hk.Linear(hidden_dim))
    else:
      raise ValueError(f"Invalid Type {t}")

  elif loc == _Location.EDGE:
    # Edge decoders.
    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
      decoders = (hk.Linear(1), hk.Linear(1), hk.Linear(1))
    elif t == _Type.CATEGORICAL:
      decoders = (hk.Linear(nb_dims), hk.Linear(nb_dims), hk.Linear(nb_dims))
    elif t == _Type.POINTER:
      decoders = (hk.Linear(hidden_dim), hk.Linear(hidden_dim),
                  hk.Linear(hidden_dim), hk.Linear(hidden_dim))
    else:
      raise ValueError(f"Invalid Type {t}")

  elif loc == _Location.GRAPH:
    # Graph decoders.
    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
      decoders = (hk.Linear(1), hk.Linear(1))
    elif t == _Type.CATEGORICAL:
      decoders = (hk.Linear(nb_dims), hk.Linear(nb_dims))
    elif t == _Type.POINTER:
      decoders = (hk.Linear(hidden_dim), hk.Linear(hidden_dim),
                  hk.Linear(hidden_dim))
    else:
      raise ValueError(f"Invalid Type {t}")

  else:
    raise ValueError(f"Invalid Location {loc}")

  return decoders


def postprocess(spec: _Spec, preds: _Array) -> Dict[str, _DataPoint]:
  """Postprocesses decoder output."""
  result = {}
  for name in preds.keys():
    _, loc, t = spec[name]
    data = preds[name]
    if t == _Type.SCALAR:
      pass
    elif t == _Type.MASK:
      data = (data > 0.0) * 1.0
    elif t in [_Type.MASK_ONE, _Type.CATEGORICAL]:
      cat_size = data.shape[-1]
      best = jnp.argmax(data, -1)
      data = hk.one_hot(best, cat_size)
    elif t == _Type.POINTER:
      data = jnp.argmax(data, -1)
    else:
      raise ValueError("Invalid type")
    result[name] = probing.DataPoint(
        name=name, location=loc, type_=t, data=data)

  return result


def decode_node_fts(decoders, t: str, h_t: _Array, adj_mat: _Array,
                    inf_bias: bool) -> _Array:
  """Decodes node features."""

  if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
    preds = jnp.squeeze(decoders[0](h_t), -1)
  elif t == _Type.CATEGORICAL:
    preds = decoders[0](h_t)
  elif t == _Type.POINTER:
    p_1 = decoders[0](h_t)
    p_2 = decoders[1](h_t)
    ptr_p = jnp.matmul(p_1, jnp.transpose(p_2, (0, 2, 1)))
    preds = ptr_p
    if inf_bias:
      preds -= (1 - adj_mat) * _BIG_NUMBER
  else:
    raise ValueError("Invalid output type")

  return preds


def decode_edge_fts(decoders, t: str, h_t: _Array, edge_fts: _Array,
                    adj_mat: _Array, inf_bias_edge: bool) -> _Array:
  """Decodes edge features."""

  pred_1 = decoders[0](h_t)
  pred_2 = decoders[1](h_t)
  pred_e = decoders[2](edge_fts)
  pred = (
      jnp.expand_dims(pred_1, -2) + jnp.expand_dims(pred_2, -3) + pred_e)
  if t in [
      _Type.SCALAR, _Type.MASK, _Type.MASK_ONE
  ]:
    preds = jnp.squeeze(pred, -1)
  elif t == _Type.CATEGORICAL:
    preds = pred
  elif t == _Type.POINTER:
    pred_2 = jnp.expand_dims(decoders[3](h_t), -1)
    ptr_p = jnp.matmul(pred, jnp.transpose(pred_2, (0, 3, 2, 1)))
    preds = ptr_p
  else:
    raise ValueError("Invalid output type")
  if inf_bias_edge and t in [_Type.MASK, _Type.MASK_ONE]:
    preds -= (1 - adj_mat) * _BIG_NUMBER

  return preds


def decode_graph_fts(decoders, t: str, h_t: _Array,
                     graph_fts: _Array) -> _Array:
  """Decodes graph features."""

  gr_emb = jnp.max(h_t, axis=-2)
  pred_n = decoders[0](gr_emb)
  pred_g = decoders[1](graph_fts)
  pred = pred_n + pred_g
  if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
    preds = jnp.squeeze(pred, -1)
  elif t == _Type.CATEGORICAL:
    preds = pred
  elif t == _Type.POINTER:
    pred_2 = decoders[2](h_t)
    ptr_p = jnp.matmul(
        jnp.expand_dims(pred, 1), jnp.transpose(pred_2, (0, 2, 1)))
    preds = jnp.squeeze(ptr_p, 1)

  return preds
