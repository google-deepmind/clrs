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

import functools
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
_Stage = specs.Stage
_Type = specs.Type

_BIG_NUMBER = 1e5


def construct_decoders(loc: str, t: str, hidden_dim: int, nb_dims: int,
                       name: str):
  """Constructs decoders."""
  linear = functools.partial(hk.Linear, name=f"{name}_dec_linear")
  if loc == _Location.NODE:
    # Node decoders.
    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
      decoders = (linear(1),)
    elif t == _Type.CATEGORICAL:
      decoders = (linear(nb_dims),)
    elif t == _Type.POINTER:
      decoders = (linear(hidden_dim), linear(hidden_dim))
    else:
      raise ValueError(f"Invalid Type {t}")

  elif loc == _Location.EDGE:
    # Edge decoders.
    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
      decoders = (linear(1), linear(1), linear(1))
    elif t == _Type.CATEGORICAL:
      decoders = (linear(nb_dims), linear(nb_dims), linear(nb_dims))
    elif t == _Type.POINTER:
      decoders = (linear(hidden_dim), linear(hidden_dim),
                  linear(hidden_dim), linear(hidden_dim))
    else:
      raise ValueError(f"Invalid Type {t}")

  elif loc == _Location.GRAPH:
    # Graph decoders.
    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
      decoders = (linear(1), linear(1))
    elif t == _Type.CATEGORICAL:
      decoders = (linear(nb_dims), linear(nb_dims))
    elif t == _Type.POINTER:
      decoders = (linear(hidden_dim), linear(hidden_dim),
                  linear(hidden_dim))
    else:
      raise ValueError(f"Invalid Type {t}")

  else:
    raise ValueError(f"Invalid Location {loc}")

  return decoders


def construct_diff_decoders(name: str):
  """Constructs diff decoders."""
  linear = functools.partial(hk.Linear, name=f"{name}_diffdec_linear")
  decoders = {}
  decoders[_Location.NODE] = linear(1)
  decoders[_Location.EDGE] = (linear(1), linear(1), linear(1))
  decoders[_Location.GRAPH] = (linear(1), linear(1))

  return decoders


def postprocess(spec: _Spec, preds: Dict[str, _Array]) -> Dict[str, _DataPoint]:
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


def decode_fts(
    decoders,
    spec: _Spec,
    h_t: _Array,
    adj_mat: _Array,
    edge_fts: _Array,
    graph_fts: _Array,
    inf_bias: bool,
    inf_bias_edge: bool,
):
  """Decodes node, edge and graph features."""
  output_preds = {}
  hint_preds = {}

  for name in decoders:
    decoder = decoders[name]
    stage, loc, t = spec[name]

    if loc == _Location.NODE:
      preds = _decode_node_fts(decoder, t, h_t, adj_mat, inf_bias)
    elif loc == _Location.EDGE:
      preds = _decode_edge_fts(decoder, t, h_t, edge_fts, adj_mat,
                               inf_bias_edge)
    elif loc == _Location.GRAPH:
      preds = _decode_graph_fts(decoder, t, h_t, graph_fts)
    else:
      raise ValueError("Invalid output type")

    if stage == _Stage.OUTPUT:
      output_preds[name] = preds
    elif stage == _Stage.HINT:
      hint_preds[name] = preds
    else:
      raise ValueError(f"Found unexpected decoder {name}")

  return hint_preds, output_preds


def _decode_node_fts(decoders, t: str, h_t: _Array, adj_mat: _Array,
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


def _decode_edge_fts(decoders, t: str, h_t: _Array, edge_fts: _Array,
                     adj_mat: _Array, inf_bias_edge: bool) -> _Array:
  """Decodes edge features."""

  pred_1 = decoders[0](h_t)
  pred_2 = decoders[1](h_t)
  pred_e = decoders[2](edge_fts)
  pred = (jnp.expand_dims(pred_1, -2) + jnp.expand_dims(pred_2, -3) + pred_e)
  if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
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


def _decode_graph_fts(decoders, t: str, h_t: _Array,
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


def maybe_decode_diffs(
    diff_decoders,
    h_t: _Array,
    edge_fts: _Array,
    graph_fts: _Array,
    batch_size: int,
    nb_nodes: int,
    decode_diffs: bool,
) -> Dict[str, _Array]:
  """Optionally decodes node, edge and graph diffs."""

  if decode_diffs:
    preds = {}
    node = _Location.NODE
    edge = _Location.EDGE
    graph = _Location.GRAPH
    preds[node] = _decode_node_diffs(diff_decoders[node], h_t)
    preds[edge] = _decode_edge_diffs(diff_decoders[edge], h_t, edge_fts)
    preds[graph] = _decode_graph_diffs(diff_decoders[graph], h_t, graph_fts)

  else:
    preds = {
        _Location.NODE: jnp.ones((batch_size, nb_nodes)),
        _Location.EDGE: jnp.ones((batch_size, nb_nodes, nb_nodes)),
        _Location.GRAPH: jnp.ones((batch_size))
    }

  return preds


def _decode_node_diffs(decoders, h_t: _Array) -> _Array:
  """Decodes node diffs."""
  return jnp.squeeze(decoders(h_t), -1)


def _decode_edge_diffs(decoders, h_t: _Array, edge_fts: _Array) -> _Array:
  """Decodes edge diffs."""

  e_pred_1 = decoders[0](h_t)
  e_pred_2 = decoders[1](h_t)
  e_pred_e = decoders[2](edge_fts)
  preds = jnp.squeeze(
      jnp.expand_dims(e_pred_1, -1) + jnp.expand_dims(e_pred_2, -2) + e_pred_e,
      -1,
  )

  return preds


def _decode_graph_diffs(decoders, h_t: _Array, graph_fts: _Array) -> _Array:
  """Decodes graph diffs."""

  gr_emb = jnp.max(h_t, axis=-2)
  g_pred_n = decoders[0](gr_emb)
  g_pred_g = decoders[1](graph_fts)
  preds = jnp.squeeze(g_pred_n + g_pred_g, -1)

  return preds
