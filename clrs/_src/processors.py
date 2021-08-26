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

"""JAX implementation of baseline processor networks."""

from typing import Any, Callable, List, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp


_Array = chex.Array
_Fn = Callable[..., Any]


class GAT(hk.Module):
  """Graph Attention Network (Velickovic et al., ICLR 2018)."""

  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      activation: Optional[_Fn] = None,
      residual: bool = True,
      name: str = 'gat_aggr',
  ):
    super().__init__(name=name)
    self.out_size = out_size
    self.nb_heads = nb_heads
    self.activation = activation
    self.residual = residual

  def __call__(
      self,
      features: _Array,
      e_features: _Array,
      g_features: _Array,
      adj: _Array,
  ) -> _Array:
    """GAT inference step.

    Args:
      features: Node features.
      e_features: Edge features.
      g_features: Graph features.
      adj: Graph adjacency matrix.

    Returns:
      Output of GAT inference step.
    """
    b, n, _ = features.shape
    assert e_features.shape[:-1] == (b, n, n)
    assert g_features.shape[:-1] == (b,)
    assert adj.shape == (b, n, n)

    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj - 1.0) * 1e9

    a_1 = hk.Linear(1)
    a_2 = hk.Linear(1)
    a_e = hk.Linear(1)
    a_g = hk.Linear(1)

    values = m(features)

    att_1 = a_1(features)
    att_2 = a_2(features)
    att_e = a_e(e_features)
    att_g = a_g(g_features)

    logits = (
        att_1 + jnp.transpose(att_2, (0, 2, 1)) + jnp.squeeze(att_e, axis=-1) +
        jnp.expand_dims(att_g, axis=-1))
    coefs = jax.nn.softmax(jax.nn.leaky_relu(logits) + bias_mat, axis=-1)
    ret = jnp.matmul(coefs, values)

    if self.residual:
      ret += skip(features)

    if self.activation is not None:
      ret = self.activation(ret)

    return ret


class MPNN(hk.Module):
  """Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

  def __init__(
      self,
      out_size: int,
      mid_size: Optional[int] = None,
      mid_act: Optional[_Fn] = None,
      activation: Optional[_Fn] = None,
      reduction: _Fn = jnp.max,
      msgs_mlp_sizes: Optional[List[int]] = None,
      name: str = 'mpnn_aggr',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
    self.out_size = out_size
    self.mid_act = mid_act
    self.activation = activation
    self.reduction = reduction
    self._msgs_mlp_sizes = msgs_mlp_sizes

  def __call__(
      self,
      features: _Array,
      e_features: _Array,
      g_features: _Array,
      adj: _Array,
  ) -> _Array:
    """MPNN inference step.

    Args:
      features: Node features.
      e_features: Edge features.
      g_features: Graph features.
      adj: Graph adjacency matrix.

    Returns:
      Output of MPNN inference step.
    """
    b, n, _ = features.shape
    assert e_features.shape[:-1] == (b, n, n)
    assert g_features.shape[:-1] == (b,)
    assert adj.shape == (b, n, n)

    m_1 = hk.Linear(self.mid_size)
    m_2 = hk.Linear(self.mid_size)
    m_e = hk.Linear(self.mid_size)
    m_g = hk.Linear(self.mid_size)

    o1 = hk.Linear(self.out_size)
    o2 = hk.Linear(self.out_size)

    msg_1 = m_1(features)
    msg_2 = m_2(features)
    msg_e = m_e(e_features)
    msg_g = m_g(g_features)

    msgs = (
        jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
        msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))
    if self._msgs_mlp_sizes is not None:
      msgs = hk.nets.MLP(self._msgs_mlp_sizes)(jax.nn.relu(msgs))

    if self.mid_act is not None:
      msgs = self.mid_act(msgs)

    if self.reduction == jnp.mean:
      msgs = jnp.sum(msgs * jnp.expand_dims(adj, -1), axis=-1)
      msgs = msgs / jnp.sum(adj, axis=-1, keepdims=True)
    else:
      msgs = self.reduction(msgs * jnp.expand_dims(adj, -1), axis=1)

    h_1 = o1(features)
    h_2 = o2(msgs)

    ret = h_1 + h_2

    if self.activation is not None:
      ret = self.activation(ret)

    return ret
