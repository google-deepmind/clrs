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
import numpy as np


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


class GATv2(hk.Module):
  """Graph Attention Network v2 (Brody et al., ICLR 2022)."""

  def __init__(
      self,
      out_size: int,
      nb_heads: int,
      mid_size: Optional[int] = None,
      activation: Optional[_Fn] = None,
      residual: bool = True,
      name: str = 'gatv2_aggr',
  ):
    super().__init__(name=name)
    if mid_size is None:
      self.mid_size = out_size
    else:
      self.mid_size = mid_size
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
    """GATv2 inference step.

    Args:
      features: Node features.
      e_features: Edge features.
      g_features: Graph features.
      adj: Graph adjacency matrix.

    Returns:
      Output of GATv2 inference step.
    """
    b, n, _ = features.shape
    assert e_features.shape[:-1] == (b, n, n)
    assert g_features.shape[:-1] == (b,)
    assert adj.shape == (b, n, n)

    m = hk.Linear(self.out_size)
    skip = hk.Linear(self.out_size)

    bias_mat = (adj - 1.0) * 1e9

    w_1 = hk.Linear(self.mid_size)
    w_2 = hk.Linear(self.mid_size)
    w_e = hk.Linear(self.mid_size)
    w_g = hk.Linear(self.mid_size)

    a = hk.Linear(1)

    values = m(features)

    pre_att_1 = w_1(features)
    pre_att_2 = w_2(features)
    pre_att_e = w_e(e_features)
    pre_att_g = w_g(g_features)

    pre_att = (
        jnp.expand_dims(pre_att_1, axis=1) + jnp.expand_dims(
            pre_att_2, axis=2) + pre_att_e + jnp.expand_dims(
                pre_att_g, axis=(1, 2)))

    logits = jnp.squeeze(a(jax.nn.leaky_relu(pre_att)), axis=-1)

    coefs = jax.nn.softmax(logits + bias_mat, axis=-1)
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


def _position_encoding(sentence_size: int, embedding_size: int) -> np.ndarray:
  """Position Encoding described in section 4.1 [1]."""
  encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
  ls = sentence_size + 1
  le = embedding_size + 1
  for i in range(1, le):
    for j in range(1, ls):
      encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
  encoding = 1 + 4 * encoding / embedding_size / sentence_size
  return np.transpose(encoding)


class MemNet(hk.Module):
  """Implementation of End-to-End Memory Networks.

  Inspired by the description in https://arxiv.org/abs/1503.08895.
  """

  def __init__(
      self,
      vocab_size: int,
      embedding_size: int,
      sentence_size: int,
      linear_output_size: int,
      memory_size: Optional[int] = None,
      num_hops: int = 1,
      nonlin: Callable[[Any], Any] = jax.nn.relu,
      apply_embeddings: bool = True,
      init_func: hk.initializers.Initializer = jnp.zeros,
      name: str = 'memnet') -> None:
    """Constructor.

    Args:
      vocab_size: the number of words in the dictionary (each story, query and
        answer come contain symbols coming from this dictionary).
      embedding_size: the dimensionality of the latent space to where all
        memories are projected.
      sentence_size: the dimensionality of each memory.
      linear_output_size: the dimensionality of the output of the last layer
        of the model.
      memory_size: the number of memories provided.
      num_hops: the number of layers in the model.
      nonlin: non-linear transformation applied at the end of each layer.
      apply_embeddings: flag whether to aply embeddings.
      init_func: initialization function for the biases.
      name: the name of the model.
    """
    super().__init__(name=name)
    self._vocab_size = vocab_size
    self._embedding_size = embedding_size
    self._sentence_size = sentence_size
    self._memory_size = memory_size
    self._linear_output_size = linear_output_size
    self._num_hops = num_hops
    self._nonlin = nonlin
    self._apply_embeddings = apply_embeddings
    self._init_func = init_func
    # Encoding part: i.e. "I" of the paper.
    self._encodings = _position_encoding(sentence_size, embedding_size)

  def __call__(self, queries: jnp.ndarray, stories: jnp.ndarray) -> jnp.ndarray:
    """Apply Memory Network to the queries and stories.

    Args:
      queries: Tensor of shape [batch_size, sentence_size].
      stories: Tensor of shape [batch_size, memory_size, sentence_size].

    Returns:
      Tensor of shape [batch_size, vocab_size].
    """
    if self._apply_embeddings:
      query_biases = hk.get_parameter(
          'query_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)
      stories_biases = hk.get_parameter(
          'stories_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)
      memory_biases = hk.get_parameter(
          'memory_contents',
          shape=[self._memory_size, self._embedding_size],
          init=self._init_func)
      output_biases = hk.get_parameter(
          'output_biases',
          shape=[self._vocab_size - 1, self._embedding_size],
          init=self._init_func)

      nil_word_slot = jnp.zeros([1, self._embedding_size])

    # This is "A" in the paper.
    if self._apply_embeddings:
      stories_biases = jnp.concatenate([stories_biases, nil_word_slot], axis=0)
      memory_embeddings = jnp.take(
          stories_biases, stories.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(stories.shape) + [self._embedding_size])
      memory = jnp.sum(memory_embeddings * self._encodings, 2) + memory_biases
    else:
      memory = stories

    # This is "B" in the paper. Also, when there are no queries (only
    # sentences), then there these lines are substituted by
    # query_embeddings = 0.1.
    if self._apply_embeddings:
      query_biases = jnp.concatenate([query_biases, nil_word_slot], axis=0)
      query_embeddings = jnp.take(
          query_biases, queries.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(queries.shape) + [self._embedding_size])
      # This is "u" in the paper.
      query_input_embedding = jnp.sum(query_embeddings * self._encodings, 1)
    else:
      query_input_embedding = queries

    # This is "C" in the paper.
    if self._apply_embeddings:
      output_biases = jnp.concatenate([output_biases, nil_word_slot], axis=0)
      output_embeddings = jnp.take(
          output_biases, stories.reshape([-1]).astype(jnp.int32),
          axis=0).reshape(list(stories.shape) + [self._embedding_size])
      output = jnp.sum(output_embeddings * self._encodings, 2)
    else:
      output = stories

    intermediate_linear = hk.Linear(self._embedding_size, with_bias=False)

    # Output_linear is "H".
    output_linear = hk.Linear(self._linear_output_size, with_bias=False)

    for hop_number in range(self._num_hops):
      query_input_embedding_transposed = jnp.transpose(
          jnp.expand_dims(query_input_embedding, -1), [0, 2, 1])

      # Calculate probabilities.
      probs = jax.nn.softmax(
          jnp.sum(memory * query_input_embedding_transposed, 2))

      # Calculate output of the layer by multiplying by C.
      transposed_probs = jnp.transpose(jnp.expand_dims(probs, -1), [0, 2, 1])
      transposed_output_embeddings = jnp.transpose(output, [0, 2, 1])

      # This is "o" in the paper.
      layer_output = jnp.sum(transposed_output_embeddings * transposed_probs, 2)

      # Finally the answer
      if hop_number == self._num_hops - 1:
        # Please note that in the TF version we apply the final linear layer
        # in all hops and this results in shape mismatches.
        output_layer = output_linear(query_input_embedding + layer_output)
      else:
        output_layer = intermediate_linear(query_input_embedding + layer_output)

      query_input_embedding = output_layer
      if self._nonlin:
        output_layer = self._nonlin(output_layer)

    # This linear here is "W".
    return hk.Linear(self._vocab_size, with_bias=False)(output_layer)
