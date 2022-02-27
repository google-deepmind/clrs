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

"""JAX implementation of CLRS basic network."""

import functools

from typing import List, Optional, Tuple

import chex

from clrs._src import decoders
from clrs._src import encoders
from clrs._src import probing
from clrs._src import processors
from clrs._src import samplers
from clrs._src import specs

import haiku as hk
import jax
import jax.numpy as jnp


_Array = chex.Array
_DataPoint = probing.DataPoint
_Features = samplers.Features
_FeaturesChunked = samplers.FeaturesChunked
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = samplers.Trajectory
_Type = specs.Type


@chex.dataclass
class _MessagePassingScanState:
  hint_preds: chex.Array
  diff_logits: chex.Array
  gt_diffs: chex.Array
  output_preds: chex.Array
  hiddens: chex.Array
  lstm_state: Optional[hk.LSTMState]


@chex.dataclass
class _MessagePassingOutputChunked:
  hint_preds: chex.Array
  diff_logits: chex.Array
  gt_diffs: chex.Array
  output_preds: chex.Array


@chex.dataclass
class MessagePassingStateChunked:
  inputs: chex.Array
  hints: chex.Array
  is_first: chex.Array
  hint_preds: chex.Array
  hiddens: chex.Array
  lstm_state: Optional[hk.LSTMState]


class Net(hk.Module):
  """Building blocks (networks) used to encode and decode messages."""

  def __init__(
      self,
      spec: _Spec,
      hidden_dim: int,
      encode_hints: bool,
      decode_hints: bool,
      decode_diffs: bool,
      kind: str,
      use_lstm: bool,
      dropout_prob: float,
      nb_heads: int,
      nb_dims=None,
      name: str = 'net',
  ):
    """Constructs a `Net`."""
    super().__init__(name=name)

    self._dropout_prob = dropout_prob
    self.spec = spec
    self.hidden_dim = hidden_dim
    self.encode_hints = encode_hints
    self.decode_hints = decode_hints
    self.decode_diffs = decode_diffs
    self.kind = kind
    self.nb_dims = nb_dims
    self.use_lstm = use_lstm
    self.nb_heads = nb_heads

  def _msg_passing_step(self,
                        mp_state: _MessagePassingScanState,
                        i: int,
                        hints: List[_DataPoint],
                        repred: bool,
                        lengths: chex.Array,
                        batch_size: int,
                        nb_nodes: int,
                        inputs: _Trajectory,
                        first_step: bool = False):
    if (not first_step) and repred and self.decode_hints:
      decoded_hint = decoders.postprocess(self.spec, mp_state.hint_preds)
      cur_hint = []
      for hint in decoded_hint:
        cur_hint.append(decoded_hint[hint])
    else:
      cur_hint = []
      for hint in hints:
        hint.data = jnp.asarray(hint.data)
        _, loc, typ = self.spec[hint.name]
        cur_hint.append(
            probing.DataPoint(
                name=hint.name, location=loc, type_=typ, data=hint.data[i]))

    gt_diffs = None
    if hints[0].data.shape[0] > 1 and self.decode_diffs:
      gt_diffs = {
          _Location.NODE: jnp.zeros((batch_size, nb_nodes)),
          _Location.EDGE: jnp.zeros((batch_size, nb_nodes, nb_nodes)),
          _Location.GRAPH: jnp.zeros((batch_size))
      }
      for hint in hints:
        hint_cur = jax.lax.dynamic_index_in_dim(hint.data, i, 0, keepdims=False)
        hint_nxt = jax.lax.dynamic_index_in_dim(
            hint.data, i+1, 0, keepdims=False)
        if len(hint_cur.shape) == len(gt_diffs[hint.location].shape):
          hint_cur = jnp.expand_dims(hint_cur, -1)
          hint_nxt = jnp.expand_dims(hint_nxt, -1)
        gt_diffs[hint.location] += jnp.any(hint_cur != hint_nxt, axis=-1)
      for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
        gt_diffs[loc] = (gt_diffs[loc] > 0.0).astype(jnp.float32) * 1.0

    (hiddens, output_preds_cand, hint_preds, diff_logits,
     lstm_state) = self._one_step_pred(inputs, cur_hint, mp_state.hiddens,
                                       batch_size, nb_nodes,
                                       mp_state.lstm_state)

    if first_step:
      output_preds = output_preds_cand
    else:
      output_preds = {}
      for outp in mp_state.output_preds:
        is_not_done = _is_not_done_broadcast(lengths, i,
                                             output_preds_cand[outp])
        output_preds[outp] = is_not_done * output_preds_cand[outp] + (
            1.0 - is_not_done) * mp_state.output_preds[outp]

    if self.decode_diffs:
      if self.decode_hints:
        if hints[0].data.shape[0] == 1 or repred:
          diff_preds = {}
          for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
            diff_preds[loc] = (diff_logits[loc] > 0.0).astype(jnp.float32) * 1.0
        else:
          diff_preds = gt_diffs
        for hint in hints:
          prev_hint = (
              hint.data[0]
              if first_step else mp_state.hint_preds[hint.name])
          if first_step and hint.type_ == _Type.POINTER:
            prev_hint = hk.one_hot(prev_hint, nb_nodes)
          cur_diffs = diff_preds[hint.location]
          while len(prev_hint.shape) > len(cur_diffs.shape):
            cur_diffs = jnp.expand_dims(cur_diffs, -1)
          hint_preds[hint.name] = (
              cur_diffs * hint_preds[hint.name] + (1.0 - cur_diffs) * prev_hint)

    new_mp_state = _MessagePassingScanState(
        hint_preds=hint_preds, diff_logits=diff_logits, gt_diffs=gt_diffs,
        output_preds=output_preds, hiddens=hiddens, lstm_state=lstm_state)

    # Complying to jax.scan, the first returned value is the state we carry over
    # the second value is the output that will be stacked over steps.
    return new_mp_state, new_mp_state

  def __call__(self, features: _Features, repred: bool):
    """Network inference step."""
    inputs = features.inputs
    hints = features.hints
    lengths = features.lengths

    batch_size, nb_nodes = _data_dimensions(features)

    (self.encoders, self.decoders,
     self.diff_decoders) = self._construct_encoders_decoders()
    self.processor = processors.construct_processor(
        kind=self.kind, hidden_dim=self.hidden_dim, nb_heads=self.nb_heads)

    nb_mp_steps = max(1, hints[0].data.shape[0] - 1)
    hiddens = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))

    # Optionally construct LSTM.
    if self.use_lstm:
      self.lstm = hk.LSTM(
          hidden_size=self.hidden_dim,
          name='processor_lstm')
      lstm_state = self.lstm.initial_state(batch_size * nb_nodes)
      lstm_state = jax.tree_multimap(
          lambda x: jnp.reshape(x, [batch_size, nb_nodes, -1]), lstm_state)
    else:
      self.lstm = None
      lstm_state = None

    mp_state = _MessagePassingScanState(
        hint_preds=None, diff_logits=None, gt_diffs=None,
        output_preds=None, hiddens=hiddens, lstm_state=lstm_state)

    # Do the first step outside of the scan because it has a different
    # computation graph.
    mp_state, _ = self._msg_passing_step(
        mp_state,
        i=0,
        first_step=True,
        hints=hints,
        repred=repred,
        inputs=inputs,
        batch_size=batch_size,
        nb_nodes=nb_nodes,
        lengths=lengths)

    # Then scan through the rest.
    scan_fn = functools.partial(
        self._msg_passing_step,
        first_step=False,
        hints=hints,
        repred=repred,
        inputs=inputs,
        batch_size=batch_size,
        nb_nodes=nb_nodes,
        lengths=lengths)

    _, output_mp_state = hk.scan(
        scan_fn,
        mp_state,
        jnp.arange(nb_mp_steps - 1) + 1,
        length=nb_mp_steps - 1)

    output_mp_state = jax.tree_multimap(
        lambda init, tail: jnp.concatenate([init[None], tail], axis=0),
        mp_state, output_mp_state)

    def invert(d):
      """Dict of lists -> list of dicts."""
      if d:
        return [dict(zip(d, i)) for i in zip(*d.values())]

    output_preds = invert(output_mp_state.output_preds)
    hint_preds = invert(output_mp_state.hint_preds)
    diff_logits = invert(output_mp_state.diff_logits)
    gt_diffs = invert(output_mp_state.gt_diffs)

    return output_preds[-1], hint_preds, diff_logits, gt_diffs

  def _construct_encoders_decoders(self):
    """Constructs encoders and decoders."""
    encoders_ = {}
    decoders_ = {}

    for name, (stage, loc, t) in self.spec.items():
      if stage == _Stage.INPUT or (stage == _Stage.HINT and self.encode_hints):
        # Build input encoders.
        encoders_[name] = encoders.construct_encoders(
            loc, t, hidden_dim=self.hidden_dim)

      if stage == _Stage.OUTPUT or (stage == _Stage.HINT and self.decode_hints):
        # Build output decoders.
        decoders_[name] = decoders.construct_decoders(
            loc, t, hidden_dim=self.hidden_dim, nb_dims=self.nb_dims[name])

    if self.decode_diffs:
      # Optionally build diff decoders.
      diff_decoders = decoders.construct_diff_decoders()
    else:
      diff_decoders = None

    return encoders_, decoders_, diff_decoders

  def _one_step_pred(
      self,
      inputs: _Trajectory,
      hints: _Trajectory,
      hidden: _Array,
      batch_size: int,
      nb_nodes: int,
      lstm_state: Optional[hk.LSTMState],
  ):
    """Generates one-step predictions."""

    # Initialise empty node/edge/graph features and adjacency matrix.
    node_fts = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))
    edge_fts = jnp.zeros((batch_size, nb_nodes, nb_nodes, self.hidden_dim))
    graph_fts = jnp.zeros((batch_size, self.hidden_dim))
    adj_mat = jnp.repeat(
        jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)

    # ENCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Encode node/edge/graph features from inputs and (optionally) hints.
    trajectories = [inputs]
    if self.encode_hints:
      trajectories.append(hints)

    for trajectory in trajectories:
      for dp in trajectory:
        try:
          data = encoders.preprocess(dp, nb_nodes)
          adj_mat = encoders.accum_adj_mat(dp, data, adj_mat)
          encoder = self.encoders[dp.name]
          edge_fts = encoders.accum_edge_fts(encoder, dp, data, edge_fts)
          node_fts = encoders.accum_node_fts(encoder, dp, data, node_fts)
          graph_fts = encoders.accum_graph_fts(encoder, dp, data, graph_fts)
        except Exception as e:
          raise Exception(f'Failed to process {dp}') from e

    # PROCESS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nxt_hidden = self.processor(
        node_fts,
        edge_fts,
        graph_fts,
        adj_mat,
        hidden,
        batch_size=batch_size,
        nb_nodes=nb_nodes,
    )
    nxt_hidden = hk.dropout(hk.next_rng_key(), self._dropout_prob, nxt_hidden)

    if self.use_lstm:
      # lstm doesn't accept multiple batch dimensions (in our case, batch and
      # nodes), so we vmap over the (first) batch dimension.
      nxt_hidden, nxt_lstm_state = jax.vmap(self.lstm)(nxt_hidden, lstm_state)
    else:
      nxt_lstm_state = None

    h_t = jnp.concatenate([node_fts, hidden, nxt_hidden], axis=-1)

    # DECODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Decode features and (optionally) hints.
    hint_preds, output_preds = decoders.decode_fts(
        decoders=self.decoders,
        spec=self.spec,
        h_t=h_t,
        adj_mat=adj_mat,
        edge_fts=edge_fts,
        graph_fts=graph_fts,
        inf_bias=self.processor.inf_bias,
        inf_bias_edge=self.processor.inf_bias_edge,
    )

    # Optionally decode diffs.
    diff_preds = decoders.maybe_decode_diffs(
        diff_decoders=self.diff_decoders,
        h_t=h_t,
        edge_fts=edge_fts,
        graph_fts=graph_fts,
        batch_size=batch_size,
        nb_nodes=nb_nodes,
        decode_diffs=self.decode_diffs,
    )

    return nxt_hidden, output_preds, hint_preds, diff_preds, nxt_lstm_state


class NetChunked(Net):
  """A Net that will process time-chunked data instead of full samples."""

  def _msg_passing_step(self,
                        mp_state: MessagePassingStateChunked,
                        xs,
                        repred: bool,
                        init_mp_state: bool,
                        batch_size: int,
                        nb_nodes: int,
                        ):
    """Perform one message passing step.

    This function is unrolled along the time axis to process a data chunk.

    Args:
      mp_state: message-passing state. Includes the inputs, hints,
        beginning-of-sample markers, hint predictions, hidden and lstm state
        to be used for prediction in the current step.
      xs: A 3-tuple of with the next timestep's inputs, hints, and
        beginning-of-sample markers. These will replace the contents of
        the `mp_state` at the output, in readiness for the next unroll step of
        the chunk (or the first step of the next chunk). Besides, the next
        timestep's hints are necessary to compute diffs when `decode_diffs`
        is True.
      repred: False during training, when we have access to ground-truth hints
        and diffs. True in validation/test mode, when we have to use our own
        hint and diff predictions.
      init_mp_state: Indicates if we are calling the method just to initialise
        the message-passing state, before the beginning of training or
        validation.
      batch_size: Size of batch dimension.
      nb_nodes: Number of nodes in graph.
    Returns:
      A 2-tuple with the next mp_state and an output consisting of
      hint predictions, diff logits, ground-truth diffs, and output predictions.
      The diffs are between the next-step data (provided in `xs`) and the
      current-step data (provided in `mp_state`).
    """
    def _as_prediction_data(hint):
      if hint.type_ == _Type.POINTER:
        return hk.one_hot(hint.data, nb_nodes)
      return hint.data

    nxt_inputs, nxt_hints, nxt_is_first = xs
    inputs = mp_state.inputs
    is_first = mp_state.is_first
    hints = mp_state.hints
    if init_mp_state:
      prev_hint_preds = {h.name: _as_prediction_data(h) for h in hints}
      hints_for_pred = hints
    else:
      prev_hint_preds = mp_state.hint_preds
      if repred and self.decode_hints:
        decoded_hints = decoders.postprocess(self.spec, prev_hint_preds)
        hints_for_pred = []
        for h in hints:
          hints_for_pred.append(probing.DataPoint(
              name=h.name, location=h.location, type_=h.type_,
              data=jnp.where(_expand_to(is_first, h.data),
                             h.data, decoded_hints[h.name].data)))
      else:
        hints_for_pred = hints

    gt_diffs = None
    if self.decode_diffs:
      gt_diffs = {
          _Location.NODE: jnp.zeros((batch_size, nb_nodes)),
          _Location.EDGE: jnp.zeros((batch_size, nb_nodes, nb_nodes)),
          _Location.GRAPH: jnp.zeros((batch_size))
      }
      for hint, nxt_hint in zip(hints, nxt_hints):
        data = hint.data
        nxt_data = nxt_hint.data
        if len(nxt_data.shape) == len(gt_diffs[hint.location].shape):
          data = jnp.expand_dims(data, -1)
          nxt_data = jnp.expand_dims(nxt_data, -1)
        gt_diffs[hint.location] += jnp.any(data != nxt_data, axis=-1)
      for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
        gt_diffs[loc] = (gt_diffs[loc] > 0.0).astype(jnp.float32) * 1.0

    hiddens = jnp.where(is_first[..., None, None], 0.0, mp_state.hiddens)
    if self.use_lstm:
      lstm_state = jax.tree_map(
          lambda x: jnp.where(is_first[..., None, None], 0.0, x),
          mp_state.lstm_state)
    else:
      lstm_state = None
    (hiddens, output_preds, hint_preds, diff_logits,
     lstm_state) = self._one_step_pred(inputs, hints_for_pred, hiddens,
                                       batch_size, nb_nodes, lstm_state)

    if self.decode_diffs and self.decode_hints:
      # Only output a hint predicted for this step if a difference
      # happened in the ground-truth hint - or, at test time, if a difference
      # was predicted for the hint. Otherwise replace the predicted hint with
      # the hint predicted at the previous step.
      if repred:
        diff_preds = jax.tree_map(lambda x: x > 0.0, diff_logits)
      else:
        diff_preds = gt_diffs
      for hint in hints:
        hint_data = _as_prediction_data(hint)
        prev_hint = jnp.where(_expand_to(is_first, hint_data),
                              hint_data, prev_hint_preds[hint.name])
        cur_diffs = _expand_to(diff_preds[hint.location], hint_data)
        hint_preds[hint.name] = jnp.where(cur_diffs,
                                          hint_preds[hint.name],
                                          prev_hint)

    new_mp_state = MessagePassingStateChunked(
        hiddens=hiddens, lstm_state=lstm_state, hint_preds=hint_preds,
        inputs=nxt_inputs, hints=nxt_hints, is_first=nxt_is_first)
    mp_output = _MessagePassingOutputChunked(
        hint_preds=hint_preds, diff_logits=diff_logits, gt_diffs=gt_diffs,
        output_preds=output_preds)
    return new_mp_state, mp_output

  def __call__(self, features: _FeaturesChunked,
               mp_state: MessagePassingStateChunked,
               repred: bool, init_mp_state: bool):
    """Process one chunk of data.

    Args:
      features: Inputs, hints and beginning- and end-of-sample markers for
        a chunk (i.e., fixed time length) of data. All features are expected
        to have dimensions chunk_length x batch_size x ...
      mp_state: message-passing state. Includes the inputs, hints,
        beginning-of-sample markers, hint prediction, hidden and lstm state
        from the end of the previous chunk.
      repred: False during training, when we have access to ground-truth hints
        and diffs. True in validation/test mode, when we have to use our own
        hint and diff predictions.
      init_mp_state: Indicates if we are calling the network just to initialise
        the message-passing state, before the beginning of training or
        validation.

    Returns:
      A 2-tuple consisting of:
      - A 4-tuple with (output predictions, hint predictions, diff logits,
        ground-truth diffs). Each of these has chunk_length x batch_size x ...
        data, where the first time slice contains outputs for the mp_state
        that was passed as input, and the last time slice contains outputs
        for the next-to-last slice of the input features. The outputs that
        correspond to the final time slice of the input features will be
        calculated when the next chunk is processed, using the data in the
        mp_state returned here (see below).
      - The mp_state (message-passing state) for the next chunk of data.
    """
    inputs = features.inputs
    hints = features.hints
    is_first = features.is_first

    batch_size, nb_nodes = _data_dimensions_chunked(features)

    (self.encoders, self.decoders,
     self.diff_decoders) = self._construct_encoders_decoders()
    self.processor = processors.construct_processor(
        kind=self.kind, hidden_dim=self.hidden_dim, nb_heads=self.nb_heads)

    if init_mp_state:
      if self.use_lstm:
        self.lstm = hk.LSTM(
            hidden_size=self.hidden_dim,
            name='processor_lstm')
        lstm_state = self.lstm.initial_state(batch_size * nb_nodes)
        lstm_state = jax.tree_multimap(
            lambda x: jnp.reshape(x, [batch_size, nb_nodes, -1]), lstm_state)
        mp_state.lstm_state = lstm_state
      mp_state.inputs = jax.tree_map(lambda x: x[0], inputs)
      mp_state.hints = jax.tree_map(lambda x: x[0], hints)
      mp_state.is_first = jnp.zeros(batch_size, dtype=int)
      mp_state.hiddens = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))
      next_is_first = jnp.ones(batch_size, dtype=int)

      mp_state, _ = self._msg_passing_step(
          mp_state,
          (mp_state.inputs, mp_state.hints, next_is_first),
          repred=repred,
          init_mp_state=True,
          batch_size=batch_size,
          nb_nodes=nb_nodes)
      scan_output = _MessagePassingOutputChunked(
          hint_preds=None, diff_logits=None, gt_diffs=None, output_preds=None)
    else:
      if self.use_lstm:
        self.lstm = hk.LSTM(
            hidden_size=self.hidden_dim,
            name='processor_lstm')
      else:
        self.lstm = None

      scan_fn = functools.partial(
          self._msg_passing_step,
          repred=repred,
          init_mp_state=False,
          batch_size=batch_size,
          nb_nodes=nb_nodes)

      mp_state, scan_output = hk.scan(
          scan_fn,
          mp_state,
          (inputs, hints, is_first),
      )

    return (scan_output.output_preds, scan_output.hint_preds,
            scan_output.diff_logits, scan_output.gt_diffs), mp_state


def _data_dimensions(features: _Features) -> Tuple[int, int]:
  """Returns (batch_size, nb_nodes)."""
  for inp in features.inputs:
    if inp.location in [_Location.NODE, _Location.EDGE]:
      return inp.data.shape[:2]
  assert False


def _data_dimensions_chunked(features: _FeaturesChunked) -> Tuple[int, int]:
  """Returns (batch_size, nb_nodes)."""
  for inp in features.inputs:
    if inp.location in [_Location.NODE, _Location.EDGE]:
      return inp.data.shape[1:3]
  assert False


def _expand_to(x: _Array, y: _Array) -> _Array:
  while len(y.shape) > len(x.shape):
    x = jnp.expand_dims(x, -1)
  return x


def _is_not_done_broadcast(lengths, i, tensor):
  is_not_done = (lengths > i + 1) * 1.0
  while len(is_not_done.shape) < len(tensor.shape):
    is_not_done = jnp.expand_dims(is_not_done, -1)
  return is_not_done
