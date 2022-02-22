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

"""JAX implementation of CLRS baseline models."""

import functools
import os
import pickle

from typing import Dict, Tuple, List, Optional

import chex

from clrs._src import decoders
from clrs._src import encoders
from clrs._src import losses
from clrs._src import model
from clrs._src import probing
from clrs._src import processors
from clrs._src import samplers
from clrs._src import specs

import haiku as hk
import jax
import jax.numpy as jnp
import optax


_Array = chex.Array
_DataPoint = probing.DataPoint
_Features = samplers.Features
_Feedback = samplers.Feedback
_Location = specs.Location
_Seed = jnp.ndarray
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = samplers.Trajectory
_Type = specs.Type
_OutputClass = specs.OutputClass


@chex.dataclass
class _MessagePassingScanState:
  hint_preds: chex.Array
  diff_logits: chex.Array
  gt_diffs: chex.Array
  output_preds: chex.Array
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
          _Location.NODE: jnp.zeros((self.batch_size, nb_nodes)),
          _Location.EDGE: jnp.zeros((self.batch_size, nb_nodes, nb_nodes)),
          _Location.GRAPH: jnp.zeros((self.batch_size))
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
     lstm_state) = self._one_step_pred(
         inputs, cur_hint, mp_state.hiddens, nb_nodes, mp_state.lstm_state)

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

    for inp in inputs:
      if inp.location in [_Location.NODE, _Location.EDGE]:
        self.batch_size = inp.data.shape[0]
        nb_nodes = inp.data.shape[1]
        break

    # Construct encoders and decoders.
    (self.encoders, self.decoders,
     self.diff_decoders) = self._construct_encoders_decoders()
    self.processor = processors.construct_processor(
        kind=self.kind, hidden_dim=self.hidden_dim, nb_heads=self.nb_heads)

    nb_mp_steps = max(1, hints[0].data.shape[0] - 1)
    hiddens = jnp.zeros((self.batch_size, nb_nodes, self.hidden_dim))

    # Optionally construct LSTM.
    if self.use_lstm:
      self.lstm = hk.LSTM(
          hidden_size=self.hidden_dim,
          name='processor_lstm')
      lstm_state = self.lstm.initial_state(self.batch_size * nb_nodes)
      lstm_state = jax.tree_multimap(
          lambda x: jnp.reshape(x, [self.batch_size, nb_nodes, -1]), lstm_state)
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
        nb_nodes=nb_nodes,
        lengths=lengths)

    # Then scan through the rest.
    scan_fn = functools.partial(
        self._msg_passing_step,
        first_step=False,
        hints=hints,
        repred=repred,
        inputs=inputs,
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

    return encoders_, decoders_, diff_decoders

  def _one_step_pred(
      self,
      inputs: _Trajectory,
      hints: _Trajectory,
      hidden: _Array,
      nb_nodes: int,
      lstm_state: Optional[hk.LSTMState],
  ):
    """Generates one-step predictions."""

    # Initialise empty node/edge/graph features and adjacency matrix.
    node_fts = jnp.zeros((self.batch_size, nb_nodes, self.hidden_dim))
    edge_fts = jnp.zeros((self.batch_size, nb_nodes, nb_nodes, self.hidden_dim))
    graph_fts = jnp.zeros((self.batch_size, self.hidden_dim))
    adj_mat = jnp.repeat(
        jnp.expand_dims(jnp.eye(nb_nodes), 0), self.batch_size, axis=0)

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
        batch_size=self.batch_size,
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
        batch_size=self.batch_size,
        nb_nodes=nb_nodes,
        decode_diffs=self.decode_diffs,
    )

    return nxt_hidden, output_preds, hint_preds, diff_preds, nxt_lstm_state


class BaselineModel(model.Model):
  """Model implementation with selectable message passing algorithm."""

  def __init__(
      self,
      spec,
      nb_heads=1,
      hidden_dim=32,
      kind='mpnn',
      encode_hints=False,
      decode_hints=True,
      decode_diffs=False,
      use_lstm=False,
      learning_rate=0.005,
      checkpoint_path='/tmp/clrs3',
      freeze_processor=False,
      dummy_trajectory=None,
      dropout_prob=0.0,
      name='base_model',
  ):
    super(BaselineModel, self).__init__(spec=spec)

    if encode_hints and not decode_hints:
      raise ValueError('`encode_hints=True`, `decode_hints=False` is invalid.')

    self.spec = spec
    self.decode_hints = decode_hints
    self.decode_diffs = decode_diffs
    self.checkpoint_path = checkpoint_path
    self.name = name
    self._freeze_processor = freeze_processor
    self.opt = optax.adam(learning_rate)

    self.nb_dims = {}
    for inp in dummy_trajectory.features.inputs:
      self.nb_dims[inp.name] = inp.data.shape[-1]
    for hint in dummy_trajectory.features.hints:
      self.nb_dims[hint.name] = hint.data.shape[-1]
    for outp in dummy_trajectory.outputs:
      self.nb_dims[outp.name] = outp.data.shape[-1]

    def _use_net(*args, **kwargs):
      return Net(spec, hidden_dim, encode_hints, decode_hints, decode_diffs,
                 kind, use_lstm, dropout_prob, nb_heads, self.nb_dims)(*args,
                                                                       **kwargs)

    self.net_fn = hk.transform(_use_net)
    self.net_fn_apply = jax.jit(self.net_fn.apply, static_argnums=3)
    self.params = None
    self.opt_state = None

  def init(self, features: _Features, seed: _Seed):
    self.params = self.net_fn.init(jax.random.PRNGKey(seed), features, True)
    self.opt_state = self.opt.init(self.params)

  def feedback(self, rng_key: hk.PRNGSequence, feedback: _Feedback) -> float:
    """Advance to the next task, incorporating any available feedback."""
    self.params, self.opt_state, cur_loss = self.update(
        rng_key, self.params, self.opt_state, feedback)
    return cur_loss

  def predict(self, rng_key: hk.PRNGSequence, features: _Features):
    """Model inference step."""
    outs, hint_preds, diff_logits, gt_diff = self.net_fn_apply(
        self.params, rng_key, features, True)
    return decoders.postprocess(self.spec,
                                outs), (hint_preds, diff_logits, gt_diff)

  def update(
      self,
      rng_key: hk.PRNGSequence,
      params: hk.Params,
      opt_state: optax.OptState,
      feedback: _Feedback,
  ) -> Tuple[hk.Params, optax.OptState, _Array]:
    """Model update step."""

    def loss(params, rng_key, feedback):
      """Calculates model loss f(feedback; params)."""
      (output_preds, hint_preds, diff_logits,
       gt_diffs) = self.net_fn_apply(params, rng_key, feedback.features, False)

      nb_nodes = _nb_nodes(feedback)
      lengths = feedback.features.lengths
      total_loss = 0.0

      # Calculate output loss.
      for truth in feedback.outputs:
        total_loss += losses.output_loss(
            truth=truth,
            pred=output_preds[truth.name],
            nb_nodes=nb_nodes,
        )

      # Optionally accumulate diff losses.
      if self.decode_diffs:
        total_loss += losses.diff_loss(
            diff_logits=diff_logits,
            gt_diffs=gt_diffs,
            lengths=lengths,
        )

      # Optionally accumulate hint losses.
      if self.decode_hints:
        for truth in feedback.features.hints:
          total_loss += losses.hint_loss(
              truth=truth,
              preds=[x[truth.name] for x in hint_preds],
              gt_diffs=gt_diffs,
              lengths=lengths,
              nb_nodes=nb_nodes,
              decode_diffs=self.decode_diffs,
          )

      return total_loss

    # Calculate and apply gradients.
    lss, grads = jax.value_and_grad(loss)(params, rng_key, feedback)
    updates, opt_state = self.opt.update(grads, opt_state)
    if self._freeze_processor:
      params_subset = _filter_processor(params)
      updates_subset = _filter_processor(updates)
      new_params = optax.apply_updates(params_subset, updates_subset)
      new_params = hk.data_structures.merge(params, new_params)
    else:
      new_params = optax.apply_updates(params, updates)

    return new_params, opt_state, lss

  def verbose_loss(self, feedback: _Feedback, extra_info) -> Dict[str, _Array]:
    """Gets verbose loss information."""
    hint_preds, diff_logits, gt_diffs = extra_info

    nb_nodes = _nb_nodes(feedback)
    lengths = feedback.features.lengths
    losses_ = {}

    # Optionally accumulate diff losses.
    if self.decode_diffs:
      losses_.update(
          losses.diff_loss(
              diff_logits=diff_logits,
              gt_diffs=gt_diffs,
              lengths=lengths,
              verbose=True,
          ))

    # Optionally accumulate hint losses.
    if self.decode_hints:
      for truth in feedback.features.hints:
        losses_.update(
            losses.hint_loss(
                truth=truth,
                preds=hint_preds,
                gt_diffs=gt_diffs,
                lengths=lengths,
                nb_nodes=nb_nodes,
                decode_diffs=self.decode_diffs,
                verbose=True,
            ))

    return losses_

  def restore_model(self, file_name: str, only_load_processor: bool = False):
    """Restore model from `file_name`."""
    path = os.path.join(self.checkpoint_path, file_name)
    with open(path, 'rb') as f:
      restored_state = pickle.load(f)
      if only_load_processor:
        restored_params = _filter_processor(restored_state['params'])
      else:
        restored_params = restored_state['params']
      self.params = hk.data_structures.merge(self.params, restored_params)
      self.opt_state = restored_state['opt_state']

  def save_model(self, file_name: str):
    """Save model (processor weights only) to `file_name`."""
    os.makedirs(self.checkpoint_path, exist_ok=True)
    to_save = {'params': self.params, 'opt_state': self.opt_state}
    path = os.path.join(self.checkpoint_path, file_name)
    with open(path, 'wb') as f:
      pickle.dump(to_save, f)


def _nb_nodes(feedback: _Feedback):
  for inp in feedback.features.inputs:
    if inp.location in [_Location.NODE, _Location.EDGE]:
      return inp.data.shape[1]
  assert False


def _filter_processor(params: hk.Params) -> hk.Params:
  return hk.data_structures.filter(
      lambda module_name, n, v: 'construct_processor' in module_name, params)


def _is_not_done_broadcast(lengths, i, tensor):
  is_not_done = (lengths > i + 1) * 1.0
  while len(is_not_done.shape) < len(tensor.shape):
    is_not_done = jnp.expand_dims(is_not_done, -1)
  return is_not_done
