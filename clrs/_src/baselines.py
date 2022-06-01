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

from typing import Dict, List, Optional, Tuple, Union

import chex

from clrs._src import decoders
from clrs._src import losses
from clrs._src import model
from clrs._src import nets
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
_FeaturesChunked = samplers.FeaturesChunked
_Feedback = samplers.Feedback
_Location = specs.Location
_Seed = jnp.ndarray
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = samplers.Trajectory
_Type = specs.Type
_OutputClass = specs.OutputClass


class BaselineModel(model.Model):
  """Model implementation with selectable message passing algorithm."""

  def __init__(
      self,
      spec: Union[_Spec, List[_Spec]],
      dummy_trajectory: Union[List[_Feedback], _Feedback],
      processor_factory: processors.ProcessorFactory,
      hidden_dim: int = 32,
      encode_hints: bool = False,
      decode_hints: bool = True,
      decode_diffs: bool = False,
      use_lstm: bool = False,
      learning_rate: float = 0.005,
      checkpoint_path: str = '/tmp/clrs3',
      freeze_processor: bool = False,
      dropout_prob: float = 0.0,
      hint_teacher_forcing_noise: float = 0.0,
      name: str = 'base_model',
  ):
    """Constructor for BaselineModel.

    The model consists of encoders, processor and decoders. It can train
    and evaluate either a single algorithm or a set of algorithms; in the
    latter case, a single processor is shared among all the algorithms, while
    the encoders and decoders are separate for each algorithm.

    Args:
      spec: Either a single spec for one algorithm, or a list of specs for
        multiple algorithms to be trained and evaluated.
      dummy_trajectory: Either a single feedback batch, in the single-algorithm
        case, or a list of feedback batches, in the multi-algorithm case, that
        comply with the `spec` (or list of specs), to initialize network size.
      processor_factory: A callable that takes an `out_size` parameter
        and returns a processor (see `processors.py`).
      hidden_dim: Size of the hidden state of the model, i.e., size of the
        message-passing vectors.
      encode_hints: Whether to provide hints as model inputs.
      decode_hints: Whether to provide hints as model outputs.
      decode_diffs: Whether to predict masks within the model.
      use_lstm: Whether to insert an LSTM after message passing.
      learning_rate: Learning rate for training.
      checkpoint_path: Path for loading/saving checkpoints.
      freeze_processor: If True, the processor weights will be frozen and
        only encoders and decoders (and, if used, the lstm) will be trained.
      dropout_prob: Dropout rate in the message-passing stage.
      hint_teacher_forcing_noise: Probability of using predicted hints instead
        of ground-truth hints as inputs during training (only relevant if
        `encode_hints`=True
      name: Model name.

    Raises:
      ValueError: if `encode_hints=True` and `decode_hints=False`.
    """
    super(BaselineModel, self).__init__(spec=spec)

    if encode_hints and not decode_hints:
      raise ValueError('`encode_hints=True`, `decode_hints=False` is invalid.')

    self.decode_hints = decode_hints
    self.decode_diffs = decode_diffs
    self.checkpoint_path = checkpoint_path
    self.name = name
    self._freeze_processor = freeze_processor
    self.opt = optax.adam(learning_rate)

    self.nb_dims = []
    if isinstance(dummy_trajectory, _Feedback):
      assert len(self._spec) == 1
      dummy_trajectory = [dummy_trajectory]
    for traj in dummy_trajectory:
      nb_dims = {}
      for inp in traj.features.inputs:
        nb_dims[inp.name] = inp.data.shape[-1]
      for hint in traj.features.hints:
        nb_dims[hint.name] = hint.data.shape[-1]
      for outp in traj.outputs:
        nb_dims[outp.name] = outp.data.shape[-1]
      self.nb_dims.append(nb_dims)

    self._create_net_fns(hidden_dim, encode_hints, processor_factory, use_lstm,
                         dropout_prob, hint_teacher_forcing_noise)
    self.params = None
    self.opt_state = None
    self.opt_state_skeleton = None

  def _create_net_fns(self, hidden_dim, encode_hints, processor_factory,
                      use_lstm, dropout_prob, hint_teacher_forcing_noise):
    def _use_net(*args, **kwargs):
      return nets.Net(self._spec, hidden_dim, encode_hints,
                      self.decode_hints, self.decode_diffs,
                      processor_factory, use_lstm, dropout_prob,
                      hint_teacher_forcing_noise, self.nb_dims)(*args, **kwargs)

    self.net_fn = hk.transform(_use_net)
    self.net_fn_apply = jax.jit(self.net_fn.apply,
                                static_argnames=['repred', 'algorithm_index'])

  def init(self, features: Union[_Features, List[_Features]], seed: _Seed):
    if not isinstance(features, list):
      assert len(self._spec) == 1
      features = [features]
    self.params = self.net_fn.init(jax.random.PRNGKey(seed), features, True, -1)
    self.opt_state = self.opt.init(self.params)
    # We will use the optimizer state skeleton for traversal when we
    # want to avoid updating the state of params of untrained algorithms.
    self.opt_state_skeleton = self.opt.init(jnp.zeros(1))

  def feedback(self, rng_key: hk.PRNGSequence, feedback: _Feedback,
               algorithm_index: Optional[int] = None) -> float:
    """Advance to the next task, incorporating any available feedback."""
    if algorithm_index is None:
      assert len(self._spec) == 1
      algorithm_index = 0

    self.params, self.opt_state, cur_loss = self.update(
        rng_key, self.params, self.opt_state, feedback, algorithm_index)
    return cur_loss

  def predict(self, rng_key: hk.PRNGSequence, features: _Features,
              algorithm_index: Optional[int] = None):
    """Model inference step."""
    if algorithm_index is None:
      assert len(self._spec) == 1
      algorithm_index = 0

    outs, hint_preds, diff_logits, gt_diff = self.net_fn_apply(
        self.params, rng_key, [features],
        repred=True, algorithm_index=algorithm_index)
    return decoders.postprocess(self._spec[algorithm_index],
                                outs), (hint_preds, diff_logits, gt_diff)

  def update(
      self,
      rng_key: hk.PRNGSequence,
      params: hk.Params,
      opt_state: optax.OptState,
      feedback: _Feedback,
      algorithm_index: Optional[int] = None,
  ) -> Tuple[hk.Params, optax.OptState, _Array]:
    """Model update step."""

    if algorithm_index is None:
      assert len(self._spec) == 1
      algorithm_index = 0

    def loss(params, rng_key, feedback):
      """Calculates model loss f(feedback; params)."""
      (output_preds, hint_preds, diff_logits,
       gt_diffs) = self.net_fn_apply(params, rng_key, [feedback.features],
                                     repred=False,
                                     algorithm_index=algorithm_index)

      nb_nodes = _nb_nodes(feedback, is_chunked=False)
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
    assert algorithm_index >= 0
    lss, grads = jax.value_and_grad(loss)(params, rng_key, feedback)
    new_params, opt_state = self._update_params(params, grads, opt_state)

    return new_params, opt_state, lss

  def _update_params(self, params, grads, opt_state):
    updates, opt_state = filter_null_grads(
        grads, self.opt, opt_state, self.opt_state_skeleton)
    if self._freeze_processor:
      params_subset = _filter_processor(params)
      updates_subset = _filter_processor(updates)
      new_params = optax.apply_updates(params_subset, updates_subset)
      new_params = hk.data_structures.merge(params, new_params)
    else:
      new_params = optax.apply_updates(params, updates)

    return new_params, opt_state

  def verbose_loss(self, feedback: _Feedback, extra_info) -> Dict[str, _Array]:
    """Gets verbose loss information."""
    hint_preds, diff_logits, gt_diffs = extra_info

    nb_nodes = _nb_nodes(feedback, is_chunked=False)
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
                preds=[x[truth.name] for x in hint_preds],
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


class BaselineModelChunked(BaselineModel):
  """Model that processes time-chunked data.

    Unlike `BaselineModel`, which processes full samples, `BaselineModelChunked`
    processes fixed-timelength chunks of data. Each tensor of inputs and hints
    has dimensions chunk_length x batch_size x ... The beginning of a new
    sample withing the chunk is signalled by a tensor called `is_first` of
    dimensions chunk_length x batch_size.

    The chunked model is intended for training. For validation and test, use
    `BaselineModel`.
  """

  def _create_net_fns(self, hidden_dim, encode_hints, processor_factory,
                      use_lstm, dropout_prob, hint_teacher_forcing_noise):
    def _use_net(*args, **kwargs):
      return nets.NetChunked(
          self._spec, hidden_dim, encode_hints,
          self.decode_hints, self.decode_diffs,
          processor_factory, use_lstm, dropout_prob,
          hint_teacher_forcing_noise, self.nb_dims)(*args, **kwargs)

    self.net_fn = hk.transform(_use_net)
    self.net_fn_apply = jax.jit(
        functools.partial(self.net_fn.apply, init_mp_state=False),
        static_argnames=['repred', 'algorithm_index'])

  def _init_mp_state(self, features_list: List[_FeaturesChunked],
                     rng_key: _Array):
    def _empty_mp_state():
      return nets.MessagePassingStateChunked(
          inputs=None, hints=None, is_first=None,
          hint_preds=None, hiddens=None, lstm_state=None)
    empty_mp_states = [_empty_mp_state() for _ in range(len(features_list))]
    dummy_params = self.net_fn.init(
        rng_key, features_list, empty_mp_states, False,
        init_mp_state=True, algorithm_index=-1)
    _, mp_states = self.net_fn.apply(
        dummy_params, rng_key, features_list, empty_mp_states, False,
        init_mp_state=True, algorithm_index=-1)
    return mp_states

  def init(self, features: Union[_FeaturesChunked, List[_FeaturesChunked]],
           seed: _Seed):
    if not isinstance(features, list):
      assert len(self._spec) == 1
      features = [features]
    self.mp_states = self._init_mp_state(features,
                                         jax.random.PRNGKey(seed))
    self.params = self.net_fn.init(
        jax.random.PRNGKey(seed), features, self.mp_states,
        True, init_mp_state=False, algorithm_index=-1)
    self.opt_state = self.opt.init(self.params)
    # We will use the optimizer state skeleton for traversal when we
    # want to avoid updating the state of params of untrained algorithms.
    self.opt_state_skeleton = self.opt.init(jnp.zeros(1))

  def predict(self, rng_key: hk.PRNGSequence, features: _FeaturesChunked,
              algorithm_index: Optional[int] = None):
    """Inference not implemented. Chunked model intended for training only."""
    raise NotImplementedError

  def update(
      self,
      rng_key: hk.PRNGSequence,
      params: hk.Params,
      opt_state: optax.OptState,
      feedback: _Feedback,
      algorithm_index: Optional[int] = None,
  ) -> Tuple[hk.Params, optax.OptState, _Array]:
    """Model update step."""

    if algorithm_index is None:
      assert len(self._spec) == 1
      algorithm_index = 0
    def loss(params, rng_key, feedback):
      ((output_preds, hint_preds, diff_logits, gt_diffs),
       mp_state) = self.net_fn_apply(params, rng_key, [feedback.features],
                                     [self.mp_states[algorithm_index]],
                                     repred=False,
                                     algorithm_index=algorithm_index)

      nb_nodes = _nb_nodes(feedback, is_chunked=True)

      total_loss = 0.0
      is_first = feedback.features.is_first
      is_last = feedback.features.is_last

      # Calculate output loss.
      for truth in feedback.outputs:
        total_loss += losses.output_loss_chunked(
            truth=truth,
            pred=output_preds[truth.name],
            is_last=is_last,
            nb_nodes=nb_nodes,
        )

      # Optionally accumulate diff losses.
      if self.decode_diffs:
        total_loss += losses.diff_loss_chunked(
            diff_logits=diff_logits,
            gt_diffs=gt_diffs,
            is_first=is_first,
        )

      # Optionally accumulate hint losses.
      if self.decode_hints:
        for truth in feedback.features.hints:
          loss = losses.hint_loss_chunked(
              truth=truth,
              pred=hint_preds[truth.name],
              gt_diffs=gt_diffs,
              is_first=is_first,
              nb_nodes=nb_nodes,
              decode_diffs=self.decode_diffs,
          )
          total_loss += loss

      return total_loss, (mp_state,)

    (lss, (self.mp_states[algorithm_index],)), grads = jax.value_and_grad(
        loss, has_aux=True)(params, rng_key, feedback)
    new_params, opt_state = self._update_params(params, grads, opt_state)

    return new_params, opt_state, lss

  def verbose_loss(self, *args, **kwargs):
    raise NotImplementedError


def _nb_nodes(feedback: _Feedback, is_chunked) -> int:
  for inp in feedback.features.inputs:
    if inp.location in [_Location.NODE, _Location.EDGE]:
      if is_chunked:
        return inp.data.shape[2]  # inputs are time x batch x nodes x ...
      else:
        return inp.data.shape[1]  # inputs are batch x nodes x ...
  assert False


def _filter_processor(params: hk.Params) -> hk.Params:
  return hk.data_structures.filter(
      lambda module_name, n, v: 'construct_processor' in module_name, params)


def _is_not_done_broadcast(lengths, i, tensor):
  is_not_done = (lengths > i + 1) * 1.0
  while len(is_not_done.shape) < len(tensor.shape):
    is_not_done = jnp.expand_dims(is_not_done, -1)
  return is_not_done


def filter_null_grads(grads, opt, opt_state, opt_state_skeleton):
  """Compute updates ignoring params that have no gradients.

  This prevents untrained params (e.g., encoders/decoders for algorithms
  that are not being trained) to accumulate, e.g., momentum from spurious
  zero gradients.

  Note: this works as intended for "per-parameter" optimizer state, such as
    momentum. However, when the optimizer has some global state (such as the
    step counts in Adam), the global state will be updated every time,
    affecting also future updates of parameters that had null gradients in the
    current step.

  Args:
    grads: Gradients for all parameters.
    opt: Optax optmizier.
    opt_state: Optimizer state.
    opt_state_skeleton: A "skeleton" of optimizer state that has been
      initialized with scalar parameters. This serves to traverse each parameter
      of the otpimizer state during the opt state update.
  Returns:
    Updates and new optimizer state, where the parameters with null gradient
      have not been taken into account.
  """
  # Ignore params with no gradient.
  masked_grads = jax.tree_map(lambda x: x if jnp.abs(x).sum() > 0.0 else None,
                              grads)
  flat_grads, treedef = jax.tree_flatten(masked_grads)
  flat_opt_state = jax.tree_map(
      lambda _, x: treedef.flatten_up_to(x) if not isinstance(x, _Array) else x,
      opt_state_skeleton, opt_state)

  # Compute updates only for the params with gradient.
  flat_updates, flat_opt_state = opt.update(flat_grads, flat_opt_state)

  def unflatten(flat, original):
    """Restore tree structure, filling missing (None) leaves with original."""
    if isinstance(flat, _Array):
      return flat
    return jax.tree_map(lambda x, y: x if y is None else y,
                        original, treedef.unflatten(flat))

  # Restore the state and updates tree structure.
  new_opt_state = jax.tree_map(
      lambda _, x, y: unflatten(x, y),
      opt_state_skeleton, flat_opt_state, opt_state)
  updates = unflatten(flat_updates, jax.tree_map(lambda x: 0., grads))
  return updates, new_opt_state
