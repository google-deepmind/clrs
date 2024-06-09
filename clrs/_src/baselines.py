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
import numpy as np
import optax


_Array = chex.Array
_DataPoint = probing.DataPoint
_Features = samplers.Features
_FeaturesChunked = samplers.FeaturesChunked
_Feedback = samplers.Feedback
_Location = specs.Location
_Seed = jnp.integer
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = samplers.Trajectory
_Type = specs.Type
_OutputClass = specs.OutputClass

# pytype: disable=signature-mismatch


def _maybe_pick_first_pmapped(tree):
  if jax.local_device_count() == 1:
    return tree
  return jax.tree_util.tree_map(lambda x: x[0], tree)


@jax.jit
def _restack_from_pmap(tree):
  """Stack the results of a pmapped computation across the first two axes."""
  restack_array = lambda x: jnp.reshape(x, (-1,) + x.shape[2:])
  return jax.tree_util.tree_map(restack_array, tree)


def _maybe_restack_from_pmap(tree):
  if jax.local_device_count() == 1:
    return tree
  return _restack_from_pmap(tree)


@functools.partial(jax.jit, static_argnums=[1, 2])
def _pmap_reshape(x, n_devices, split_axis=0):
  """Splits a pytree over n_devices on axis split_axis for pmapping."""
  def _reshape(arr):
    new_shape = (arr.shape[:split_axis] +
                 (n_devices, arr.shape[split_axis] // n_devices) +
                 arr.shape[split_axis + 1:])
    return jnp.moveaxis(jnp.reshape(arr, new_shape), split_axis, 0)
  return jax.tree_util.tree_map(_reshape, x)


def _maybe_pmap_reshape(x, split_axis=0):
  n_devices = jax.local_device_count()
  if n_devices == 1:
    return x
  return _pmap_reshape(x, n_devices, split_axis)


@functools.partial(jax.jit, static_argnums=1)
def _pmap_data(data: Union[_Feedback, _Features], n_devices: int):
  """Replicate/split feedback or features for pmapping."""
  if isinstance(data, _Feedback):
    features = data.features
  else:
    features = data
  pmap_data = features._replace(
      inputs=_pmap_reshape(features.inputs, n_devices),
      hints=_pmap_reshape(features.hints, n_devices, split_axis=1),
      lengths=_pmap_reshape(features.lengths, n_devices),
  )
  if isinstance(data, _Feedback):
    pmap_data = data._replace(
        features=pmap_data,
        outputs=_pmap_reshape(data.outputs, n_devices)
    )
  return pmap_data


def _maybe_pmap_data(data: Union[_Feedback, _Features]):
  n_devices = jax.local_device_count()
  if n_devices == 1:
    return data
  return _pmap_data(data, n_devices)


def _maybe_put_replicated(tree):
  if jax.local_device_count() == 1:
    return jax.device_put(tree)
  else:
    return jax.device_put_replicated(tree, jax.local_devices())


def _maybe_pmap_rng_key(rng_key: _Array):
  n_devices = jax.local_device_count()
  if n_devices == 1:
    return rng_key
  pmap_rng_keys = jax.random.split(rng_key, n_devices)
  return jax.device_put_sharded(list(pmap_rng_keys), jax.local_devices())


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
      encoder_init: str = 'default',
      use_lstm: bool = False,
      learning_rate: float = 0.005,
      grad_clip_max_norm: float = 0.0,
      checkpoint_path: str = '/tmp/clrs3',
      freeze_processor: bool = False,
      dropout_prob: float = 0.0,
      hint_teacher_forcing: float = 0.0,
      hint_repred_mode: str = 'soft',
      name: str = 'base_model',
      nb_msg_passing_steps: int = 1,
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
      encoder_init: The initialiser type to use for the encoders.
      use_lstm: Whether to insert an LSTM after message passing.
      learning_rate: Learning rate for training.
      grad_clip_max_norm: if greater than 0, the maximum norm of the gradients.
      checkpoint_path: Path for loading/saving checkpoints.
      freeze_processor: If True, the processor weights will be frozen and
        only encoders and decoders (and, if used, the lstm) will be trained.
      dropout_prob: Dropout rate in the message-passing stage.
      hint_teacher_forcing: Probability of using ground-truth hints instead
        of predicted hints as inputs during training (only relevant if
        `encode_hints`=True)
      hint_repred_mode: How to process predicted hints when fed back as inputs.
        Only meaningful when `encode_hints` and `decode_hints` are True.
        Options are:
          - 'soft', where we use softmaxes for categoricals, pointers
              and mask_one, and sigmoids for masks. This will allow gradients
              to flow through hints during training.
          - 'hard', where we use argmax instead of softmax, and hard
              thresholding of masks. No gradients will go through the hints
              during training; even for scalar hints, which don't have any
              kind of post-processing, gradients will be stopped.
          - 'hard_on_eval', which is soft for training and hard for evaluation.
      name: Model name.
      nb_msg_passing_steps: Number of message passing steps per hint.

    Raises:
      ValueError: if `encode_hints=True` and `decode_hints=False`.
    """
    super(BaselineModel, self).__init__(spec=spec)

    if encode_hints and not decode_hints:
      raise ValueError('`encode_hints=True`, `decode_hints=False` is invalid.')

    assert hint_repred_mode in ['soft', 'hard', 'hard_on_eval']

    self.decode_hints = decode_hints
    self.checkpoint_path = checkpoint_path
    self.name = name
    self._freeze_processor = freeze_processor
    if grad_clip_max_norm != 0.0:
      optax_chain = [optax.clip_by_global_norm(grad_clip_max_norm),
                     optax.scale_by_adam(),
                     optax.scale(-learning_rate)]
      self.opt = optax.chain(*optax_chain)
    else:
      self.opt = optax.adam(learning_rate)

    self.nb_msg_passing_steps = nb_msg_passing_steps

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
                         encoder_init, dropout_prob, hint_teacher_forcing,
                         hint_repred_mode)
    self._device_params = None
    self._device_opt_state = None
    self.opt_state_skeleton = None

  def _create_net_fns(self, hidden_dim, encode_hints, processor_factory,
                      use_lstm, encoder_init, dropout_prob,
                      hint_teacher_forcing, hint_repred_mode):
    def _use_net(*args, **kwargs):
      return nets.Net(self._spec, hidden_dim, encode_hints, self.decode_hints,
                      processor_factory, use_lstm, encoder_init,
                      dropout_prob, hint_teacher_forcing,
                      hint_repred_mode,
                      self.nb_dims, self.nb_msg_passing_steps)(*args, **kwargs)

    self.net_fn = hk.transform(_use_net)
    pmap_args = dict(axis_name='batch', devices=jax.local_devices())
    n_devices = jax.local_device_count()
    func, static_arg, extra_args = (
        (jax.jit, 'static_argnums', {}) if n_devices == 1 else
        (jax.pmap, 'static_broadcasted_argnums', pmap_args))
    pmean = functools.partial(jax.lax.pmean, axis_name='batch')
    self._maybe_pmean = pmean if n_devices > 1 else lambda x: x
    extra_args[static_arg] = 3
    self.jitted_grad = func(self._compute_grad, **extra_args)
    extra_args[static_arg] = 4
    self.jitted_feedback = func(self._feedback, donate_argnums=[0, 3],
                                **extra_args)
    extra_args[static_arg] = [3, 4, 5]
    self.jitted_predict = func(self._predict, **extra_args)
    extra_args[static_arg] = [3, 4]
    self.jitted_accum_opt_update = func(accum_opt_update, donate_argnums=[0, 2],
                                        **extra_args)

  def init(self, features: Union[_Features, List[_Features]], seed: _Seed):
    if not isinstance(features, list):
      assert len(self._spec) == 1
      features = [features]
    self.params = self.net_fn.init(jax.random.PRNGKey(seed), features, True,  # pytype: disable=wrong-arg-types  # jax-ndarray
                                   algorithm_index=-1,
                                   return_hints=False,
                                   return_all_outputs=False)
    self.opt_state = self.opt.init(self.params)
    # We will use the optimizer state skeleton for traversal when we
    # want to avoid updating the state of params of untrained algorithms.
    self.opt_state_skeleton = self.opt.init(jnp.zeros(1))

  @property
  def params(self):
    if self._device_params is None:
      return None
    return jax.device_get(_maybe_pick_first_pmapped(self._device_params))

  @params.setter
  def params(self, params):
    self._device_params = _maybe_put_replicated(params)

  @property
  def opt_state(self):
    if self._device_opt_state is None:
      return None
    return jax.device_get(_maybe_pick_first_pmapped(self._device_opt_state))

  @opt_state.setter
  def opt_state(self, opt_state):
    self._device_opt_state = _maybe_put_replicated(opt_state)

  def _compute_grad(self, params, rng_key, feedback, algorithm_index):
    lss, grads = jax.value_and_grad(self._loss)(
        params, rng_key, feedback, algorithm_index)
    return self._maybe_pmean(lss), self._maybe_pmean(grads)

  def _feedback(self, params, rng_key, feedback, opt_state, algorithm_index):
    lss, grads = jax.value_and_grad(self._loss)(
        params, rng_key, feedback, algorithm_index)
    grads = self._maybe_pmean(grads)
    params, opt_state = self._update_params(params, grads, opt_state,
                                            algorithm_index)
    lss = self._maybe_pmean(lss)
    return lss, params, opt_state

  def _predict(self, params, rng_key: hk.PRNGSequence, features: _Features,
               algorithm_index: int, return_hints: bool,
               return_all_outputs: bool):
    outs, hint_preds = self.net_fn.apply(
        params, rng_key, [features],
        repred=True, algorithm_index=algorithm_index,
        return_hints=return_hints,
        return_all_outputs=return_all_outputs)
    outs = decoders.postprocess(self._spec[algorithm_index],
                                outs,
                                sinkhorn_temperature=0.1,
                                sinkhorn_steps=50,
                                hard=True,
                                )
    return outs, hint_preds

  def compute_grad(
      self,
      rng_key: hk.PRNGSequence,
      feedback: _Feedback,
      algorithm_index: Optional[int] = None,
  ) -> Tuple[float, _Array]:
    """Compute gradients."""

    if algorithm_index is None:
      assert len(self._spec) == 1
      algorithm_index = 0
    assert algorithm_index >= 0

    # Calculate gradients.
    rng_keys = _maybe_pmap_rng_key(rng_key)  # pytype: disable=wrong-arg-types  # numpy-scalars
    feedback = _maybe_pmap_data(feedback)
    loss, grads = self.jitted_grad(
        self._device_params, rng_keys, feedback, algorithm_index)
    loss = _maybe_pick_first_pmapped(loss)
    grads = _maybe_pick_first_pmapped(grads)

    return  loss, grads

  def feedback(self, rng_key: hk.PRNGSequence, feedback: _Feedback,
               algorithm_index=None) -> float:
    if algorithm_index is None:
      assert len(self._spec) == 1
      algorithm_index = 0
    # Calculate and apply gradients.
    rng_keys = _maybe_pmap_rng_key(rng_key)  # pytype: disable=wrong-arg-types  # numpy-scalars
    feedback = _maybe_pmap_data(feedback)
    loss, self._device_params, self._device_opt_state = self.jitted_feedback(
        self._device_params, rng_keys, feedback,
        self._device_opt_state, algorithm_index)
    loss = _maybe_pick_first_pmapped(loss)
    return loss

  def predict(self, rng_key: hk.PRNGSequence, features: _Features,
              algorithm_index: Optional[int] = None,
              return_hints: bool = False,
              return_all_outputs: bool = False):
    """Model inference step."""
    if algorithm_index is None:
      assert len(self._spec) == 1
      algorithm_index = 0

    rng_keys = _maybe_pmap_rng_key(rng_key)  # pytype: disable=wrong-arg-types  # numpy-scalars
    features = _maybe_pmap_data(features)
    return _maybe_restack_from_pmap(
        self.jitted_predict(
            self._device_params, rng_keys, features,
            algorithm_index,
            return_hints,
            return_all_outputs))

  def _loss(self, params, rng_key, feedback, algorithm_index):
    """Calculates model loss f(feedback; params)."""
    output_preds, hint_preds = self.net_fn.apply(
        params, rng_key, [feedback.features],
        repred=False,
        algorithm_index=algorithm_index,
        return_hints=True,
        return_all_outputs=False)

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

    # Optionally accumulate hint losses.
    if self.decode_hints:
      for truth in feedback.features.hints:
        total_loss += losses.hint_loss(
            truth=truth,
            preds=[x[truth.name] for x in hint_preds],
            lengths=lengths,
            nb_nodes=nb_nodes,
        )

    return total_loss

  def _update_params(self, params, grads, opt_state, algorithm_index):
    updates, opt_state = filter_null_grads(
        grads, self.opt, opt_state, self.opt_state_skeleton, algorithm_index)
    if self._freeze_processor:
      params_subset = _filter_out_processor(params)
      updates_subset = _filter_out_processor(updates)
      assert len(params) > len(params_subset)
      assert params_subset
      new_params = optax.apply_updates(params_subset, updates_subset)
      new_params = hk.data_structures.merge(params, new_params)
    else:
      new_params = optax.apply_updates(params, updates)

    return new_params, opt_state

  def update_model_params_accum(self, grads) -> None:
    grads = _maybe_put_replicated(grads)
    self._device_params, self._device_opt_state = self.jitted_accum_opt_update(
        self._device_params, grads, self._device_opt_state, self.opt,
        self._freeze_processor)

  def verbose_loss(self, feedback: _Feedback, extra_info) -> Dict[str, _Array]:
    """Gets verbose loss information."""
    hint_preds = extra_info

    nb_nodes = _nb_nodes(feedback, is_chunked=False)
    lengths = feedback.features.lengths
    losses_ = {}

    # Optionally accumulate hint losses.
    if self.decode_hints:
      for truth in feedback.features.hints:
        losses_.update(
            losses.hint_loss(
                truth=truth,
                preds=[x[truth.name] for x in hint_preds],
                lengths=lengths,
                nb_nodes=nb_nodes,
                verbose=True,
            ))

    return losses_

  def restore_model(self, file_name: str, only_load_processor: bool = False):
    """Restore model from `file_name`."""
    path = os.path.join(self.checkpoint_path, file_name)
    with open(path, 'rb') as f:
      restored_state = pickle.load(f)
      if only_load_processor:
        restored_params = _filter_in_processor(restored_state['params'])
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

  mp_states: List[List[nets.MessagePassingStateChunked]]
  init_mp_states: List[List[nets.MessagePassingStateChunked]]

  def _create_net_fns(self, hidden_dim, encode_hints, processor_factory,
                      use_lstm, encoder_init, dropout_prob,
                      hint_teacher_forcing, hint_repred_mode):
    def _use_net(*args, **kwargs):
      return nets.NetChunked(
          self._spec, hidden_dim, encode_hints, self.decode_hints,
          processor_factory, use_lstm, encoder_init, dropout_prob,
          hint_teacher_forcing, hint_repred_mode,
          self.nb_dims, self.nb_msg_passing_steps)(*args, **kwargs)

    self.net_fn = hk.transform(_use_net)
    pmap_args = dict(axis_name='batch', devices=jax.local_devices())
    n_devices = jax.local_device_count()
    func, static_arg, extra_args = (
        (jax.jit, 'static_argnums', {}) if n_devices == 1 else
        (jax.pmap, 'static_broadcasted_argnums', pmap_args))
    pmean = functools.partial(jax.lax.pmean, axis_name='batch')
    self._maybe_pmean = pmean if n_devices > 1 else lambda x: x
    extra_args[static_arg] = 4
    self.jitted_grad = func(self._compute_grad, **extra_args)
    extra_args[static_arg] = 5
    self.jitted_feedback = func(self._feedback, donate_argnums=[0, 4],
                                **extra_args)
    extra_args[static_arg] = [3, 4]
    self.jitted_accum_opt_update = func(accum_opt_update, donate_argnums=[0, 2],
                                        **extra_args)

  def _init_mp_state(self, features_list: List[List[_FeaturesChunked]],
                     rng_key: _Array):
    def _empty_mp_state():
      return nets.MessagePassingStateChunked(  # pytype: disable=wrong-arg-types  # numpy-scalars
          inputs=None, hints=None, is_first=None,
          hint_preds=None, hiddens=None, lstm_state=None)
    empty_mp_states = [[_empty_mp_state() for _ in f] for f in features_list]
    dummy_params = [self.net_fn.init(rng_key, f, e, False,
                                     init_mp_state=True, algorithm_index=-1)
                    for (f, e) in zip(features_list, empty_mp_states)]
    mp_states = [
        self.net_fn.apply(d, rng_key, f, e, False,
                          init_mp_state=True, algorithm_index=-1)[1]
        for (d, f, e) in zip(dummy_params, features_list, empty_mp_states)]
    return mp_states

  def init(self,
           features: List[List[_FeaturesChunked]],
           seed: _Seed):
    self.mp_states = self._init_mp_state(features,
                                         jax.random.PRNGKey(seed))  # pytype: disable=wrong-arg-types  # jax-ndarray
    self.init_mp_states = [list(x) for x in self.mp_states]
    self.params = self.net_fn.init(
        jax.random.PRNGKey(seed), features[0], self.mp_states[0],  # pytype: disable=wrong-arg-types  # jax-ndarray
        True, init_mp_state=False, algorithm_index=-1)
    self.opt_state = self.opt.init(self.params)
    # We will use the optimizer state skeleton for traversal when we
    # want to avoid updating the state of params of untrained algorithms.
    self.opt_state_skeleton = self.opt.init(jnp.zeros(1))

  def predict(self, rng_key: hk.PRNGSequence, features: _FeaturesChunked,
              algorithm_index: Optional[int] = None):
    """Inference not implemented. Chunked model intended for training only."""
    raise NotImplementedError

  def _loss(self, params, rng_key, feedback, mp_state, algorithm_index):
    (output_preds, hint_preds), mp_state = self.net_fn.apply(
        params, rng_key, [feedback.features],
        [mp_state],
        repred=False,
        init_mp_state=False,
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

    # Optionally accumulate hint losses.
    if self.decode_hints:
      for truth in feedback.features.hints:
        loss = losses.hint_loss_chunked(
            truth=truth,
            pred=hint_preds[truth.name],
            is_first=is_first,
            nb_nodes=nb_nodes,
        )
        total_loss += loss

    return total_loss, (mp_state,)

  def _compute_grad(self, params, rng_key, feedback, mp_state, algorithm_index):
    (lss, (mp_state,)), grads = jax.value_and_grad(self._loss, has_aux=True)(
        params, rng_key, feedback, mp_state, algorithm_index)
    return self._maybe_pmean(lss), mp_state, self._maybe_pmean(grads)

  def _feedback(self, params, rng_key, feedback, mp_state, opt_state,
                algorithm_index):
    (lss, (mp_state,)), grads = jax.value_and_grad(self._loss, has_aux=True)(
        params, rng_key, feedback, mp_state, algorithm_index)
    grads = self._maybe_pmean(grads)
    params, opt_state = self._update_params(params, grads, opt_state,
                                            algorithm_index)
    lss = self._maybe_pmean(lss)
    return lss, params, opt_state, mp_state

  def compute_grad(
      self,
      rng_key: hk.PRNGSequence,
      feedback: _Feedback,
      algorithm_index: Optional[Tuple[int, int]] = None,
  ) -> Tuple[float, _Array]:
    """Compute gradients."""

    if algorithm_index is None:
      assert len(self._spec) == 1
      algorithm_index = (0, 0)
    length_index, algorithm_index = algorithm_index
    # Reusing init_mp_state improves performance.
    # The next, commented out line, should be used for proper state keeping.
    # mp_state = self.mp_states[length_index][algorithm_index]
    mp_state = self.init_mp_states[length_index][algorithm_index]
    rng_keys = _maybe_pmap_rng_key(rng_key)  # pytype: disable=wrong-arg-types  # numpy-scalars
    feedback = _maybe_pmap_reshape(feedback, split_axis=1)
    mp_state = _maybe_pmap_reshape(mp_state, split_axis=0)

    loss, mp_state, grads = self.jitted_grad(
        self._device_params, rng_keys, feedback, mp_state, algorithm_index)
    loss = _maybe_pick_first_pmapped(loss)
    grads = _maybe_pick_first_pmapped(grads)
    mp_state = _maybe_restack_from_pmap(mp_state)
    self.mp_states[length_index][algorithm_index] = mp_state
    return loss, grads

  def feedback(self, rng_key: hk.PRNGSequence, feedback: _Feedback,
               algorithm_index=None) -> float:
    if algorithm_index is None:
      assert len(self._spec) == 1
      algorithm_index = (0, 0)
    length_index, algorithm_index = algorithm_index
    # Reusing init_mp_state improves performance.
    # The next, commented out line, should be used for proper state keeping.
    # mp_state = self.mp_states[length_index][algorithm_index]
    mp_state = self.init_mp_states[length_index][algorithm_index]
    rng_keys = _maybe_pmap_rng_key(rng_key)  # pytype: disable=wrong-arg-types  # numpy-scalars
    feedback = _maybe_pmap_reshape(feedback, split_axis=1)
    mp_state = _maybe_pmap_reshape(mp_state, split_axis=0)
    loss, self._device_params, self._device_opt_state, mp_state = (
        self.jitted_feedback(
            self._device_params, rng_keys, feedback,
            mp_state, self._device_opt_state, algorithm_index))
    loss = _maybe_pick_first_pmapped(loss)
    mp_state = _maybe_restack_from_pmap(mp_state)
    self.mp_states[length_index][algorithm_index] = mp_state
    return loss

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


def _param_in_processor(module_name):
  return processors.PROCESSOR_TAG in module_name


def _filter_out_processor(params: hk.Params) -> hk.Params:
  return hk.data_structures.filter(
      lambda module_name, n, v: not _param_in_processor(module_name), params)


def _filter_in_processor(params: hk.Params) -> hk.Params:
  return hk.data_structures.filter(
      lambda module_name, n, v: _param_in_processor(module_name), params)


def _is_not_done_broadcast(lengths, i, tensor):
  is_not_done = (lengths > i + 1) * 1.0
  while len(is_not_done.shape) < len(tensor.shape):
    is_not_done = jnp.expand_dims(is_not_done, -1)
  return is_not_done


def accum_opt_update(params, grads, opt_state, opt, freeze_processor):
  """Update params from gradients collected from several algorithms."""
  # Average the gradients over all algos
  grads = jax.tree_util.tree_map(
      lambda *x: sum(x) / (sum([jnp.any(k) for k in x]) + 1e-12), *grads)
  updates, opt_state = opt.update(grads, opt_state)
  if freeze_processor:
    params_subset = _filter_out_processor(params)
    assert len(params) > len(params_subset)
    assert params_subset
    updates_subset = _filter_out_processor(updates)
    new_params = optax.apply_updates(params_subset, updates_subset)
    new_params = hk.data_structures.merge(params, new_params)
  else:
    new_params = optax.apply_updates(params, updates)

  return new_params, opt_state


@functools.partial(jax.jit, static_argnames=['opt'])
def opt_update(opt, flat_grads, flat_opt_state):
  return opt.update(flat_grads, flat_opt_state)


def filter_null_grads(grads, opt, opt_state, opt_state_skeleton, algo_idx):
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
    opt: Optax optimizer.
    opt_state: Optimizer state.
    opt_state_skeleton: A "skeleton" of optimizer state that has been
      initialized with scalar parameters. This serves to traverse each parameter
      of the otpimizer state during the opt state update.
    algo_idx: Index of algorithm, to filter out unused encoders/decoders.
      If None, no filtering happens.
  Returns:
    Updates and new optimizer state, where the parameters with null gradient
      have not been taken into account.
  """
  def _keep_in_algo(k, v):
    """Ignore params of encoders/decoders irrelevant for this algo."""
    # Note: in shared pointer decoder modes, we should exclude shared params
    #       for algos that do not have pointer outputs.
    if ((processors.PROCESSOR_TAG in k) or
        (f'algo_{algo_idx}_' in k)):
      return v
    return jax.tree_util.tree_map(lambda x: None, v)

  if algo_idx is None:
    masked_grads = grads
  else:
    masked_grads = {k: _keep_in_algo(k, v) for k, v in grads.items()}
  flat_grads, treedef = jax.tree_util.tree_flatten(
      masked_grads, is_leaf=lambda x: x is None
  )
  flat_opt_state = jax.tree_util.tree_map(
      lambda _, x: x  # pylint:disable=g-long-lambda
      if isinstance(x, (np.ndarray, jax.Array))
      else treedef.flatten_up_to(x),
      opt_state_skeleton,
      opt_state,
  )

  # Compute updates only for the params with gradient.
  flat_updates, flat_opt_state = opt_update(opt, flat_grads, flat_opt_state)

  def unflatten(flat, original):
    """Restore tree structure, filling missing (None) leaves with original."""
    if isinstance(flat, (np.ndarray, jax.Array)):
      return flat
    return jax.tree_util.tree_map(lambda x, y: x if y is None else y, original,
                                  treedef.unflatten(flat))

  # Restore the state and updates tree structure.
  new_opt_state = jax.tree_util.tree_map(lambda _, x, y: unflatten(x, y),
                                         opt_state_skeleton, flat_opt_state,
                                         opt_state)
  updates = unflatten(flat_updates,
                      jax.tree_util.tree_map(lambda x: 0., grads))
  return updates, new_opt_state
