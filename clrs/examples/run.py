# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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

"""Run training of one or more algorithmic tasks from CLRS."""

import functools
import os
import shutil
from typing import Any, Dict, List

import scipy.special
import torch
from absl import app
from absl import flags
from absl import logging
import clrs
import jax
import numpy as np
import requests
import tensorflow as tf
import sklearn.preprocessing

#NEW
import pandas as pd # saving results to dataframe for easy visualization
import time         # measuring model training time
import pickle       # saving model on kaggle
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from clrs import _src
from clrs._src.algorithms import check_graphs

flags.DEFINE_list('algorithms', ['dfs'], 'Which algorithms to run.')
flags.DEFINE_list('train_lengths', ['4', '7', '11', '13', '16'],
                  'Which training sizes to use. A size of -1 means '
                  'use the benchmark dataset.')
flags.DEFINE_integer('length_needle', -8,
                     'Length of needle for training and validation '
                     '(not testing) in string matching algorithms. '
                     'A negative value randomizes the length for each sample '
                     'between 1 and the opposite of the value. '
                     'A value of 0 means use always 1/4 of the length of '
                     'the haystack (the default sampler behavior).')
flags.DEFINE_integer('seed', 42, 'Random seed to set')

flags.DEFINE_boolean('random_pos', True,
                     'Randomize the pos input common to all algos.')
flags.DEFINE_boolean('enforce_permutations', True,
                     'Whether to enforce permutation-type node pointers.')
flags.DEFINE_boolean('enforce_pred_as_input', True,
                     'Whether to change pred_h hints into pred inputs.')
flags.DEFINE_integer('batch_size', 1, 'Batch size used for training.')
flags.DEFINE_boolean('chunked_training', False,
                     'Whether to use chunking for training.')
flags.DEFINE_integer('chunk_length', 16,
                     'Time chunk length used for training (if '
                     '`chunked_training` is True.')
flags.DEFINE_integer('train_steps', 1000, 'Number of training iterations.')
flags.DEFINE_integer('eval_every', 50, 'Evaluation frequency (in steps).')
flags.DEFINE_integer('test_every', 500, 'Evaluation frequency (in steps).')

flags.DEFINE_integer('hidden_size', 128,
                     'Number of hidden units of the model.')
flags.DEFINE_integer('nb_heads', 1, 'Number of heads for GAT processors')
flags.DEFINE_integer('nb_msg_passing_steps', 1,
                     'Number of message passing steps to run per hint.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate to use.')
flags.DEFINE_float('grad_clip_max_norm', 1.0,
                   'Gradient clipping by norm. 0.0 disables grad clipping')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')
flags.DEFINE_float('hint_teacher_forcing', 0.0,
                   'Probability that ground-truth teacher hints are encoded '
                   'during training instead of predicted hints. Only '
                   'pertinent in encoded_decoded modes.')
flags.DEFINE_enum('hint_mode', 'encoded_decoded',
                  ['encoded_decoded', 'decoded_only', 'none'],
                  'How should hints be used? Note, each mode defines a '
                  'separate task, with various difficulties. `encoded_decoded` '
                  'requires the model to explicitly materialise hint sequences '
                  'and therefore is hardest, but also most aligned to the '
                  'underlying algorithmic rule. Hence, `encoded_decoded` '
                  'should be treated as the default mode for our benchmark. '
                  'In `decoded_only`, hints are only used for defining '
                  'reconstruction losses. Often, this will perform well, but '
                  'note that we currently do not make any efforts to '
                  'counterbalance the various hint losses. Hence, for certain '
                  'tasks, the best performance will now be achievable with no '
                  'hint usage at all (`none`).')
flags.DEFINE_enum('hint_repred_mode', 'soft', ['soft', 'hard', 'hard_on_eval'],
                  'How to process predicted hints when fed back as inputs.'
                  'In soft mode, we use softmaxes for categoricals, pointers '
                  'and mask_one, and sigmoids for masks. '
                  'In hard mode, we use argmax instead of softmax, and hard '
                  'thresholding of masks. '
                  'In hard_on_eval mode, soft mode is '
                  'used for training and hard mode is used for evaluation.')
flags.DEFINE_boolean('use_ln', True,
                     'Whether to use layer normalisation in the processor.')
flags.DEFINE_boolean('use_lstm', False,
                     'Whether to insert an LSTM after message passing.')
flags.DEFINE_integer('nb_triplet_fts', 8,
                     'How many triplet features to compute?')

flags.DEFINE_enum('encoder_init', 'xavier_on_scalars',
                  ['default', 'xavier_on_scalars'],
                  'Initialiser to use for the encoders.')
flags.DEFINE_enum('processor_type', 'triplet_gmpnn',
                  ['deepsets', 'mpnn', 'pgn', 'pgn_mask',
                   'triplet_mpnn', 'triplet_pgn', 'triplet_pgn_mask',
                   'gat', 'gatv2', 'gat_full', 'gatv2_full',
                   'gpgn', 'gpgn_mask', 'gmpnn',
                   'triplet_gpgn', 'triplet_gpgn_mask', 'triplet_gmpnn'],
                  'Processor type to use as the network P.')

flags.DEFINE_string('checkpoint_path', '/tmp/CLRS30',
                    'Path in which checkpoints are saved.')
flags.DEFINE_string('dataset_path', '/tmp/CLRS30',
                    'Path in which dataset is stored.')
flags.DEFINE_boolean('freeze_processor', False,
                     'Whether to freeze the processor of the model.')

# NEW
flags.DEFINE_boolean(name='results_df', default=False,
                     help='Whether to save loss per step in df for plotting.')
flags.DEFINE_boolean('save_df', False,
                     'Whether to save model. !! Requires results_df=True !!')
flags.DEFINE_boolean('save_model_to_file', False,
                     'Whether to save model to .pkl or similar, intended for kaggle')


FLAGS = flags.FLAGS


PRED_AS_INPUT_ALGOS = [
    'binary_search',
    'minimum',
    'find_maximum_subarray',
    'find_maximum_subarray_kadane',
    'matrix_chain_order',
    'lcs_length',
    'optimal_bst',
    'activity_selector',
    'task_scheduling',
    'naive_string_matcher',
    'kmp_matcher',
    'jarvis_march']


def unpack(v):
  try:
    return v.item()  # DeviceArray
  except (AttributeError, ValueError):
    return v


def _iterate_sampler(sampler, batch_size):
  while True:
    yield sampler.next(batch_size)


def _maybe_download_dataset(dataset_path):
  """Download CLRS30 dataset if needed."""
  dataset_folder = os.path.join(dataset_path, clrs.get_clrs_folder())
  if os.path.isdir(dataset_folder):
    logging.info('Dataset found at %s. Skipping download.', dataset_folder)
    return dataset_folder
  logging.info('Dataset not found in %s. Downloading...', dataset_folder)

  clrs_url = clrs.get_dataset_gcp_url()
  request = requests.get(clrs_url, allow_redirects=True)
  clrs_file = os.path.join(dataset_path, os.path.basename(clrs_url))
  os.makedirs(dataset_folder)
  open(clrs_file, 'wb').write(request.content)
  shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
  os.remove(clrs_file)
  return dataset_folder


def make_sampler(length: int,
                 rng: Any,
                 algorithm: str,
                 split: str,
                 batch_size: int,
                 multiplier: int,
                 randomize_pos: bool,
                 enforce_pred_as_input: bool,
                 enforce_permutations: bool,
                 chunked: bool,
                 chunk_length: int,
                 sampler_kwargs: Dict[str, Any]):
  """Create a sampler with given options.

  Args:
    length: Size of samples (i.e., number of nodes in the graph).
      A length of -1 will mean that the benchmark
      dataset (for the given split) is used. Positive sizes will instantiate
      samplers of the corresponding size.
    rng: Numpy random state.
    algorithm: The name of the algorithm to sample from.
    split: 'train', 'val' or 'test'.
    batch_size: Samples per batch.
    multiplier: Integer multiplier for the number of samples in the dataset,
      only used for positive sizes. Negative multiplier means infinite samples.
    randomize_pos: Whether to randomize the `pos` input.
    enforce_pred_as_input: Whether to convert fixed pred_h hints to inputs.
    enforce_permutations: Whether to enforce permutation pointers.
    chunked: Whether to chunk the dataset.
    chunk_length: Unroll length of chunks, if `chunked` is True.
    sampler_kwargs: Extra args passed to the sampler.
  Returns:
    A sampler (iterator), the number of samples in the iterator (negative
    if infinite samples), and the spec.
  """
  if length < 0:  # load from file
    #print('run.py loading from dataset')
    dataset_folder = _maybe_download_dataset(FLAGS.dataset_path)
    sampler, num_samples, spec = clrs.create_dataset(folder=dataset_folder,
                                                     algorithm=algorithm,
                                                     batch_size=batch_size,
                                                     split=split)
    sampler = sampler.as_numpy_iterator()
  else:
    num_samples = clrs.CLRS30[split]['num_samples'] * multiplier
    sampler, spec = clrs.build_sampler(
        algorithm,
        seed=rng.randint(2**32),
        num_samples=num_samples,
        length=length,
        **sampler_kwargs,
        )
    sampler = _iterate_sampler(sampler, batch_size)

  if randomize_pos:
    sampler = clrs.process_random_pos(sampler, rng)
  if enforce_pred_as_input and algorithm in PRED_AS_INPUT_ALGOS:
    spec, sampler = clrs.process_pred_as_input(spec, sampler)
  spec, sampler = clrs.process_permutations(spec, sampler, enforce_permutations)
  if chunked:
    sampler = clrs.chunkify(sampler, chunk_length)
  return sampler, num_samples, spec


def make_multi_sampler(sizes, rng, **kwargs):
  """Create a sampler with cycling sample sizes."""
  ss = []
  tot_samples = 0
  for length in sizes:
    sampler, num_samples, spec = make_sampler(length, rng, **kwargs)
    ss.append(sampler)
    tot_samples += num_samples

  def cycle_samplers():
    while True:
      for s in ss:
        yield next(s)
  return cycle_samplers(), tot_samples, spec


def _concat(dps, axis):
  return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)


def collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras):
  """Collect batches of output and hint preds and evaluate them."""
  processed_samples = 0
  preds = []
  outputs = []
  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    new_rng_key, rng_key = jax.random.split(rng_key)
    cur_preds, _ = predict_fn(new_rng_key, feedback.features)
    preds.append(cur_preds)
    processed_samples += batch_size
  outputs = _concat(outputs, axis=0)
  preds = _concat(preds, axis=0)
  out = clrs.evaluate(outputs, preds)
  #breakpoint()
  if extras:
    out.update(extras)
  return {k: unpack(v) for k, v in out.items()}


def BF_collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras):
  """Collect batches of output and hint preds and evaluate them."""
  processed_samples = 0
  preds = []
  outputs = []
  As = []
  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    new_rng_key, rng_key = jax.random.split(rng_key)
    cur_preds, _ = predict_fn(new_rng_key, feedback.features)
    preds.append(cur_preds)
    processed_samples += batch_size
    As.append(feedback[0][0][2].data)
  outputs = _concat(outputs, axis=0)
  As = _concat(As, axis=0)  # concatenate batches
  #breakpoint()
  preds = _concat(preds, axis=0)
  out = clrs.evaluate(outputs, preds)
  #breakpoint()
  # TODO sample from probabilities to values. Log Results

  model_sample_random = sample_random_list(preds)
  true_sample_random = sample_random_list(outputs)

  model_random_truthmask = [check_graphs.check_valid_dfsTree(As[i], model_sample_random[i]) for i in range(len(model_sample_random))]
  correctness_model_random = sum(model_random_truthmask) / len(model_random_truthmask)

  true_random_truthmask = [check_graphs.check_valid_dfsTree(As[i], true_sample_random[i]) for i in range(len(true_sample_random))]
  correctness_true_random = sum(true_random_truthmask) / len(true_random_truthmask)

  ##### ARGMAX
  ## remember to convert from jax arrays to lists for easy subsequent methods using .tolist()
  model_sample_argmax = sample_argmax_listofdict(preds)
  true_sample_argmax = sample_argmax_listofdatapoint(outputs)

  # compute the fraction of trees sampled from model output fulfilling the necessary conditions
  model_argmax_truthmask = [check_graphs.check_valid_dfsTree(As[i], model_sample_argmax[i].tolist()) for i in range(len(model_sample_argmax))]
  correctness_model_argmax = sum(model_argmax_truthmask) / len(model_argmax_truthmask)

  # compute the fraction of trees sampled from true distributions fulfilling the necessary conditions
  true_argmax_truthmask = [check_graphs.check_valid_dfsTree(As[i], true_sample_argmax[i].tolist()) for i in range(len(true_sample_argmax))]
  correctness_true_argmax = sum(true_argmax_truthmask) / len(true_argmax_truthmask)

  
  ### LOGGING ###
  As = [i.flatten() for i in As]
  result_dict = {"As": As,
                 #
                 "Argmax_Model_Trees": model_sample_argmax,
                 "Argmax_True_Trees": true_sample_argmax,
                 #
                 "Argmax_Model_Mask": model_argmax_truthmask,
                 "Argmax_True_Mask": true_argmax_truthmask,
                 #
                 "Argmax_Model_Accuracy": correctness_model_argmax,
                 "Argmax_True_Accuracy": correctness_true_argmax,
                 #
                 ###
                 #
                 "Random_Model_Trees": model_sample_random,
                 "Random_True_Trees": true_sample_random,
                 #
                 "Random_Model_Mask": model_random_truthmask,
                 "Random_True_Mask": true_random_truthmask,
                 #
                 "Random_Model_Accuracy": correctness_model_random,
                 "Random_True_Accuracy": correctness_true_random,
                 #
                 ###
                 }
  result_df = pd.DataFrame.from_dict(result_dict)
  result_df.to_csv('bf_accuracy.csv', encoding='utf-8', index=False)
  
  if extras:
    out.update(extras)
  return {k: unpack(v) for k, v in out.items()}



def DFS_collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras):
  """Collect batch of output preds and evaluate them."""
  processed_samples = 0
  preds = []
  outputs = []
  As = []
  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    new_rng_key, rng_key = jax.random.split(rng_key)
    cur_preds, _ = predict_fn(new_rng_key, feedback.features)
    preds.append(cur_preds)
    processed_samples += batch_size
    As.append(feedback[0][0][1].data)
  outputs = _concat(outputs, axis=0)
  As = _concat(As, axis=0) # concatenate batches
  #breakpoint()

  ### We need preds and A. We want to
    # 1. Sample from preds a candidate tree
    # 2. run check_graphs on candidate tree (using A as groundtruth)
    # 3. Collect validity result into a dataframe.

##### RANDOM
  model_sample_random = sample_random_list(preds)
  true_sample_random = sample_random_list(outputs)

  model_random_truthmask = [check_graphs.check_valid_dfsTree(As[i], model_sample_random[i]) for i in range(len(model_sample_random))]
  correctness_model_random = sum(model_random_truthmask) / len(model_random_truthmask)

  true_random_truthmask = [check_graphs.check_valid_dfsTree(As[i], true_sample_random[i]) for i in range(len(true_sample_random))]
  correctness_true_random = sum(true_random_truthmask) / len(true_random_truthmask)

##### ARGMAX
  ## remember to convert from jax arrays to lists for easy subsequent methods using .tolist()
  model_sample_argmax = sample_argmax_listofdict(preds)
  true_sample_argmax = sample_argmax_listofdatapoint(outputs)

  # compute the fraction of trees sampled from model output fulfilling the necessary conditions
  model_argmax_truthmask = [check_graphs.check_valid_dfsTree(As[i],model_sample_argmax[i].tolist()) for i in range(len(model_sample_argmax))]
  correctness_model_argmax = sum(model_argmax_truthmask) / len(model_argmax_truthmask)

  # compute the fraction of trees sampled from true distributions fulfilling the necessary conditions
  true_argmax_truthmask = [check_graphs.check_valid_dfsTree(As[i], true_sample_argmax[i].tolist()) for i in range(len(true_sample_argmax))]
  correctness_true_argmax = sum(true_argmax_truthmask) / len(true_argmax_truthmask)

  ##### UPWARDS
  model_sample_upwards = sample_upwards(preds)
  true_sample_upwards = sample_upwards(outputs)
  #breakpoint()

  model_upwards_truthmask = [check_graphs.check_valid_dfsTree(As[i], model_sample_upwards[i].astype(int)) for i in range(len(model_sample_upwards))]
  correctness_model_upwards = sum(model_upwards_truthmask) / len(model_upwards_truthmask)

  true_upwards_truthmask = [check_graphs.check_valid_dfsTree(As[i], true_sample_upwards[i].astype(int)) for i in range(len(true_sample_upwards))]
  correctness_true_upwards = sum(true_upwards_truthmask) / len(true_upwards_truthmask)

  ##### ALTUPWARDS
  model_sample_altUpwards = sample_altUpwards(preds)
  true_sample_altUpwards = sample_altUpwards(outputs)

  model_altUpwards_truthmask = [check_graphs.check_valid_dfsTree(As[i], model_sample_altUpwards[i].astype(int)) for i in
                             range(len(model_sample_altUpwards))]
  correctness_model_altUpwards = sum(model_altUpwards_truthmask) / len(model_altUpwards_truthmask)

  true_altUpwards_truthmask = [check_graphs.check_valid_dfsTree(As[i], true_sample_altUpwards[i].astype(int)) for i in
                            range(len(true_sample_altUpwards))]
  correctness_true_altUpwards = sum(true_altUpwards_truthmask) / len(true_altUpwards_truthmask)

  #breakpoint()
  As = [i.flatten() for i in As]
  result_dict = {"As": As,
                 #
                 "Argmax_Model_Trees": model_sample_argmax,
                 "Argmax_True_Trees": true_sample_argmax,
                 #
                 "Argmax_Model_Mask": model_argmax_truthmask,
                 "Argmax_True_Mask": true_argmax_truthmask,
                 #
                 "Argmax_Model_Accuracy": correctness_model_argmax,
                 "Argmax_True_Accuracy": correctness_true_argmax,
                 #
                 ###
                 #
                 "Random_Model_Trees": model_sample_random,
                 "Random_True_Trees": true_sample_random,
                 #
                 "Random_Model_Mask": model_random_truthmask,
                 "Random_True_Mask": true_random_truthmask,
                 #
                 "Random_Model_Accuracy": correctness_model_random,
                 "Random_True_Accuracy": correctness_true_random,
                 #
                 ###
                 #
                 "Upwards_Model_Trees": model_sample_upwards,
                 "Upwards_True_Trees": true_sample_upwards,
                 #
                 "Upwards_Model_Mask": model_upwards_truthmask,
                 "Upwards_True_Mask": true_upwards_truthmask,
                 #
                 "Upwards_Model_Accuracy": correctness_model_upwards,
                 "Upwards_True_Accuracy": correctness_true_upwards,
                 #
                 ###
                 #
                 "altUpwards_Model_Trees": model_sample_altUpwards,
                 "altUpwards_True_Trees": true_sample_altUpwards,
                 #
                 "altUpwards_Model_Mask": model_altUpwards_truthmask,
                 "altUpwards_True_Mask": true_altUpwards_truthmask,
                 #
                 "altUpwards_Model_Accuracy": correctness_model_altUpwards,
                 "altUpwards_True_Accuracy": correctness_true_altUpwards,
                 }
  result_df = pd.DataFrame.from_dict(result_dict)
  result_df.to_csv('accuracy.csv', encoding='utf-8', index=False)

  #As[0].reshape((np.sqrt(len(lAs[0])).astype(int)), np.sqrt(len(lAs[0])).astype(int))


  preds = _concat(preds, axis=0)
  out = clrs.evaluate(outputs, preds)


  #breakpoint()
  if extras:
    out.update(extras)
  return {k: unpack(v) for k, v in out.items()}


def sample_argmax_listofdict(preds):
    trees = []
    for i in preds: # de-listify into dict, happens twice
        distlist = i["pi"].data
        for prob in distlist:
            amax = np.argmax(prob, axis=1)
            #print(amax)
            trees.append(amax)
    return trees

def sample_argmax_listofdatapoint(outputs):
    '''argmax'ing index for each row in probMatrix'''
    trees = []
    for i in outputs: #de-listify into datapoint
        distlist = i.data
        for prob in distlist:
            amax = np.argmax(prob, axis=1)
            #print(amax)
            trees.append(amax)
    return trees

def sample_random_list(outsOrPreds):
    '''Random Number for each row in probMatrix'''
    trees = []
    rng = np.random.default_rng()
    for i in outsOrPreds:
        if type(i) == type({}):
            distlist = i["pi"].data
        else:
            distlist = i.data
        for probMatrix in distlist:
            pi = []
            for row in probMatrix:
                pi.append(rng.integers(len(row)))
            trees.append(pi)
            #breakpoint()
    return trees

def leafinessSort(probMatrix):
    '''
    Args:
        probMatrix: Expects probMatrix[i][j] to indicate the probability that node j is the parent of node i

    Returns: sorted-list of vertex indices where first node had lowest probability of being parent. (column with lowest sum).
    '''
    sums = np.sum(probMatrix, axis=0)
    # sort by sum column, remember the original column number
    leafiness = np.argsort(sums)
    return leafiness

def rowWiseProb(probMatrix):
    for row_ix in range(len(probMatrix)):
        if probMatrix[row_ix].sum() != 0:
            probMatrix[row_ix] = probMatrix[row_ix].astype(np.float64)/probMatrix[row_ix].astype(np.float64).sum()
    return probMatrix

def chooseUniformly(notProbArray):
    '''Expects notProbArray'''
    val = np.random.uniform(low=0, high=sum(notProbArray))
    sums = np.cumsum(notProbArray)
    for threshold_ix in range(len(sums)):
        if val < sums[threshold_ix]:
            return threshold_ix
    return np.random.randint(len(notProbArray))

def sample_upwards(outsOrPreds):
    '''

    Args:
        outsOrPreds: A list of Datapoint or Dictionaries containing JaxArray Probability Matrices

    Returns:
        trees: A list of parent trees, one for each probability Matrix in unpacked outsOrPreds.
        Each parent tree, pi, an array of length := #nodes. pi[i] = 3 indicates 3 is the parent of node i.

    Data Structures:
        distlist: A list of probability matrices
        probMatrix: A matrix, #nodes by #nodes, where probMatrix[i][j] indicates the probability that node j is the parent of node i.
        leafiness: A sorted list. 0th element is the node least likely to be a parent: the sum its probMatrix column is the lowest
        pi: the candidate parent-tree being built
    '''
    trees = []
    # PREPROCESS TO EXTRACT probMatrix
    for i in outsOrPreds:
        if type(i) == type({}):
            distlist = i["pi"].data
        else:
            distlist = i.data
        for probMatrix in distlist: # note, probMatrix is a jax ArrayImpl
            probMatrix = np.array(probMatrix) # deepcopy to numpy so mutable
            ### COMPUTATION HERE
            #. sort by leafiness
            leafiness = np.asarray(leafinessSort(probMatrix)) # shallowcopy jax array to numpy array so no problems indexing

            # turn probmatrix into true probability dist (summing to 1)
            #altered_ProbMatrix= probMatrix
            #. grab most leafy, find its parent, continue till already-discovered (self-parent or prev. iter).
            pi = np.full(len(probMatrix), np.inf)
            #breakpoint()
            while sum(leafiness) > -len(leafiness):
               #print('leafiness, ', leafiness)
                #print("pi", pi)
                altered_ProbMatrix = rowWiseProb(probMatrix)
                leaf = leafiness[leafiness != -1][0]
                #print(leafiness[leafiness != -1])
                #rint("leaf", leaf)
                # sample the leafs parent
                parent = chooseUniformly(altered_ProbMatrix[leaf])
                #if parent == None:
                #    breakpoint()
                #breakpoint()
                pi[leaf] = parent # FIXME sometimes index error
                leafiness[leaf] = -1
                leafiness[parent] = -1
                altered_ProbMatrix[:,leaf] = 0 # set leaf's column to 0: leaf should be nobody's parent, unless there's a restart, to avoid cycles... BREAKS

                # sample up the tree until parent is the start node, a self-loop or already has a parent
                #breakpoint()
                #print('before \n', parent)
                #if parent == None:
                #    breakpoint()
                while pi[parent] == np.inf:
                    #print('after \n', parent)
                    #print(pi)
                    #print(np.inf)
                    # sample up the tree

                    #print('run.py \n', altered_ProbMatrix)
                    #breakpoint()
                    leaf = parent
                    parent = chooseUniformly(altered_ProbMatrix[leaf])
                    pi[leaf] = parent
                    # remove parent as potential
                    leafiness[leaf] = -1
                    leafiness[parent] = -1
                    altered_ProbMatrix[:, leaf] = 0 # set leaf's column to 0: leaf should be nobody's parent, unless there's a restart, to avoid cycles

            if sum(np.isin(pi, np.inf)) > 0:
                #breakpoint()
                #print(leafiness)
                #print(pi)
                #print(altered_ProbMatrix)
                raise ValueError("Leaf with no parent")
            trees.append(pi)
            #breakpoint()
        #print("done w a prob matrix!")
    return trees

def extract_probMatrices(outsOrPreds):
    ''''
    handles ugly formatting difference: outs a list of dicts of datapoints,
    preds a list of datapoints
    '''
    big_probmatrix_list = []
    for i in outsOrPreds:
        if type(i) == type({}):
            distlist = i["pi"].data
        else:
            distlist = i.data
        big_probmatrix_list.extend(distlist)
    return big_probmatrix_list

def explore_upwards(orphan_ix, parent_guesses, probMatrix):
    '''starting at node_ix, sample parents upwards until you find a node which already has a parent'''
    while parent_guesses[orphan_ix] == np.inf: # until you find node who already has parent,
        # sample parent according to row of notExactlyProbs
        parent_guess = chooseUniformly(probMatrix[orphan_ix])
        parent_guesses[orphan_ix] = parent_guess
        # try further up the tree
        orphan_ix = parent_guess
    return parent_guesses


def get_parent_tree_upwards(probMatrix):
    '''according to leafiness, explore upwards, until all nodes have parents (could be themselves)'''
    parent_guesses = np.full(len(probMatrix), np.inf)
    leafiness = leafinessSort(probMatrix)  # most leafy is least-likely to be a parent: lowest column sum of probMatrix
    for node_ix in leafiness:
        if parent_guesses[node_ix] == np.inf:
            parent_guesses = explore_upwards(node_ix, parent_guesses, probMatrix)
    if np.inf in parent_guesses:
        raise ValueError('not guessing parent for someone')
    return parent_guesses


def sample_altUpwards(outsOrPreds):
    # search up
    probMatrix_list = extract_probMatrices(outsOrPreds)
    pi_trees = []
    for probMatrix in probMatrix_list:
        # build a pi-tree, sampling up
        pi_trees.append(get_parent_tree_upwards(probMatrix))
    return pi_trees

def create_samplers(rng, train_lengths: List[int]):
  """Create all the samplers."""
  train_samplers = []
  val_samplers = []
  val_sample_counts = []
  test_samplers = []
  test_sample_counts = []
  spec_list = []

  for algo_idx, algorithm in enumerate(FLAGS.algorithms):
    # Make full dataset pipeline run on CPU (including prefetching).
    with tf.device('/cpu:0'):

      if algorithm in ['naive_string_matcher', 'kmp_matcher']:
        # Fixed haystack + needle; variability will be in needle
        # Still, for chunked training, we maintain as many samplers
        # as train lengths, since, for each length there is a separate state,
        # and we must keep the 1:1 relationship between states and samplers.
        max_length = max(train_lengths)
        if max_length > 0:  # if < 0, we are using the benchmark data
          max_length = (max_length * 5) // 4
        train_lengths = [max_length]
        if FLAGS.chunked_training:
          train_lengths = train_lengths * len(train_lengths)

      logging.info('Creating samplers for algo %s', algorithm)

      p = tuple([0.1 + 0.1 * i for i in range(9)])
      if p and algorithm in ['articulation_points', 'bridges',
                             'mst_kruskal', 'bipartite_matching']:
        # Choose a lower connection probability for the above algorithms,
        # otherwise trajectories are very long
        p = tuple(np.array(p) / 2)
      length_needle = FLAGS.length_needle
      sampler_kwargs = dict(p=p, length_needle=length_needle)
      if length_needle == 0:
        sampler_kwargs.pop('length_needle')

      common_sampler_args = dict(
          algorithm=FLAGS.algorithms[algo_idx],
          rng=rng,
          enforce_pred_as_input=FLAGS.enforce_pred_as_input,
          enforce_permutations=FLAGS.enforce_permutations,
          chunk_length=FLAGS.chunk_length,
          )

      train_args = dict(sizes=train_lengths,
                        split='train',
                        batch_size=FLAGS.batch_size,
                        multiplier=-1,
                        randomize_pos=FLAGS.random_pos,
                        chunked=FLAGS.chunked_training,
                        sampler_kwargs=sampler_kwargs,
                        **common_sampler_args)
      train_sampler, _, spec = make_multi_sampler(**train_args)

      mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
      val_args = dict(sizes=[np.amax(train_lengths)],
                      split='val',
                      batch_size=32,
                      multiplier=2 * mult,
                      randomize_pos=FLAGS.random_pos,
                      chunked=False,
                      sampler_kwargs=sampler_kwargs,
                      **common_sampler_args)
      val_sampler, val_samples, spec = make_multi_sampler(**val_args)

      test_args = dict(sizes=[5], #TODO vary, old code: sizes=[-1],
                       split='test',
                       batch_size=32,
                       multiplier=2 * mult,
                       randomize_pos=False,
                       chunked=False,
                       sampler_kwargs={},
                       **common_sampler_args)
      test_sampler, test_samples, spec = make_multi_sampler(**test_args)

    spec_list.append(spec)
    train_samplers.append(train_sampler)
    val_samplers.append(val_sampler)
    val_sample_counts.append(val_samples)
    test_samplers.append(test_sampler)
    test_sample_counts.append(test_samples)

  return (train_samplers,
          val_samplers, val_sample_counts,
          test_samplers, test_sample_counts,
          spec_list)


def main(unused_argv):
  if FLAGS.hint_mode == 'encoded_decoded':
    encode_hints = True
    decode_hints = True
  elif FLAGS.hint_mode == 'decoded_only':
    encode_hints = False
    decode_hints = True
  elif FLAGS.hint_mode == 'none':
    encode_hints = False
    decode_hints = False
  else:
    raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')

  if FLAGS.results_df:
      RESULTS = {} # for a model, save best_val_error, test_error, and train time
      PRE_DF_RESULTS = ([['Train KlDiv', 'Mean 1-abs(error)', 'Num Steps',
                                         'Examples Seen']])


  train_lengths = [int(x) for x in FLAGS.train_lengths]

  rng = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(rng.randint(2**32))

 # print('calling create samplers')
  # Create samplers
  (train_samplers,
   val_samplers, val_sample_counts,
   test_samplers, test_sample_counts,
   spec_list) = create_samplers(rng, train_lengths)
 # print('run.py made samplers')

  processor_factory = clrs.get_processor_factory(
      FLAGS.processor_type,
      use_ln=FLAGS.use_ln,
      nb_triplet_fts=FLAGS.nb_triplet_fts,
      nb_heads=FLAGS.nb_heads
  )
  model_params = dict(
      processor_factory=processor_factory,
      hidden_dim=FLAGS.hidden_size,
      encode_hints=encode_hints,
      decode_hints=decode_hints,
      encoder_init=FLAGS.encoder_init,
      use_lstm=FLAGS.use_lstm,
      learning_rate=FLAGS.learning_rate,
      grad_clip_max_norm=FLAGS.grad_clip_max_norm,
      checkpoint_path=FLAGS.checkpoint_path,
      freeze_processor=FLAGS.freeze_processor,
      dropout_prob=FLAGS.dropout_prob,
      hint_teacher_forcing=FLAGS.hint_teacher_forcing,
      hint_repred_mode=FLAGS.hint_repred_mode,
      nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
      )

  eval_model = clrs.models.BaselineModel(
      spec=spec_list,
      dummy_trajectory=[next(t) for t in val_samplers],
      **model_params
  )
  if FLAGS.chunked_training:
    train_model = clrs.models.BaselineModelChunked(
        spec=spec_list,
        dummy_trajectory=[next(t) for t in train_samplers],
        **model_params
        )
  else:
    train_model = eval_model

  #exit(0)

 # print('run.py starting training')

  train_start_time = time.time()
  # Training loop.
  best_score = -1.0
  current_train_items = [0] * len(FLAGS.algorithms)
  step = 0
  next_eval = 0
  # Make sure scores improve on first step, but not overcome best score
  # until all algos have had at least one evaluation.
  val_scores = [-99999.9] * len(FLAGS.algorithms)
  length_idx = 0
  TEMP = []

  while step < FLAGS.train_steps:
    feedback_list = [next(t) for t in train_samplers]
    # check after feedback list what we get is ground-truth probabilities
   # print('run.py, feedback_list[0]', feedback_list[0])
    #breakpoint()

    # Initialize model.
    if step == 0:
      all_features = [f.features for f in feedback_list]
      if FLAGS.chunked_training:
        # We need to initialize the model with samples of all lengths for
        # all algorithms. Also, we need to make sure that the order of these
        # sample sizes is the same as the order of the actual training sizes.
        all_length_features = [all_features] + [
            [next(t).features for t in train_samplers]
            for _ in range(len(train_lengths))]
        train_model.init(all_length_features[:-1], FLAGS.seed + 1)
      else:
        train_model.init(all_features, FLAGS.seed + 1)

   # print('run.py model initialized')
    # Training step.
    for algo_idx in range(len(train_samplers)):
      feedback = feedback_list[algo_idx]
      rng_key, new_rng_key = jax.random.split(rng_key)
      if FLAGS.chunked_training:
        # In chunked training, we must indicate which training length we are
        # using, so the model uses the correct state.
        length_and_algo_idx = (length_idx, algo_idx)
      else:
        # In non-chunked training, all training lengths can be treated equally,
        # since there is no state to maintain between batches.
        length_and_algo_idx = algo_idx
      cur_loss = train_model.feedback(rng_key, feedback, length_and_algo_idx)
      rng_key = new_rng_key

      TEMP.append(cur_loss)

      if FLAGS.chunked_training:
        examples_in_chunk = np.sum(feedback.features.is_last).item()
      else:
        examples_in_chunk = len(feedback.features.lengths)
      current_train_items[algo_idx] += examples_in_chunk
      #logging.info('Algo %s step %i current loss %f, current_train_items %i.',
      #             FLAGS.algorithms[algo_idx], step,
      #             cur_loss, current_train_items[algo_idx])

    # Periodically evaluate model
    if step >= next_eval:
      eval_model.params = train_model.params
      for algo_idx in range(len(train_samplers)):
        common_extras = {'examples_seen': current_train_items[algo_idx],
                         'step': step,
                         'algorithm': FLAGS.algorithms[algo_idx]}
        #breakpoint()


        # Validation info.
        new_rng_key, rng_key = jax.random.split(rng_key)
        val_stats = collect_and_eval(
            val_samplers[algo_idx],
            functools.partial(eval_model.predict, algorithm_index=algo_idx),
            val_sample_counts[algo_idx],
            new_rng_key,
            extras=common_extras)
        logging.info('(val) algo %s step %d: %s',
                     FLAGS.algorithms[algo_idx], step, val_stats)
        val_scores[algo_idx] = val_stats['score']

        if FLAGS.results_df:
            # train_loss:= cur_loss, val_score := sum(val_scores), test_score is not-yet calculable
            # epoch is num_train_steps?
            this_result = [np.mean(TEMP), sum(val_scores), step, current_train_items[algo_idx]]
            #{'Train KlDiv': cur_loss, 'Val MAE': sum(val_scores), 'Num Steps': step,
            #                            'Examples Seen': current_train_items[algo_idx]}])
            PRE_DF_RESULTS.append(this_result)
            TEMP = []
            #breakpoint()

      next_eval += FLAGS.eval_every

      # If best total score, update best checkpoint.
      # Also save a best checkpoint on the first step.
      msg = (f'best avg val score was '
             f'{best_score/len(FLAGS.algorithms):.3f}, '
             f'current avg val score is {np.mean(val_scores):.3f}, '
             f'val scores are: ')
      msg += ', '.join(
          ['%s: %.3f' % (x, y) for (x, y) in zip(FLAGS.algorithms, val_scores)])
      if (sum(val_scores) > best_score) or step == 0:
        best_score = sum(val_scores)
        logging.info('Checkpointing best model, %s', msg)
        train_model.save_model('best.pkl')
      else:
        logging.info('Not saving new best model, %s', msg)

    step += 1
    length_idx = (length_idx + 1) % len(train_lengths)

    train_end_time = time.time()
    train_time = train_end_time-train_start_time # timing includes occasional validation and checkpointing

  logging.info('Restoring best model from checkpoint...')
  eval_model.restore_model('best.pkl', only_load_processor=False)


 # print('run.py doing logging?')
  for algo_idx in range(len(train_samplers)):
    common_extras = {'examples_seen': current_train_items[algo_idx],
                     'step': step,
                     'algorithm': FLAGS.algorithms[algo_idx]}

    new_rng_key, rng_key = jax.random.split(rng_key)
    #breakpoint()
    if FLAGS.algorithms[algo_idx] == "dfs":
        test_stats = DFS_collect_and_eval(
            test_samplers[algo_idx],
            functools.partial(eval_model.predict, algorithm_index=algo_idx),
            test_sample_counts[algo_idx],
            new_rng_key,
            extras=common_extras)
    elif FLAGS.algorithms[algo_idx] == 'bellman_ford':
        test_stats = BF_collect_and_eval(
            test_samplers[algo_idx],
            functools.partial(eval_model.predict, algorithm_index=algo_idx),
            test_sample_counts[algo_idx],
            new_rng_key,
            extras=common_extras)
    else:
        test_stats = collect_and_eval(
        test_samplers[algo_idx],
        functools.partial(eval_model.predict, algorithm_index=algo_idx),
        test_sample_counts[algo_idx],
        new_rng_key,
        extras=common_extras)
    logging.info('(test) algo %s : %s', FLAGS.algorithms[algo_idx], test_stats)

  if FLAGS.results_df:
      RESULTS['run0'] = (train_time, best_score) # best_score given by highest val score, which is MAE by EVAL_FN
      DF_RESULTS = pd.DataFrame(PRE_DF_RESULTS)
      if FLAGS.save_df:
          DF_RESULTS.to_csv('score-results-UPDATEMYNAME.csv', encoding='utf-8', index=False)

  if FLAGS.save_model_to_file: #saving full model. Remember to call loadel_model.eval() on loaded model if you want to do inference
      ## doesnt worKtorch.save(eval_model.state_dict(), 'best_model_state_dict.pth') # saves eval_model to PATH='best_model.pth'
      eval_model.save_model_to_permanent_file('eval_model_pickle-UPDATEMYNAME.pkl')
      ## load with filepointer! look at baselines.py restore_model for example
  #breakpoint()
  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)
