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

"""Tests for utils.py."""

from absl.testing import absltest
import chex
from clrs._src import utils
import haiku as hk
import jax.numpy as jnp

class UtilsTest(absltest.TestCase):

  def test_sample_msgs(self):

    b = 1
    n = 2
    h = 2
    num_samples_per_node = 1

    # The messages tensor has shape [b, n, n, h]
    # msgs[0,:,0,:] = [[0, 1], [2, 3]] ; msgs[0,:,1,:] = [[4, 5], [6, 7]]  
    msgs = jnp.array([[[[0,1],[4,5]],[[2,3],[6,7]]]])
    # Fully-connected graph
    adj_mat = jnp.ones((b, n, n))
    # The random indices generated for the seed zero are [[[1, 1]]]
    expected_sampled_msgs = jnp.array([[[[2,3],[6,7]]]])
    # Sample messages
    sampled_msgs = utils.sample_msgs(msgs, adj_mat, num_samples_per_node, seed=0)

    chex.assert_equal((sampled_msgs == expected_sampled_msgs).all(), True)


if __name__ == '__main__':
  absltest.main()
