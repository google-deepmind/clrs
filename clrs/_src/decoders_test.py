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

"""Unit tests for `decoders.py`."""

from absl.testing import absltest

import chex
from clrs._src import decoders
import jax
import jax.numpy as jnp


class DecodersTest(absltest.TestCase):

  def test_log_sinkhorn(self):
    x = jax.random.normal(jax.random.PRNGKey(42), (10, 10))
    y = jnp.exp(decoders.log_sinkhorn(x, steps=10, temperature=1.0,
                                      zero_diagonal=False,
                                      noise_rng_key=None))
    chex.assert_trees_all_close(jnp.sum(y, axis=-1), 1., atol=1e-4)
    chex.assert_trees_all_close(jnp.sum(y, axis=-2), 1., atol=1e-4)

  def test_log_sinkhorn_zero_diagonal(self):
    x = jax.random.normal(jax.random.PRNGKey(42), (10, 10))
    y = jnp.exp(decoders.log_sinkhorn(x, steps=10, temperature=1.0,
                                      zero_diagonal=True,
                                      noise_rng_key=None))
    chex.assert_trees_all_close(jnp.sum(y, axis=-1), 1., atol=1e-4)
    chex.assert_trees_all_close(jnp.sum(y, axis=-2), 1., atol=1e-4)
    chex.assert_trees_all_close(jnp.sum(y.diagonal()), 0., atol=1e-4)


if __name__ == '__main__':
  absltest.main()
