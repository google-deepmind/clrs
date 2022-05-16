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

"""Tests for processors.py."""

from absl.testing import absltest
import chex
from clrs._src import processors
import haiku as hk
import jax.numpy as jnp


class MemnetTest(absltest.TestCase):

  def test_simple_run_and_check_shapes(self):

    batch_size = 64
    vocab_size = 177
    embedding_size = 64
    sentence_size = 11
    memory_size = 320
    linear_output_size = 128
    num_hops = 2
    use_ln = True

    def forward_fn(queries, stories):
      model = processors.MemNetFull(
          vocab_size=vocab_size,
          embedding_size=embedding_size,
          sentence_size=sentence_size,
          memory_size=memory_size,
          linear_output_size=linear_output_size,
          num_hops=num_hops,
          use_ln=use_ln)
      return model._apply(queries, stories)

    forward = hk.transform(forward_fn)

    queries = jnp.ones([batch_size, sentence_size], dtype=jnp.int32)
    stories = jnp.ones([batch_size, memory_size, sentence_size],
                       dtype=jnp.int32)

    key = hk.PRNGSequence(42)
    params = forward.init(next(key), queries, stories)

    model_output = forward.apply(params, None, queries, stories)
    chex.assert_shape(model_output, [batch_size, vocab_size])
    chex.assert_type(model_output, jnp.float32)


if __name__ == '__main__':
  absltest.main()
