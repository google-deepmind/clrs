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

"""Unit tests for `evaluation.py`."""

from absl.testing import absltest
from clrs._src import evaluation
from clrs._src import probing
from clrs._src import specs

import jax
import jax.numpy as jnp
import numpy as np


class EvaluationTest(absltest.TestCase):

  def test_reduce_permutations(self):
    b = 8
    n = 16
    pred = jnp.stack([jax.random.permutation(jax.random.PRNGKey(i), n)
                      for i in range(b)])
    heads = jax.random.randint(jax.random.PRNGKey(42), (b,), 0, n)

    perm = probing.DataPoint(name='test',
                             type_=specs.Type.PERMUTATION_POINTER,
                             location=specs.Location.NODE,
                             data=np.asarray(jax.nn.one_hot(pred, n)))
    mask = probing.DataPoint(name='test_mask',
                             type_=specs.Type.MASK_ONE,
                             location=specs.Location.NODE,
                             data=np.asarray(jax.nn.one_hot(heads, n)))
    output = evaluation.fuse_perm_and_mask(perm=perm, mask=mask)
    expected_output = np.array(pred)
    expected_output[np.arange(b), heads] = heads
    self.assertEqual(output.name, 'test')
    self.assertEqual(output.type_, specs.Type.POINTER)
    self.assertEqual(output.location, specs.Location.NODE)
    np.testing.assert_allclose(output.data, expected_output)


if __name__ == '__main__':
  absltest.main()
