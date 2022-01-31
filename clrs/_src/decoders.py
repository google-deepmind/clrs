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

"""Decoder utilities."""

from clrs._src import specs
import haiku as hk


_Location = specs.Location
_Spec = specs.Spec
_Type = specs.Type


def construct_decoder(loc: str, t: str, hidden_dim: int, nb_dims: int):
  """Constructs a decoder."""
  if loc == _Location.NODE:
    # Node decoders.
    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
      decoder = (hk.Linear(1),)
    elif t == _Type.CATEGORICAL:
      decoder = (hk.Linear(nb_dims),)
    elif t == _Type.POINTER:
      decoder = (hk.Linear(hidden_dim), hk.Linear(hidden_dim))
    else:
      raise ValueError(f"Invalid Type {t}")

  elif loc == _Location.EDGE:
    # Edge decoders.
    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
      decoder = (hk.Linear(1), hk.Linear(1), hk.Linear(1))
    elif t == _Type.CATEGORICAL:
      decoder = (hk.Linear(nb_dims), hk.Linear(nb_dims), hk.Linear(nb_dims))
    elif t == _Type.POINTER:
      decoder = (hk.Linear(hidden_dim), hk.Linear(hidden_dim),
                 hk.Linear(hidden_dim), hk.Linear(hidden_dim))
    else:
      raise ValueError(f"Invalid Type {t}")

  elif loc == _Location.GRAPH:
    # Graph decoders.
    if t in [_Type.SCALAR, _Type.MASK, _Type.MASK_ONE]:
      decoder = (hk.Linear(1), hk.Linear(1))
    elif t == _Type.CATEGORICAL:
      decoder = (hk.Linear(nb_dims), hk.Linear(nb_dims))
    elif t == _Type.POINTER:
      decoder = (hk.Linear(hidden_dim), hk.Linear(hidden_dim),
                 hk.Linear(hidden_dim))
    else:
      raise ValueError(f"Invalid Type {t}")

  else:
    raise ValueError(f"Invalid Location {loc}")

  return decoder
