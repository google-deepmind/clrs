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

"""Model base classes and utilities."""

import abc
from typing import Dict, List, Optional, Union

from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs


Result = Dict[str, probing.DataPoint]


class Model(abc.ABC):
  """Abstract base class for CLRS3-B models."""

  def __init__(self, spec: Union[specs.Spec, List[specs.Spec]]):
    """Set up the problem, prepare to predict on first task."""
    if not isinstance(spec, list):
      spec = [spec]
    self._spec = spec

  @abc.abstractmethod
  def predict(self, features: samplers.Features) -> Result:
    """Make predictions about the current task."""
    pass

  @abc.abstractmethod
  def feedback(self, feedback: Optional[samplers.Feedback]):
    """Advance to the next task, incorporating any available feedback."""
    pass
