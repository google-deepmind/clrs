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

"""The CLRS Algorithmic Reasoning Benchmark."""

from clrs import models
from clrs._src import algorithms
from clrs._src import decoders
from clrs._src import processors
from clrs._src.dataset import CLRSDataset
from clrs._src.dataset import create_chunked_dataset
from clrs._src.dataset import create_dataset
from clrs._src.dataset import get_clrs_folder
from clrs._src.dataset import get_dataset_gcp_url
from clrs._src.model import evaluate
from clrs._src.model import evaluate_hints
from clrs._src.model import Model
from clrs._src.probing import DataPoint
from clrs._src.processors import get_processor_factory
from clrs._src.samplers import build_sampler
from clrs._src.samplers import CLRS30
from clrs._src.samplers import Features
from clrs._src.samplers import Feedback
from clrs._src.samplers import Sampler
from clrs._src.samplers import Trajectory
from clrs._src.specs import Location
from clrs._src.specs import OutputClass
from clrs._src.specs import Spec
from clrs._src.specs import SPECS
from clrs._src.specs import Stage
from clrs._src.specs import Type

__version__ = "1.0.0"

__all__ = (
    "build_sampler",
    "CLRS30",
    "create_chunked_dataset",
    "create_dataset",
    "get_clrs_folder",
    "get_dataset_gcp_url",
    "get_processor_factory",
    "DataPoint",
    "evaluate",
    "evaluate_hints",
    "Features",
    "Feedback",
    "Location",
    "Model",
    "Sampler",
    "Spec",
    "SPECS",
    "Stage",
    "Trajectory",
    "Type",
)
