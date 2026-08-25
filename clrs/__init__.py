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
from clrs._src import clrs_text
from clrs._src import decoders
from clrs._src import processors
from clrs._src import specs

from clrs._src.dataset import chunkify
from clrs._src.dataset import CLRSDataset
from clrs._src.dataset import create_chunked_dataset
from clrs._src.dataset import create_dataset
from clrs._src.dataset import get_clrs_folder
from clrs._src.dataset import get_dataset_gcp_url

from clrs._src.evaluation import evaluate
from clrs._src.evaluation import evaluate_hints

from clrs._src.model import Model

from clrs._src.probing import DataPoint
from clrs._src.probing import predecessor_to_cyclic_predecessor_and_first

from clrs._src.processors import get_processor_factory

from clrs._src.samplers import build_sampler
from clrs._src.samplers import CLRS30
from clrs._src.samplers import Features
from clrs._src.samplers import Feedback
from clrs._src.samplers import process_permutations
from clrs._src.samplers import process_pred_as_input
from clrs._src.samplers import process_random_pos
from clrs._src.samplers import Sampler
from clrs._src.samplers import Trajectory

from clrs._src.specs import ALGO_IDX_INPUT_NAME
from clrs._src.specs import CLRS_30_ALGS_SETTINGS
from clrs._src.specs import Location
from clrs._src.specs import OutputClass
from clrs._src.specs import Spec
from clrs._src.specs import SPECS
from clrs._src.specs import Stage
from clrs._src.specs import Type

__version__ = "2.0.3"

__all__ = (
    "ALGO_IDX_INPUT_NAME",
    "build_sampler",
    "chunkify",
    "CLRS30",
    "CLRS_30_ALGS_SETTINGS",
    "create_chunked_dataset",
    "create_dataset",
    "clrs_text",
    "get_clrs_folder",
    "get_dataset_gcp_url",
    "get_processor_factory",
    "DataPoint",
    "predecessor_to_cyclic_predecessor_and_first",
    "process_permutations",
    "process_pred_as_input",
    "process_random_pos",
    "specs",
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
