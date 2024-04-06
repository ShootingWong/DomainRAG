# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.append('tasks')
# from . import Datasets
from Datasets import ConversationDataset
from Datasets import FactualDataset
from Datasets import BasicDataset
from Datasets import StructureDataset
from Datasets import MultidocDataset
from Datasets import NoisyDataset

AVAILABLE_TASKS = {
    "conversation": ConversationDataset,
    "counterfactual": FactualDataset,
    "basic": BasicDataset,
    "time": BasicDataset,
    "structure": StructureDataset,
    "multidoc": MultidocDataset,
    "noisy": NoisyDataset
    
}



def get_dataset(opt, path):
    if opt.task not in AVAILABLE_TASKS:
        raise ValueError(f"{opt.task} not recognised")
    task_dataset = AVAILABLE_TASKS[opt.task]
    return task_dataset(opt, path)
