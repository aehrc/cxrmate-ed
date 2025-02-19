import itertools
from typing import List

import torch

from .utils import compute_time_delta


class PriorsDataset:
    def __init__(self, dataset, history, time_delta_map):
        self.dataset = dataset
        self.history = history
        self.study_id_to_index = dict(zip(dataset['study_id'], range(len(dataset))))
        self.time_delta_map = time_delta_map
        self.inf_time_delta_value = time_delta_map(float('inf'))
        
    def __getitem__(self, idx):
        batch = self.dataset[idx]
        
        if self.history:
            
            raise NotImplementedError("Priors were made not available in the public release.")

        return batch

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)
    
    def __getitems__(self, keys: List):
        batch = self.__getitem__(keys)
        n_examples = len(batch[next(iter(batch))])
        return [{col: array[i] for col, array in batch.items()} for i in range(n_examples)]
    