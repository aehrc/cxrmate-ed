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
            
            # Prior studies:
            prior_study_indices = [
                None if i is None else [self.study_id_to_index[j] for j in i[:self.history]] for i in batch['prior_study_ids']
            ]
            prior_studies = [None if i is None else [self.dataset[j] for j in i] for i in prior_study_indices]

            # Prior time deltas:
            time_deltas = [
                None if i is None else [compute_time_delta(k['latest_study_datetime'], j, self.time_delta_map, to_tensor=False) for k in i] for i, j in zip(prior_studies, batch['latest_study_datetime'])
            ]     
            
            # Prior findings and impressions:
            batch['prior_findings'] = [
                None if i is None else [j['findings'] for j in i] for i in prior_studies
            ]     
            batch['prior_impression'] = [
                None if i is None else [j['findings'] for j in i] for i in prior_studies
            ]     
            batch['prior_findings_time_delta'] = time_deltas.copy()
            batch['prior_impression_time_delta'] = time_deltas.copy()

            # Prior images:
            """
            Note:
            
            Random selection of max_train_images_per_study from the study if the number of images for a study exceeds max_train_images_per_study is performed in train_set_transform and test_set_transform.
            
            Sorting the images based on the view is done in test_set_transform.
            
            No need to do it here.
            """
            prior_images = [
                torch.cat(
                    [
                        torch.empty(0, *batch['images'].shape[-3:])
                    ] if i is None else [j['images'] for j in i]
                ) for i in prior_studies
            ]            
            prior_images = torch.nn.utils.rnn.pad_sequence(prior_images, batch_first=True, padding_value=0.0)
            batch['images'] = torch.cat([batch['images'], prior_images], dim=1)
            prior_image_time_deltas = [
                None if i is None else list(itertools.chain.from_iterable([y] * x['images'].shape[0] for x, y in zip(i, j)))
                for i, j in zip(prior_studies, time_deltas)
            ]
            max_len = max((len(item) for item in prior_image_time_deltas if item is not None), default=0)            
            prior_image_time_deltas = [i + [self.inf_time_delta_value] * (max_len - len(i)) if i else [self.inf_time_delta_value] * max_len for i in prior_image_time_deltas]
            batch['image_time_deltas'] = [i + j for i, j in zip(batch['image_time_deltas'], prior_image_time_deltas)]

        return batch

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)
    
    def __getitems__(self, keys: List):
        batch = self.__getitem__(keys)
        n_examples = len(batch[next(iter(batch))])
        return [{col: array[i] for col, array in batch.items()} for i in range(n_examples)]
    