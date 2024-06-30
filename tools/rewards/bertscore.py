import os

import torch
from bert_score import BERTScorer


class BERTScoreReward:

    def __init__(self, device, num_workers):
        
        self.bert_scorer = BERTScorer(
            model_type='roberta-large',
            num_layers=17,
            nthreads=num_workers,
            all_layers=False,
            idf=False,
            lang='en',
            device=device,
            rescale_with_baseline=True,
        )
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def __call__(self, predictions, labels):
        return self.reward(predictions, labels)

    def reward(self, predictions, labels):

        with torch.no_grad() and torch.autocast(device_type='cuda', dtype=torch.float32):

            bert_scores = self.bert_scorer.score(predictions, labels, batch_size=len(predictions))
            f1 = bert_scores[2].to(device=self.bert_scorer.device)

        return f1
    