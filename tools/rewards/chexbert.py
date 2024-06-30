import pandas as pd
import torch

from tools.metrics.chexbert import PATHOLOGIES, CheXbert


class CheXbertReward:

    def __init__(self, device, ckpt_dir, bert_path, checkpoint_path):
        self.device = device
        self.chexbert = CheXbert(
            ckpt_dir=ckpt_dir,
            bert_path=bert_path,
            checkpoint_path=checkpoint_path,
            device=self.device,
        ).to(self.device)

    def __call__(self, predictions, labels):
        return self.reward(predictions, labels)

    def reward(self, predictions, labels):

        with torch.no_grad():
            y_hat_chexbert = self.chexbert(list(predictions)).tolist()
            y_chexbert = self.chexbert(list([i[0] for i in labels])).tolist()
        
        y_hat_rows = [{k: v for k, v in zip(PATHOLOGIES, i)} for i in y_hat_chexbert]
        y_rows = [{k: v for k, v in zip(PATHOLOGIES, i)} for i in y_chexbert]

        scores = {'y_hat': pd.DataFrame(y_hat_rows), 'y': pd.DataFrame(y_rows)}
        
        # Positive is 1/positive, negative is 0/not mentioned, 2/negative, and 3/uncertain:
        scores['y_hat'][PATHOLOGIES] = (scores['y_hat'][PATHOLOGIES] == 1)
        scores['y'][PATHOLOGIES] = (scores['y'][PATHOLOGIES] == 1)

        # Create dataframes for each error type:
        # for i in ['tp', 'tn', 'fp', 'fn']:
        #     scores[i] = scores['y'][['study_id']].copy()

        # Calculate errors:
        scores['tp'] = (scores['y_hat'][PATHOLOGIES]).astype(float) * (scores['y'][PATHOLOGIES]).astype(float)
        scores['fp'] = (scores['y_hat'][PATHOLOGIES]).astype(float) * (~scores['y'][PATHOLOGIES]).astype(float)
        scores['fn'] = (~scores['y_hat'][PATHOLOGIES]).astype(float) * (scores['y'][PATHOLOGIES]).astype(float)

        # Initialise example scores dataframe:
        # scores['example'] = scores['tp'][['study_id']].copy()
        
        # Initialise example scores:
        scores['example'] = {}

        # Errors per study_id:
        for i in ['tp', 'fp', 'fn']:
            scores['example'][f'{i}'] = scores[i][PATHOLOGIES].sum(1)

        # Scores per example:
        scores['example']['precision'] = (
            scores['example']['tp'] / (scores['example']['tp'] + scores['example']['fp'])
        ).fillna(0)
        scores['example']['recall'] = (
            scores['example']['tp'] / (scores['example']['tp'] + scores['example']['fn'])
        ).fillna(0)
        scores['example']['f1'] = (
            (2 * scores['example']['precision'] * scores['example']['recall']) / (
                scores['example']['precision'] + scores['example']['recall']
            )
        ).fillna(0)

        return torch.tensor(scores['example']['f1'].tolist(), device=self.device)
    