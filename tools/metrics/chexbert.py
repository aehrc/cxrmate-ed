import os
import time
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer

from tools.metrics.mimic_cxr import MIMICCXRReportGenerationMetric

"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""

CLASSES = {
    0: 'not mentioned',
    1: 'positive',
    2: 'negative',
    3: 'uncertain',
}

PATHOLOGIES = [
    'enlarged_cardiomediastinum',
    'cardiomegaly',
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_other',
    'fracture',
    'support_devices',
    'no_finding',
]


class CheXbert(nn.Module):
    def __init__(self, ckpt_dir, bert_path, checkpoint_path, device, p=0.1):
        super(CheXbert, self).__init__()

        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(bert_path, cache_dir=ckpt_dir)
        config = BertConfig().from_pretrained(bert_path, cache_dir=ckpt_dir)

        with torch.no_grad():

            self.bert = BertModel(config)
            self.dropout = nn.Dropout(p)

            hidden_size = self.bert.pooler.dense.in_features

            # Classes: present, absent, unknown, blank for 12 conditions + support devices:
            self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])

            # Classes: yes, no for the 'no finding' observation:
            self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

            # Load CheXbert checkpoint:
            ckpt_path = os.path.join(ckpt_dir, checkpoint_path)
            assert os.path.exists(ckpt_path)
            state_dict = torch.load(ckpt_path, map_location=device)['model_state_dict']

            new_state_dict = OrderedDict()
            # new_state_dict['bert.embeddings.position_ids'] = torch.arange(config.max_position_embeddings).expand((1, -1))
            for key, value in state_dict.items():
                if 'bert' in key:
                    new_key = key.replace('module.bert.', 'bert.')
                elif 'linear_heads' in key:
                    new_key = key.replace('module.linear_heads.', 'linear_heads.')
                new_state_dict[new_key] = value

            self.load_state_dict(new_state_dict)

        self.eval()

    def forward(self, reports):

        for i in range(len(reports)):
            reports[i] = reports[i].strip()
            reports[i] = reports[i].replace("\n", " ")
            reports[i] = reports[i].replace("\s+", " ")
            reports[i] = reports[i].replace("\s+(?=[\.,])", "")
            reports[i] = reports[i].strip()

        with torch.no_grad():

            tokenized = self.tokenizer(
                reports,
                padding='longest',
                return_tensors='pt',
                truncation=True,
                max_length=self.bert.config.max_position_embeddings,
            )

            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            last_hidden_state = self.bert(**tokenized)[0]

            cls = last_hidden_state[:, 0, :]
            cls = self.dropout(cls)

            predictions = []
            for i in range(14):
                predictions.append(self.linear_heads[i](cls).argmax(dim=1))

        return torch.stack(predictions, dim=1)


class CheXbertClassificationMetrics(MIMICCXRReportGenerationMetric):
    """
    CheXbert classification metrics for MIMIC-CXR. If multiple reports are generated per study_id, each error type is
    summed over the dicom_ids.
    """

    def __init__(self, ckpt_dir, bert_path, checkpoint_path, **kwargs):
        """
        Argument/s:
            ckpt_dir - path to the checkpoint directory.
            bert_path - path to the Hugging Face BERT checkpoint (for the BERT configuration).
            checkpoint_path - path to the CheXbert checkpoint.
        """
        super().__init__(metric_name='chexbert', **kwargs)

        self.ckpt_dir = ckpt_dir
        self.bert_path = bert_path
        self.checkpoint_path = checkpoint_path

        self.add_state('reports', default=[])

    def init_metric(self):
        self.chexbert = CheXbert(
            ckpt_dir=self.ckpt_dir,
            bert_path=self.bert_path,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        ).to(self.device)

    def cleanup_metric(self):
        del self.chexbert

    def metric_scoring(self, batch):
        
        y_hat = [i['synthetic'] for i in batch]
        y = [i['radiologist'] for i in batch]
        study_ids = [i['study_id'] for i in batch]
        if self.accumulate_over_dicoms:
            dicom_ids = [i['dicom_id'] for i in batch]

        y_hat_chexbert = self.chexbert(list(y_hat)).tolist()
        y_chexbert = self.chexbert(list(y)).tolist()

        mbatch_rows = []
        if self.accumulate_over_dicoms:
            for i_1, i_2, i_3, i_4 in zip(dicom_ids, study_ids, y_hat_chexbert, y_chexbert):
                mbatch_rows.append(
                    {
                        **{'dicom_id': i_1, 'study_id': i_2}, 
                        **{f'y_hat_{k}': v for k, v in zip(PATHOLOGIES, i_3)},
                        **{f'y_label_{k}': v for k, v in zip(PATHOLOGIES, i_4)},
                    }
                )
        else:
            for i_1, i_2, i_3 in zip(study_ids, y_hat_chexbert, y_chexbert):
                mbatch_rows.append(
                    {
                        **{'study_id': i_1}, 
                        **{f'y_hat_{k}': v for k, v in zip(PATHOLOGIES, i_2)},
                        **{f'y_label_{k}': v for k, v in zip(PATHOLOGIES, i_3)},
                    }
                )

        return mbatch_rows

    def accumulate_scores(self, examples, epoch):

        y_hat_rows = [{k.replace('y_hat_', ''): v for k, v in i.items() if 'y_label_' not in k} for i in examples]
        y_rows = [{k.replace('y_label_', ''): v for k, v in i.items() if 'y_hat_' not in k} for i in examples]

        scores = {'y_hat': pd.DataFrame(y_hat_rows), 'y': pd.DataFrame(y_rows)}

        # Drop duplicates caused by DDP:
        key = 'dicom_id' if self.accumulate_over_dicoms else 'study_id'
        scores['y_hat'] = scores['y_hat'].drop_duplicates(subset=[key])
        scores['y'] = scores['y'].drop_duplicates(subset=[key])

        def save_chexbert_outputs():
            scores['y_hat'].to_csv(
                os.path.join(
                    self.save_dir, f'{self.split}_epoch-{epoch}_y_hat_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'
                ),
                index=False,
            )
            scores['y'].to_csv(
                os.path.join(
                    self.save_dir, f'{self.split}_epoch-{epoch}_y_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'
                ),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save_chexbert_outputs()
        elif torch.distributed.get_rank() == 0:
            save_chexbert_outputs()

        # Positive is 1/positive, negative is 0/not mentioned, 2/negative, and 3/uncertain:
        scores['y_hat'][PATHOLOGIES] = (scores['y_hat'][PATHOLOGIES] == 1)
        scores['y'][PATHOLOGIES] = (scores['y'][PATHOLOGIES] == 1)

        # Create dataframes for each error type:
        for i in ['tp', 'tn', 'fp', 'fn']:
            scores[i] = scores['y'][['study_id', 'dicom_id']].copy() if self.accumulate_over_dicoms \
                else scores['y'][['study_id']].copy()

        # Calculate errors:
        scores['tp'][PATHOLOGIES] = \
            (scores['y_hat'][PATHOLOGIES]).astype(float) * (scores['y'][PATHOLOGIES]).astype(float)
        scores['tn'][PATHOLOGIES] = \
            (~scores['y_hat'][PATHOLOGIES]).astype(float) * (~scores['y'][PATHOLOGIES]).astype(float)
        scores['fp'][PATHOLOGIES] = \
            (scores['y_hat'][PATHOLOGIES]).astype(float) * (~scores['y'][PATHOLOGIES]).astype(float)
        scores['fn'][PATHOLOGIES] = \
            (~scores['y_hat'][PATHOLOGIES]).astype(float) * (scores['y'][PATHOLOGIES]).astype(float)

        # Take the mean error over the DICOMs (if the sum is taken instead, studies with more DICOMs would be given more
        # importance. We want every study to be given equal importance).
        if self.accumulate_over_dicoms:
            for i in ['tp', 'tn', 'fp', 'fn']:
                scores[i] = scores[i].drop(['dicom_id'], axis=1).groupby('study_id', as_index=False).mean()

        # Initialise example scores dataframe:
        scores['example'] = scores['tp'][['study_id']].copy()

        # Errors per study_id:
        for i in ['tp', 'tn', 'fp', 'fn']:
            scores['example'][f'{i}'] = scores[i][PATHOLOGIES].sum(1)

        # Scores per example:
        scores['example']['accuracy'] = (
            (scores['example']['tp'] + scores['example']['tn']) /
            (scores['example']['tp'] + scores['example']['tn'] + scores['example']['fp'] + scores['example']['fn'])
        ).fillna(0)
        scores['example']['precision'] = (
            scores['example']['tp'] / (scores['example']['tp'] + scores['example']['fp'])
        ).fillna(0)
        scores['example']['recall'] = (
            scores['example']['tp'] / (scores['example']['tp'] + scores['example']['fn'])
        ).fillna(0)
        scores['example']['f1'] = (
            scores['example']['tp'] / (scores['example']['tp'] + 0.5 * (
                scores['example']['fp'] + scores['example']['fn'])
            )
        ).fillna(0)
        scores['example']['f1_alternate'] = (
            (2 * scores['example']['precision'] * scores['example']['recall']) / (
                scores['example']['precision'] + scores['example']['recall']
            )
        ).fillna(0)


        # Average example scores:
        scores['averaged'] = pd.DataFrame(
            scores['example'].drop(['study_id', 'tp', 'tn', 'fp', 'fn'], axis=1).mean().rename('{}_example'.format)
        ).transpose()

        # Initialise class scores dataframe:
        scores['class'] = pd.DataFrame()

        # Sum over study_ids for class scores:
        for i in ['tp', 'tn', 'fp', 'fn']:
            scores['class'][i] = scores[i][PATHOLOGIES].sum()

        # Scores for each class:
        scores['class']['accuracy'] = (
            (scores['class']['tn'] + scores['class']['tp']) / (
                scores['class']['tp'] + scores['class']['tn'] +
                scores['class']['fp'] + scores['class']['fn']
            )
        ).fillna(0)
        scores['class']['precision'] = (
            scores['class']['tp'] / (
                scores['class']['tp'] + scores['class']['fp']
            )
        ).fillna(0)
        scores['class']['recall'] = (
            scores['class']['tp'] / (
                scores['class']['tp'] + scores['class']['fn']
            )
        ).fillna(0)
        scores['class']['f1'] = (
            scores['class']['tp'] / (
                scores['class']['tp'] + 0.5 * (scores['class']['fp'] + scores['class']['fn'])
            )
        ).fillna(0)
        scores['class']['f1_alternate'] = (
            (2 * scores['class']['precision'] * scores['class']['recall']) / (
                scores['class']['precision'] + scores['class']['recall']
            )
        ).fillna(0)

        # Macro-averaging:
        for i in ['accuracy', 'precision', 'recall', 'f1', 'f1_alternate']:
            scores['averaged'][f'{i}_macro'] = [scores['class'][i].mean()]

        # Micro-averaged over the classes:
        scores['averaged']['accuracy_micro'] = (scores['class']['tp'].sum() + scores['class']['tn'].sum()) / (
            scores['class']['tp'].sum() + scores['class']['tn'].sum() +
            scores['class']['fp'].sum() + scores['class']['fn'].sum()
        )
        scores['averaged']['precision_micro'] = scores['class']['tp'].sum() / (
            scores['class']['tp'].sum() + scores['class']['fp'].sum()
        )
        scores['averaged']['recall_micro'] = scores['class']['tp'].sum() / (
            scores['class']['tp'].sum() + scores['class']['fn'].sum()
        )
        scores['averaged']['f1_micro'] = scores['class']['tp'].sum() / (
            scores['class']['tp'].sum() + 0.5 * (scores['class']['fp'].sum() + scores['class']['fn'].sum())
        )

        # Reformat classification scores for individual pathologies:
        scores['class'].insert(loc=0, column='pathology', value=scores['class'].index)
        scores['class'] = scores['class'].drop(['tp', 'tn', 'fp', 'fn'], axis=1).melt(
            id_vars=['pathology'],
            var_name='metric',
            value_name='score',
        )
        scores['class']['metric'] = scores['class']['metric'] + '_' + scores['class']['pathology']
        scores['class'] = pd.DataFrame([scores['class']['score'].tolist()], columns=scores['class']['metric'].tolist())

        # Save the example and class scores:
        def save_scores():
            scores['class'].to_csv(
                os.path.join(
                    self.save_dir,
                    f'{self.split}_epoch-{epoch}_class_scores_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv',
                ),
                index=False,
            )
            scores['example'].to_csv(
                os.path.join(
                    self.save_dir,
                    f'{self.split}_epoch-{epoch}_example_scores_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv',
                ),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save_scores()
        elif torch.distributed.get_rank() == 0:
            save_scores()

        score_dict = {
            **scores['averaged'].to_dict(orient='records')[0],
            **scores['class'].to_dict(orient='records')[0],
            'num_study_ids': float(scores['y'].study_id.nunique()),
        }

        # Number of examples:
        if self.accumulate_over_dicoms:
            score_dict['num_dicom_ids'] = float(scores['y'].dicom_id.nunique())
            
        prefix = f'{self.split}_{self.metric_name}_'
        score_dict = {f'{prefix}{k}': v for k, v in score_dict.items()}

        return score_dict
