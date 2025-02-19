import os
import re
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from bert_score import BERTScorer
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from huggingface_hub import hf_hub_download
from radgraph import F1RadGraph
from rouge_score import rouge_scorer
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torchmetrics import Metric
from torchmetrics.text import BLEUScore
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
)


class NLGMetric(Metric):
    """
    Torchmetric for Natural Language Generation (NLG) metrics.
    """
    def __init__(
        self, 
        mbatch_size: int = 1, 
        scoring_after_gather: bool = False, 
        compute_in_batches: bool = True,
    ):
        super().__init__(dist_sync_on_step=False)
        self.mbatch_size = mbatch_size
        self.scoring_after_gather = scoring_after_gather
        self.compute_in_batches = compute_in_batches 
        self.epoch = None

    @staticmethod
    def mini_batch(iterable, mbatch_size=1):
        length = len(iterable)
        for i in range(0, length, mbatch_size):
            yield iterable[i:min(i + mbatch_size, length)]

    def update(self, **kwargs):
        raise NotImplementedError
    
    def init_metric(self):
        pass

    def cleanup_metric(self):
        pass

    def metric_scoring(self, batch):
        raise NotImplementedError

    def accumulate_scores(self, rows, epoch):
        raise NotImplementedError

    def metric_init_scoring_cleanup(self, rows: Optional[list] = None):
        self.init_metric()
        if self.compute_in_batches:
            input_rows, rows = rows, []
            for i in self.mini_batch(input_rows, self.mbatch_size):
                with torch.no_grad():
                    scores = self.metric_scoring(i)
                    rows.extend(scores)
        else:
            with torch.no_grad():
                rows = self.metric_scoring(rows)
        self.cleanup_metric()
        return rows

    def convert_lists_to_rows(self):
        raise NotImplementedError

    def compute(self, epoch: Optional[int] = None):

        self.epoch = epoch
        rows = self.convert_lists_to_rows()

        if not self.scoring_after_gather:

            rows = self.metric_init_scoring_cleanup(rows)

            if torch.distributed.is_initialized():
                rows_gathered = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(rows_gathered, rows)
                rows = [j for i in rows_gathered for j in i]

        if self.scoring_after_gather:
            
            if torch.distributed.is_initialized():
                rows_gathered = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(rows_gathered, rows)
                rows = [j for i in rows_gathered for j in i]

            rows = self.metric_init_scoring_cleanup(rows)

        return self.accumulate_scores(rows, epoch)


class CXRReportGenerationMetric(NLGMetric):
    """
    Torchmetric for metrics for CXR report generation evaluation.
    """

    def __init__(self, metric_name: str, split: str, exp_dir: str, accumulate_over_dicoms: bool, **kwargs):
        """
        Argument/s:
            metric_name - name of the metric.
            split - dataset split.
            exp_dir - experiment directory where outputs will be saved.
            accumulate_over_dicoms - whether to accumulate scores over the report for each DICOM for a study.
        """
        super().__init__(**kwargs)

        self.metric_name = metric_name
        self.split = split
        self.exp_dir = exp_dir
        self.accumulate_over_dicoms = accumulate_over_dicoms

        self.add_state('synthetic', default=[])
        self.add_state('radiologist', default=[])
        self.add_state('study_ids', default=[])
        self.add_state('dicom_ids', default=[])

        self.save_dir = os.path.join(self.exp_dir, 'metric_outputs', self.metric_name)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def update(self, synthetic, radiologist, study_ids, dicom_ids=None):
        """
        Argument/s:
            synthetic - the synthetic reports must be in the following format:

                [
                    '...',
                    '...',
                ]
            radiologist - the radiologist reports must be in the following format:

                [
                    '...',
                    '...',
                ]
            study_ids - list of study identifiers.
            dicom_ids - list of dicom identifiers.
        """

        assert isinstance(synthetic, list), '"synthetic" must be a list of strings.'
        assert all(isinstance(i, str) for i in synthetic), 'Each element of "synthetic" must be a string.'
        assert isinstance(radiologist, list), '"labels" must be a list of lists, where each sub-list has a multiple strings.'
        assert all(isinstance(i, str) for i in radiologist), 'Each element of "radiologist" must be a list of strings.'

        if self.accumulate_over_dicoms:
            self.synthetic.extend(synthetic)
            self.radiologist.extend(radiologist)
            self.study_ids.extend(study_ids)
            self.dicom_ids.extend(dicom_ids)
        else:
            self.synthetic.extend(synthetic)
            self.radiologist.extend(radiologist)
            self.study_ids.extend(study_ids)

    def convert_lists_to_rows(self):
        rows = []
        if self.accumulate_over_dicoms:
            for (i_1, i_2, i_3, i_4) in zip(self.synthetic, self.radiologist, self.study_ids, self.dicom_ids):
                rows.append(
                    {
                        'synthetic': i_1,
                        'radiologist': i_2,
                        'study_id': i_3,
                        'dicom_id': i_4,
                    }
                )

        else:
            for (i_1, i_2, i_3) in zip(self.synthetic, self.radiologist, self.study_ids):
                rows.append(
                    {
                        'synthetic': i_1,
                        'radiologist': i_2,
                        'study_id': i_3,
                    }
                )

        return rows

    def accumulate_scores(self, rows, epoch):

        df = pd.DataFrame(rows)

        # Drop duplicates caused by DDP:
        key = 'dicom_id' if self.accumulate_over_dicoms else 'study_id'
        df = df.drop_duplicates(subset=[key])
        df = df.drop(columns=['synthetic', 'radiologist'], axis=1, errors='ignore')

        # Save the scores:
        def save_scores():
            df.to_csv(
                os.path.join(
                    self.save_dir,
                    f'{self.split}_epoch-{epoch}_scores_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv',
                ),
                index=False,
            )
        if not torch.distributed.is_initialized():
            save_scores()
        elif torch.distributed.get_rank() == 0:
            save_scores()

        # Number of examples:
        prefix = f'{self.split}_{self.metric_name}_'
        scores = {f'{prefix}num_study_ids': float(df.study_id.nunique())}
        if self.accumulate_over_dicoms:
            scores[f'{prefix}num_dicom_ids'] = float(df.dicom_id.nunique())

        # Take the mean error over the DICOMs (if the sum is taken instead, studies with more DICOMs would be given more
        # importance. We want every study to be given equal importance).
        if self.accumulate_over_dicoms:
            df = df.drop(['dicom_id'], axis=1).groupby('study_id', as_index=False).mean()

        df = df.drop(['study_id'], axis=1)
        mean_scores = {f'{prefix}{k}': v for k, v in df.mean().to_dict().items()}
        scores = {**mean_scores, **scores}
        scores.pop('study_id', None)

        return scores


class ReportLogger(CXRReportGenerationMetric):
    """
    Logs the findings and impression sections of a report to a .csv.
    """

    def __init__(self, track_dicom_id: bool, **kwargs):
        """
        track_dicom_id - track the DICOM identifier if generating a report per DICOM.
        """
        super().__init__(metric_name='reports', accumulate_over_dicoms=track_dicom_id, **kwargs)
        self.track_dicom_id = track_dicom_id

        self.add_state('findings', default=[])
        self.add_state('impression', default=[])
        self.add_state('study_ids', default=[])
        self.add_state('dicom_ids', default=[])

    def update(self, findings, impression, study_ids, dicom_ids=None):
        """
        Argument/s:
            findings - the findings section must be in the following format:

                [
                    '...',
                    '...',
                ]
            impression - the impression section must be in the following format:

                [
                    '...',
                    '...',
                ]
            study_ids - list of study identifiers.
            dicom_ids - list of dicom identifiers.
        """

        assert isinstance(findings, list), '"findings" must be a list of strings.'
        assert all(isinstance(i, str) for i in findings), 'Each element of "findings" must be a string.'
        assert isinstance(impression, list), '"impression" must be a list of strings.'
        assert all(isinstance(i, str) for i in impression), 'Each element of "impression" must be a string.'

        if self.track_dicom_id:
            self.findings.extend(findings)
            self.impression.extend(impression)
            self.study_ids.extend(study_ids)
            self.dicom_ids.extend(dicom_ids)
        else:
            self.findings.extend(findings)
            self.impression.extend(impression)
            self.study_ids.extend(study_ids)
    
    def compute(self, epoch):

        rows = []
        if self.track_dicom_id:
            for (i_1, i_2, i_3, i_4) in zip(self.findings, self.impression, self.study_ids, self.dicom_ids):
                rows.append(
                    {
                        'findings': i_1,
                        'impression': i_2,
                        'study_id': i_3,
                        'dicom_id': i_4,
                    }
                )

        else:
            for (i_1, i_2, i_3) in zip(self.findings, self.impression, self.study_ids):
                rows.append(
                    {
                        'findings': i_1,
                        'impression': i_2,
                        'study_id': i_3,
                    }
                )

        if torch.distributed.is_initialized():  # If DDP
            rows_gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(rows_gathered, rows)
            rows = [j for i in rows_gathered for j in i]

        return self.log(epoch, rows)

    def log(self, epoch, rows):

        def save():

            key = 'dicom_id' if self.track_dicom_id else 'study_id'
            df = pd.DataFrame(rows).drop_duplicates(subset=key)

            df.to_csv(
                os.path.join(self.save_dir, f'{self.split}_epoch-{epoch}_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save()
        elif torch.distributed.get_rank() == 0:
            save()


class ReportTokenIdentifiersLogger(CXRReportGenerationMetric):
    """
    Logs the findings and impression section token identifiers of a report to a .csv.
    """

    def __init__(self, track_dicom_id: bool, **kwargs):
        """
        track_dicom_id - track the DICOM identifier if generating a report per DICOM.
        """
        super().__init__(metric_name='report_ids', accumulate_over_dicoms=track_dicom_id, **kwargs)
        self.track_dicom_id = track_dicom_id

        self.add_state('report_ids', default=[])
        self.add_state('study_ids', default=[])
        self.add_state('dicom_ids', default=[])

    def update(self, report_ids, study_ids, dicom_ids=None):
        """
        Argument/s:
            report_ids - report identifiers.
            study_ids - list of study identifiers.
            dicom_ids - list of dicom identifiers.
        """

        assert isinstance(report_ids, torch.Tensor), '"report_ids" must be a torch.Tensor.'

        if self.track_dicom_id:
            self.report_ids.extend(report_ids)
            self.study_ids.extend(study_ids)
            self.dicom_ids.extend(dicom_ids)
        else:
            self.report_ids.extend(report_ids)
            self.study_ids.extend(study_ids)

    def compute(self, epoch):
        report_ids = self.report_ids.tolist() if not isinstance(self.report_ids, list) else self.report_ids

        rows = []
        if self.track_dicom_id:
            for (i, j, k) in zip(report_ids, self.study_ids, self.dicom_ids):
                rows.append(
                    {
                        'report_ids': i,
                        'study_id': j,
                        'dicom_id': k,
                    }
                )
        else:
            for (i, j) in zip(report_ids, self.study_ids):
                rows.append(
                    {
                        'report_ids': i,
                        'study_id': j,
                    }
                )

        if torch.distributed.is_initialized():  # If DDP
            rows_gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(rows_gathered, rows)
            rows = [j for i in rows_gathered for j in i]
        return self.log(epoch, rows)

    def log(self, epoch, rows):

        def save():

            key = 'dicom_id' if self.track_dicom_id else 'study_id'
            df = pd.DataFrame(rows).drop_duplicates(subset=key)

            df.to_csv(
                os.path.join(self.save_dir, f'{self.split}_epoch-{epoch}_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save()
        elif torch.distributed.get_rank() == 0:
            save()


class SizeLogger(CXRReportGenerationMetric):

    def __init__(self, track_dicom_id: bool, **kwargs):
        """
        track_dicom_id - track the DICOM identifier if generating a report per DICOM.
        """
        super().__init__(metric_name='size', accumulate_over_dicoms=track_dicom_id, scoring_after_gather=True, compute_in_batches=False, **kwargs)
        self.track_dicom_id = track_dicom_id

        self.add_state('size', default=[])
        self.add_state('study_ids', default=[])
        self.add_state('dicom_ids', default=[])

    def update(self, sizes, study_ids, dicom_ids=None):
        """
        Argument/s:
            sizes - sizes.
            study_ids - list of study identifiers.
            dicom_ids - list of dicom identifiers.
        """
        assert isinstance(sizes, list), '"sizes" must be a list.'

        if self.track_dicom_id:
            self.size.extend(sizes)
            self.study_ids.extend(study_ids)
            self.dicom_ids.extend(dicom_ids)
        else:
            self.size.extend(sizes)
            self.study_ids.extend(study_ids)

    def metric_scoring(self, batch):

        mbatch_rows = []
        if self.accumulate_over_dicoms:
            for x, y, z in zip(self.dicom_ids, self.study_ids, self.size):
                mbatch_rows.append({'dicom_id': x, 'study_id': y, 'size': z})
        else:
            for x, y in zip(self.study_ids, self.size):
                mbatch_rows.append({'study_id': x, 'size': y})

        return mbatch_rows

    def convert_lists_to_rows(self):
        size = self.size.tolist() if not isinstance(self.size, list) else self.size

        rows = []
        if self.track_dicom_id:
            for (i, j, k) in zip(size, self.study_ids, self.dicom_ids):
                rows.append(
                    {
                        'size': i,
                        'study_id': j,
                        'dicom_id': k,
                    }
                )
        else:
            for (i, j) in zip(size, self.study_ids):
                rows.append(
                    {
                        'size': i,
                        'study_id': j,
                    }
                )

        if torch.distributed.is_initialized():  # If DDP
            rows_gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(rows_gathered, rows)
            rows = [j for i in rows_gathered for j in i]
        return rows

    def accumulate_scores(self, rows, epoch):

        df = pd.DataFrame(rows)

        # Drop duplicates caused by DDP:
        key = 'dicom_id' if self.accumulate_over_dicoms else 'study_id'
        df = df.drop_duplicates(subset=[key])

        # Save the scores:
        def save_scores():
            df.to_csv(
                os.path.join(
                    self.save_dir,
                    f'{self.split}_epoch-{epoch}_scores_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv',
                ),
                index=False,
            )
        if not torch.distributed.is_initialized():
            save_scores()
        elif torch.distributed.get_rank() == 0:
            save_scores()

        # Number of examples:
        prefix = f'{self.split}_{self.metric_name}_'
        scores = {f'{prefix}num_study_ids': float(df.study_id.nunique())}
        if self.accumulate_over_dicoms:
            scores[f'{prefix}num_dicom_ids'] = float(df.dicom_id.nunique())

        # Take the mean error over the DICOMs (if the sum is taken instead, studies with more DICOMs would be given more
        # importance. We want every study to be given equal importance).
        if self.accumulate_over_dicoms:
            df = df.drop(['dicom_id'], axis=1).groupby('study_id', as_index=False).mean()

        df = df.drop(['study_id'], axis=1)
        mean_scores = {f'{prefix}{k}': v for k, v in df.mean().to_dict().items()}
        scores = {**mean_scores, **scores}
        scores.pop('study_id', None)

        return scores


class BERTScoreRoBERTaLargeMetric(CXRReportGenerationMetric):
    """
    BERTScore for CXR report generation evaluation.
    """

    def __init__(self, num_workers, **kwargs):
        """
        Argument/s:
            num_workers - the number of workers for BERTScore.
        """
        super().__init__(metric_name='bertscore', **kwargs)
        self.num_workers = num_workers

    def init_metric(self):

        # BertScore:
        self.bert_scorer = BERTScorer(
            model_type='roberta-large',
            num_layers=17,
            batch_size=self.mbatch_size,
            nthreads=self.num_workers,
            all_layers=False,
            idf=False,
            lang='en',
            device=self.device,
            rescale_with_baseline=True,
        )

    def cleanup_metric(self):
        # del self.bert_scorer, self.tokenizer
        del self.bert_scorer

    def metric_scoring(self, batch):

        y_hat = [j['synthetic'] for j in batch]
        y = [j['radiologist'] for j in batch]
        study_ids = [j['study_id'] for j in batch]
        if self.accumulate_over_dicoms:
            dicom_ids = [j['dicom_id'] for j in batch]

        with torch.no_grad():
            bert_scores, _ = self.bert_scorer.score(y_hat, y, batch_size=self.mbatch_size, return_hash=True)

        precision = bert_scores[0].tolist()
        recall = bert_scores[1].tolist()
        f1 = bert_scores[2].tolist()

        mbatch_rows = []
        if self.accumulate_over_dicoms:
            for x, y, s_1, s_2, s_3 in zip(dicom_ids, study_ids, f1, precision, recall):
                mbatch_rows.append({'dicom_id': x, 'study_id': y, 'f1': s_1, 'precision': s_2, 'recall': s_3})
        else:
            for x, s_1, s_2, s_3 in zip(study_ids, f1, precision, recall):
                mbatch_rows.append({'study_id': x, 'f1': s_1, 'precision': s_2, 'recall': s_3})

        return mbatch_rows


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
    def __init__(self, device, p=0.1):
        super(CheXbert, self).__init__()

        self.device = device
        
        # Downloading pretrain model from huggingface:
        ckpt_path = hf_hub_download(repo_id='StanfordAIMI/RRG_scorers', filename='chexbert.pth')

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig().from_pretrained('bert-base-uncased')

        with torch.no_grad():

            self.bert = BertModel(config)
            self.dropout = nn.Dropout(p)

            hidden_size = self.bert.pooler.dense.in_features

            # Classes: present, absent, unknown, blank for 12 conditions + support devices:
            self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])

            # Classes: yes, no for the 'no finding' observation:
            self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

            # Load CheXbert checkpoint:
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
            reports[i] = reports[i].replace(r"\n", " ")
            reports[i] = reports[i].replace(r"\s+", " ")
            reports[i] = reports[i].replace(r"\s+(?=[\.,])", "")
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


class CheXbertMetric(CXRReportGenerationMetric):
    """
    CheXbert classification metrics for CXR report generation evaluation.
    """

    def __init__(self, **kwargs):
        """
        Argument/s:
            ckpt_dir - path to the checkpoint directory.
            bert_path - path to the Hugging Face BERT checkpoint (for the BERT configuration).
            checkpoint_path - path to the CheXbert checkpoint.
        """
        super().__init__(metric_name='chexbert', **kwargs)

        self.add_state('reports', default=[])

    def init_metric(self):
        self.chexbert = CheXbert(device=self.device).to(self.device)

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

        # Initialise class scores dataframe:
        scores['class'] = pd.DataFrame()

        # Sum over study_ids for class scores:
        for i in ['tp', 'tn', 'fp', 'fn']:
            scores['class'][i] = scores[i][PATHOLOGIES].sum()

        # Accuracy:
        scores['class']['accuracy'] = np.where(
            (scores['class']['tp'] + scores['class']['tn'] + scores['class']['fp'] + scores['class']['fn']) == 0,
            np.nan,  # Undefined when there are no true/false positives or negatives.
            (scores['class']['tp'] + scores['class']['tn']) / 
            (scores['class']['tp'] + scores['class']['tn'] + scores['class']['fp'] + scores['class']['fn'])
        )

        # Precision:
        scores['class']['precision'] = np.where(
            (scores['class']['tp'] + scores['class']['fp']) == 0,
            np.nan,  # Undefined when there are no true or false positives.
            scores['class']['tp'] / (scores['class']['tp'] + scores['class']['fp'])
        )

        # Recall:
        scores['class']['recall'] = np.where(
            (scores['class']['tp'] + scores['class']['fn']) == 0,
            np.nan,  # Undefined when there are no true positives or false negatives.
            scores['class']['tp'] / (scores['class']['tp'] + scores['class']['fn'])
        )

        # F1 Score:
        scores['class']['f1'] = np.where(
            (scores['class']['tp'] + 0.5 * (scores['class']['fp'] + scores['class']['fn'])) == 0,
            np.nan,  # Undefined when the denominator for F1 is zero.
            scores['class']['tp'] / (scores['class']['tp'] + 0.5 * (scores['class']['fp'] + scores['class']['fn']))
        )

        # Alternate F1 Score:
        scores['class']['f1_alternate'] = np.where(
            (scores['class']['precision'] + scores['class']['recall']) == 0,
            np.nan,  # Undefined when precision + recall is zero.
            (2 * scores['class']['precision'] * scores['class']['recall']) / (scores['class']['precision'] + scores['class']['recall'])
        )

        # Macro-averaging:
        scores['averaged'] = pd.DataFrame()
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


class BLEUMetric(CXRReportGenerationMetric):
    """
    BLEU metric for CXR report generation evaluation.
    """

    def __init__(self, **kwargs):
        super().__init__(metric_name='bleu', scoring_after_gather=True, compute_in_batches=False, **kwargs)

    def metric_scoring(self, rows):

        bleu = BLEUScore()
        for i in range(len(rows)):
            rows[i]['bleu_4'] = bleu([rows[i]['synthetic']], [[rows[i]['radiologist']]]).item()
        
        return rows


class ROUGELMetric(CXRReportGenerationMetric):
    """
    ROUGE metric for CXR report generation evaluation.
    """

    def __init__(self, **kwargs):
        super().__init__(metric_name='rouge', scoring_after_gather=True, compute_in_batches=False, **kwargs)

    def metric_scoring(self, rows):

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        for i in range(len(rows)):
            rows[i]['rouge_l_f1'] = scorer.score(rows[i]['synthetic'], rows[i]['radiologist'])['rougeL'].fmeasure
        
        return rows


class CXRBERTMetric(CXRReportGenerationMetric):
    """
    CXR-BERT similarity for CXR report generation evaluation.
    """

    def __init__(self, **kwargs):
        super().__init__(metric_name='cxrbert', **kwargs)

    def init_metric(self):

        # Load the model and tokenizer
        ckpt_name = 'microsoft/BiomedVLP-CXR-BERT-specialized'
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(ckpt_name, trust_remote_code=True).to(self.device)

        self.model.eval()

    def cleanup_metric(self):
        del self.tokenizer, self.model

    def metric_scoring(self, batch):

        y_hat = [i['synthetic'] for i in batch]
        y = [i['radiologist'] for i in batch]
        study_ids = [i['study_id'] for i in batch]
        if self.accumulate_over_dicoms:
            dicom_ids = [i['dicom_id'] for i in batch]

        # Tokenize and compute the sentence embeddings
        tokenizer_output = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=y_hat,
            add_special_tokens=True,
            padding='longest',
            return_tensors='pt',
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        )

        prediction_embeddings = self.model(
            input_ids=tokenizer_output.input_ids.to(self.device),
            attention_mask=tokenizer_output.attention_mask.to(self.device),
            output_cls_projected_embedding=True,
            return_dict=False,
        )

        tokenizer_output = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=y,
            add_special_tokens=True,
            padding='longest',
            return_tensors='pt',
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        )

        label_embeddings = self.model(
            input_ids=tokenizer_output.input_ids.to(self.device),
            attention_mask=tokenizer_output.attention_mask.to(self.device),
            output_cls_projected_embedding=True,
            return_dict=False,
        )

        # Compute the cosine similarity of sentence embeddings obtained from input text prompts.
        sim = torch.nn.functional.cosine_similarity(
            prediction_embeddings[2],
            label_embeddings[2],
        )

        mbatch_rows = []
        if self.accumulate_over_dicoms:
            for x, y, z in zip(dicom_ids, study_ids, sim.tolist()):
                mbatch_rows.append({'dicom_id': x, 'study_id': y, 'similarity': z})
        else:
            for x, y in zip(study_ids, sim.tolist()):
                mbatch_rows.append({'study_id': x, 'similarity': y})

        return mbatch_rows


class RadGraphMetric(CXRReportGenerationMetric):
    """
    RadGraph for CXR report generation evaluation.
    """

    def __init__(self, **kwargs):
        super().__init__(metric_name='rg', **kwargs)

    def init_metric(self):
        self.metric = F1RadGraph(reward_level='all').to(device=self.device)

    def cleanup_metric(self):
        del self.metric

    def metric_scoring(self, batch):

        y_hat = [i['synthetic'] for i in batch]
        y = [i['radiologist'] for i in batch]
        study_ids = [i['study_id'] for i in batch]
        if self.accumulate_over_dicoms:
            dicom_ids = [i['dicom_id'] for i in batch]

        _, scores, _, _ = self.metric(hyps=y_hat, refs=y)

        mbatch_rows = []
        if self.accumulate_over_dicoms:
            for x, y, rg_e_f1, rg_er_f1, rg_er_hat_f1 in zip(dicom_ids, study_ids, scores[0], scores[1], scores[2]):
                mbatch_rows.append({'dicom_id': x, 'study_id': y, 'rg_e_f1': rg_e_f1, 'rg_er_f1': rg_er_f1, 'rg_e_hat_f1': rg_er_hat_f1})
        else:
            for x, rg_e_f1, rg_er_f1, rg_er_hat_f1 in zip(study_ids, scores[0], scores[1], scores[2]):
                mbatch_rows.append({'study_id': x,  'rg_e_f1': rg_e_f1, 'rg_er_f1': rg_er_f1, 'rg_e_hat_f1': rg_er_hat_f1})

        return mbatch_rows
    
    
class AbsenceOfRepeatedNGramesMetric(CXRReportGenerationMetric):

    def __init__(self, **kwargs):
        super().__init__(metric_name='aor_ngrams', **kwargs)
        self.n = 3
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('aehrc/cxrmate-ed')

    def metric_scoring(self, batch):

        y_hat = [j['synthetic'] for j in batch]
        y = [j['radiologist'] for j in batch]
        study_ids = [j['study_id'] for j in batch]
        if self.accumulate_over_dicoms:
            dicom_ids = [j['dicom_id'] for j in batch]
        
        arng = []  # Absence of repeated n-grams scores.
        for i in y_hat:
            
            tokens = self.tokenizer.tokenize(i)
            
            # If the sequence is shorter than the n-gram size, maximum reward is given:
            if len(tokens) < self.n:
                arng.append(1.0)
                continue
            
            # Count the occurrences of n-grams:
            ngram_counts = {}
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                if ngram in ngram_counts:
                    ngram_counts[ngram] += 1
                else:
                    ngram_counts[ngram] = 1
            
            # Total number of n-grams:
            total_ngrams = len(tokens) - self.n + 1

            # Calculate the number of repeated n-grams:
            repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)

            # Calculate the reward as the absence of repeated n-grams:
            arng.append(1.0 - (repeated_ngrams / total_ngrams)) # Invert the penalty to get reward:
        arng = torch.tensor(arng, device=self.device)    
                
        score = arng.tolist()

        mbatch_rows = []
        if self.accumulate_over_dicoms:
            for x, y, z in zip(dicom_ids, study_ids, score):
                mbatch_rows.append({'dicom_id': x, 'study_id': y, 'score': z})
        else:
            for x, y in zip(study_ids, score):
                mbatch_rows.append({'study_id': x, 'score': y})

        return mbatch_rows
    


def compute_largest_cluster(sentences):
    """
    Computes the largest cluster of sentences using K-means clustering, finds the sentences within the largest cluster, and orders them by their distance to the cluster center.

    Args:
        sentences (list): List of sentences to be clustered.

    Returns:
        tuple: A tuple containing:
            - embeddings (ndarray): Normalized embeddings of the input sentences.
            - sentences_of_largest_cluster (list): Sentences in the largest cluster, ordered by their proximity
              to the cluster center.
    """
    if len(sentences) == 0:
        return None, None
    embeddings, kmeans = compute_kmeans(sentences)
    cluster_sizes = np.bincount(kmeans.labels_)
    largest_cluster_idx = np.argmax(cluster_sizes)
    cluster_member_ids = np.where(kmeans.labels_ == largest_cluster_idx)[0]
    sentences_of_largest_cluster = [sentences[i] for i in cluster_member_ids]

    largest_cluster_mean = kmeans.cluster_centers_[largest_cluster_idx]
    embeddings_of_largest_cluster = [embeddings[i] for i in cluster_member_ids]
    distances = distance.cdist(
        embeddings_of_largest_cluster, [largest_cluster_mean], "cosine"
    ).flatten()
    closest_point_indices = np.argsort(distances)[0]

    sentences_of_largest_cluster = sentences_of_largest_cluster[closest_point_indices]

    return embeddings, sentences_of_largest_cluster


def compute_kmeans(sentences):
    """
    Computes K-means clustering for a list of sentences by generating their embeddings, normalizing the embeddings, and determining the optimal number of clusters using binary search.

    Args:
        sentences (list): List of sentences to be clustered.

    Returns:
        tuple: A tuple containing:
            - embeddings (ndarray): Normalized embeddings of the input sentences.
            - kmeans (KMeans): The KMeans object with the optimal number of clusters determined.
    """
    # sentence embeddings
    model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
    embeddings = model.encode(sentences)
    # normalize the embeddings for equivalent computation of the cosine distance
    embeddings = preprocessing.normalize(embeddings)
    # compute the number of clusters with binary search
    kmeans = binary_search_optimal_kmeans(embeddings, min_k=0, max_k=len(sentences))
    return embeddings, kmeans


"""
Below is updated to handle when the length of data is one:
"""
def binary_search_optimal_kmeans(data, min_k, max_k):
    """
    Finds the optimal k for KMeans clustering using binary search on the silhouette score.

    Args:
        data (ndarray): Data to cluster (e.g., sentence embeddings).
        min_k (int): Minimum number of clusters for binary search.
        max_k (int): Maximum number of clusters for binary search.

    Returns:
        KMeans: The KMeans object with the optimal number of clusters determined.
    """
    if len(data) < 2:
        # If less than 2 data points, return a trivial KMeans model with 1 cluster:
        return KMeans(n_clusters=1, random_state=42).fit(data)

    best_score = -1
    best_kmeans = None

    # Binary search for the optimal k:
    while min_k <= max_k:
        mid_k = (min_k + max_k) // 2
        if mid_k < 2:  # Silhouette score requires at least 2 clusters.
            break

        # Fit KMeans with mid_k clusters:
        kmeans = KMeans(n_clusters=mid_k, random_state=42).fit(data)
        labels = kmeans.labels_

        # Compute silhouette score:
        score = silhouette_score(data, labels)

        # Update best model if score improves:
        if score > best_score:
            best_score = score
            best_kmeans = kmeans

        # Adjust binary search bounds:
        if score > best_score:
            min_k = mid_k + 1
        else:
            max_k = mid_k - 1

    # If no valid KMeans was created, fallback to 1 cluster:
    if best_kmeans is None:
        return KMeans(n_clusters=1, random_state=42).fit(data)

    return best_kmeans


def flatten_values_lists_of_list_dicts_to_dict(item):
    """
    Flattens a list of dictionaries containing lists of values into a single dictionary.

    Args:
        item (list): List of dictionaries, where each dictionary's values are lists. If any element of the list is itself a list, the function will consider only the first dictionary in that sublist.

    Returns:
        dict: A dictionary where each key corresponds to the keys in the input dictionaries, and each value is a flattened list of all values associated with that key across all input dictionaries.
    """

    result = {}
    for i in item:
        if isinstance(i, list):
            i = i[0]
        for key, lists in i.items():
            if key not in result:
                result[key] = []
            result[key].extend(lists)

    return result


def clean_responses(response):
    if "[Explanation]:" in response:
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        if (
            "[Explanation]:\n    <Explanation>\n" or "[Explanation]:\n<Explanation>"
        ) in response:
            response = response.split("[Explanation]:")[1]
        else:
            response = response.split("[Explanation]:")[-1]
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    return response.replace("</s>", "").replace("<unk>", "")


def make_prompt(text1, text2, max_len=300):
    """
    Creates a prompt for evaluating the accuracy of a candidate radiology report in comparison to a reference radiology report.

    Args:
        text1 (str): Reference radiology report.
        text2 (str): Candidate radiology report.

    Returns:
        str: Formatted prompt string.
    """
    text1 = " ".join(text1.split()[:max_len])
    text2 = " ".join(text2.split()[:max_len])
    prompt = f"Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.\n\n    Process Overview: You will be presented with:\n\n    1. The criteria for making a judgment.\n    2. The reference radiology report.\n    3. The candidate radiology report.\n    4. The desired format for your assessment.\n\n    1. Criteria for Judgment:\n\n    For each candidate report, determine:\n\n    The count of clinically significant errors.\n    The count of clinically insignificant errors.\n\n    Errors can fall into one of these categories:\n\n    a) False report of a finding in the candidate.\n    b) Missing a finding present in the reference.\n    c) Misidentification of a finding's anatomic location/position.\n    d) Misassessment of the severity of a finding.\n    e) Mentioning a comparison that isn't in the reference.\n    f) Omitting a comparison detailing a change from a prior study.\n    Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.\n\n    2. Reference Report:\n    {text1}\n\n    3. Candidate Report:\n    {text2}\n\n    4. Reporting Your Assessment:\n\n    Follow this specific format for your output, even if no errors are found:\n    ```\n    [Explanation]:\n    <Explanation>\n\n    [Clinically Significant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Clinically Insignificant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Matched Findings]:\n    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>\n    ```\n"
    return prompt


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


class GREEN:
    def __init__(self, model_name, device):
        super().__init__()
        warnings.filterwarnings(
            "ignore", message="A decoder-only architecture is being used*"
        )
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings(
            "ignore",
            category=ConvergenceWarning,
            message="Number of distinct clusters.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module="transformers.tokenization_utils_base",
        )
        self.model_name = model_name.split("/")[-1]
        self.batch_size = 4
        self.max_length = 2048
        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]
        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
        ]
        self.prompts = None
        self.completions = None
        self.green_scores = None
        self.error_counts = None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=False if "Phi" in model_name else True,
            torch_dtype=torch.float16,
        ).to(device=device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            add_eos_token=True,
            use_fast=True,
            trust_remote_code=True,
            padding_side="left",
        )

        chat_template = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

        self.tokenizer.chat_template = chat_template
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.clean_up_tokenization_spaces = True
        self.tokenizer.padding_side = "left"

    def __call__(self, refs, hyps):
        dataset = Dataset.from_dict({"reference": refs, "prediction": hyps})
        dataset = self.process_data(dataset)
        self.dataset = dataset
        mean, std, green_scores, summary, results_df = self.infer()
        return mean, std, green_scores, summary, results_df

    def process_data(self, dataset):
        def prompting(examples):
            return {
                "prompt": [
                    make_prompt(r, p)
                    for r, p in zip(examples["reference"], examples["prediction"])
                ]
            }

        dataset = dataset.map(prompting, batched=True)
        return dataset

    @torch.inference_mode()
    def infer(self):
        dataset_dist = self.dataset

        local_completions = []
        local_references = []

        for batch in dataset_dist.iter(batch_size=self.batch_size):
            local_references.extend(batch["prompt"])
            local_completions.extend(self.get_response(batch))

        self.completions = local_completions
        self.prompts = local_references

        if len(self.completions) != len(self.prompts):
            print("Length of prompts and completions are not equal!")

        return self.process_results()

    def tokenize_batch_as_chat(self, batch):
        batch = [
            self.tokenizer.apply_chat_template(
                i, tokenize=False, add_generation_prompt=True
            )
            for i in batch
        ]

        batch = self.tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        return batch

    def get_response(self, batch):
        assert "prompt" in batch.keys(), "prompt is not in batch keys"

        batch = [
            [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]
            for prompt in batch["prompt"]
        ]

        batch = self.tokenize_batch_as_chat(batch)

        outputs = self.model.generate(
            input_ids=batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=2048 + 512,  # Add 512 to allow for prompts of length 2048.
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response_list = []
        if isinstance(responses, list):
            for response in responses:
                response = clean_responses(response)
                response_list.append(response)
        else:
            responses = clean_responses(responses)
            response_list.append(responses)

        return response_list

    def process_results(self):
        self.green_scores = [
            self.compute_green(response) for response in self.completions
        ]
        self.error_counts = pd.DataFrame(
            [self.compute_error_count(response) for response in self.completions],
            columns=self.sub_categories + ["Matched Findings"],
        )

        results_df = pd.DataFrame(
            {
                "reference": self.dataset["reference"],
                "predictions": self.dataset["prediction"],
                "green_analysis": self.completions,
                "green_score": self.green_scores,
                **self.error_counts,
            }
        )

        mean, std, summary = self.compute_summary()

        return mean, std, self.green_scores, summary, results_df

    def compute_error_count(self, response):
        _, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])
        return sig_errors + [matched_findings]

    def compute_green(self, response):
        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])

        if matched_findings == 0:
            return 0

        if sig_present is None or matched_findings is None:
            return None

        return matched_findings / (matched_findings + sum(sig_errors))

    def parse_error_counts(self, text, category, for_reward=False):
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )

        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, text, re.DOTALL)

        sum_counts = 0
        sub_counts = [0 for i in range(6)]

        if not category_text:
            if for_reward:
                return None, None
            return sum_counts, sub_counts
        if category_text.group(1).startswith("No"):
            return sum_counts, sub_counts

        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", category_text.group(1))
            if len(counts) > 0:
                sum_counts = int(counts[0])
            return sum_counts, sub_counts
        else:
            sub_categories = [s.split(" ", 1)[0] + " " for s in self.sub_categories]
            matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

            if len(matches) == 0:
                matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
                sub_categories = [
                    f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
                ]

            for position, sub_category in enumerate(sub_categories):
                for match in range(len(matches)):
                    if matches[match].startswith(sub_category):
                        count = re.findall(r"(?<=: )\b\d+\b(?=\.)", matches[match])
                        if len(count) > 0:
                            sub_counts[position] = int(count[0])
            return sum(sub_counts), sub_counts

    def parse_error_sentences(self, response, category):
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )
        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, response, re.DOTALL)
        sub_category_dict_sentences = {}
        for sub_category in self.sub_categories:
            sub_category_dict_sentences[sub_category] = []

        if not category_text:
            return sub_category_dict_sentences
        if category_text.group(1).startswith("No"):
            return sub_category_dict_sentences

        if category == "Matched Findings":
            return (
                category_text.group(1).rsplit(":", 1)[-1].rsplit(".", 1)[-1].split(";")
            )

        matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

        if len(matches) == 0:
            matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
            self.sub_categories = [
                f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
            ]

        for position, sub_category in enumerate(self.sub_categories):
            for match in range(len(matches)):
                if matches[match].startswith(sub_category):
                    sentences_list = (
                        matches[match].rsplit(":", 1)[-1].split(".", 1)[-1].split(";")
                    )
                    sub_category_dict_sentences[self.sub_categories[position]] = (
                        sentences_list
                    )

        return sub_category_dict_sentences

    def compute_sentences(self, response):
        return self.parse_error_sentences(response, self.categories[0])

    def get_representative_sentences(self, responses):
        list_sentences = []
        for i in responses:
            sentences = self.compute_sentences(i)
            list_sentences.append(sentences)

        dict_sentences = flatten_values_lists_of_list_dicts_to_dict(list_sentences)

        result_sentences_dict = {}

        for i in self.sub_categories:
            sentences = dict_sentences[i]
            sentences = [i for i in sentences if i.strip() != ""]
            _, sentences_of_largest_cluster = compute_largest_cluster(sentences)
            result_sentences_dict[i] = sentences_of_largest_cluster

        return result_sentences_dict

    def compute_accuracy(self, responses):
        counts = []
        for response in responses:
            _, sig_errors = self.parse_error_counts(response, self.categories[0])
            counts.append(sig_errors)

        counts = np.array(counts)

        dict_acc = {}
        for i in range(len(self.sub_categories)):
            error_counts = counts[:, i]
            accuracy = np.mean(error_counts == 0)
            dict_acc[self.sub_categories[i]] = accuracy

        return dict_acc

    def compute_summary(self):
        representative_sentences = self.get_representative_sentences(self.completions)
        accuracies = self.compute_accuracy(self.completions)
        mean = np.mean(self.green_scores)
        std = np.std(self.green_scores)

        summary = f"\n-------------{self.model_name}----------------\n [Summary]: Green average {mean} and standard deviation {std} \n [Clinically Significant Errors Analyses]: <accuracy>. <representative error>\n\n"
        for idx, sub_category in enumerate(self.sub_categories):
            accuracy = accuracies[sub_category]
            sentences = representative_sentences[sub_category]
            summary += f"{sub_category}: {accuracy}. \n {sentences} \n\n"
        summary += "----------------------------------\n"

        return mean, std, summary
    

class GREENMetric(CXRReportGenerationMetric):
    """
    GREEN for CXR report generation evaluation.
    """

    def __init__(self, **kwargs):
        super().__init__(metric_name='green', **kwargs)
        disable_progress_bar()

    def init_metric(self):
        self.metric = GREEN('StanfordAIMI/GREEN-radllama2-7b', device=self.device)

    def cleanup_metric(self):
        del self.metric

    def metric_scoring(self, batch):

        y_hat = [i['synthetic'] for i in batch]
        y = [i['radiologist'] for i in batch]
        study_ids = [i['study_id'] for i in batch]
        if self.accumulate_over_dicoms:
            dicom_ids = [i['dicom_id'] for i in batch]

        _, _, scores, _, _ = self.metric(y, y_hat)

        mbatch_rows = []
        if self.accumulate_over_dicoms:
            for x, y, z in zip(dicom_ids, study_ids, scores):
                mbatch_rows.append({'dicom_id': x, 'study_id': y, 'green': z})
        else:
            for x, y in zip(study_ids, scores):
                mbatch_rows.append({'study_id': x,  'green': y})

        return mbatch_rows
    