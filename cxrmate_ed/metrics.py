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
import torch.nn as nn
import transformers
from bert_score import BERTScorer
from green_score import GREEN
from huggingface_hub import hf_hub_download
from radgraph import F1RadGraph
from rouge_score import rouge_scorer
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
    

class GREENModified(GREEN):
    def __init__(self, model_name, output_dir="."):
        nn.Module.__init__(self)
        warnings.filterwarnings("ignore", message="A decoder-only architecture is being used*")
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning, message="Number of distinct clusters.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

        self.model_name = model_name.split("/")[-1]
        self.output_dir = output_dir
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
        )
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


class GREENMetric(CXRReportGenerationMetric):
    """
    GREEN for CXR report generation evaluation.
    """

    def __init__(self, **kwargs):
        super().__init__(metric_name='green', **kwargs)

    def init_metric(self):
        self.metric = GREENModified('StanfordAIMI/GREEN-radllama2-7b')
        self.metric.to(device=self.device)

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