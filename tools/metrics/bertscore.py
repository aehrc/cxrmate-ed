# from torchmetrics.text import BERTScore
# from transformers import AutoTokenizer
import os

import torch
from bert_score import BERTScorer
# from pathlib import Path
from torchmetrics import Metric

from tools.metrics.mimic_cxr import MIMICCXRReportGenerationMetric


class BERTScoreRoBERTaLargeMetric(MIMICCXRReportGenerationMetric):
    """
    BERTScore for MIMIC-CXR. If multiple reports are generated per study_id, each error type is
    summed over the dicom_ids.
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
            # baseline_path=os.path.join(self.ckpt_dir, 'bert_score', 'rescale_baseline', 'en', 'roberta-large.tsv'),
        )

        # RoBERTa tokenizer:
        # self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.ckpt_dir, 'roberta-large'))
        # os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
