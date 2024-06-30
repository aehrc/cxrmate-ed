import re
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from tools.metrics.mimic_cxr import MIMICCXRReportGenerationMetric


class COCONLGMIMICCXRMetrics(MIMICCXRReportGenerationMetric):
    """
    COCO NLG metrics for MIMIC-CXR. Equal importance is given to each study. Thus, scores are averaged over the
    study_ids. If multiple reports are generated per study_id, its score is the mean score for each of
    its dicom_ids.
    """

    def __init__(self, metrics: Optional[list] = None, use_tokenizer: bool = True, **kwargs):
        """
        metrics - which metrics to use.
        use_tokenizer - use the PTBTokenizer.
        """
        super().__init__(metric_name='coco', scoring_after_gather=True, compute_in_batches=False, **kwargs)
        self.metrics = ['bleu', 'cider', 'meteor', 'rouge', 'spice'] if metrics is None else metrics
        self.metrics = [metric.lower() for metric in metrics]
        self.use_tokenizer = use_tokenizer

        if 'bleu' in self.metrics:
            self.bleu = Bleu(4)
        if 'meteor' in self.metrics:
            self.meteor = Meteor()
        if 'rouge' in self.metrics:
            self.rouge = Rouge()
        if 'cider' in self.metrics:
            self.cider = Cider()
        if 'spice' in self.metrics:
            self.spice = Spice()
        if self.use_tokenizer:
            self.tokenizer = PTBTokenizer()

    def metric_scoring(self, rows):

        for i in range(len(rows)):
            rows[i]['radiologist'] = [rows[i]['radiologist']]

        key = 'dicom_id' if self.accumulate_over_dicoms else 'study_id'
        synthetic, radiologist = {}, {}
        if self.use_tokenizer:
            for i in rows:
                idx = i[key].item() if isinstance(i[key], torch.Tensor) else i[key]
                idx = int(idx) if isinstance(idx, np.int64) else idx  # SPICE cannot handle numpy.int.
                synthetic[idx] = [{'caption': re.sub(' +', ' ', i['synthetic'])}]
                radiologist[idx] = [{'caption': re.sub(' +', ' ', m)} for m in i['radiologist']]

            synthetic = self.tokenizer.tokenize(synthetic)
            radiologist = self.tokenizer.tokenize(radiologist)

        else:
            for i in rows:
                idx = i[key].item() if isinstance(i[key], torch.Tensor) else i[key]
                idx = int(idx) if isinstance(idx, np.int64) else idx  # SPICE cannot handle numpy.int.
                synthetic[idx] = [re.sub(' +', ' ', i['synthetic'])]
                radiologist[idx] = [re.sub(' +', ' ', m) for m in i['label']]

        # COCO NLG metric scores:
        if 'bleu' in self.metrics:
            _, metric_scores = self.bleu.compute_score(radiologist, synthetic)
            rows = [dict(i, **{'bleu_1': j}) for i, j in zip(rows, metric_scores[0])]
            rows = [dict(i, **{'bleu_2': j}) for i, j in zip(rows, metric_scores[1])]
            rows = [dict(i, **{'bleu_3': j}) for i, j in zip(rows, metric_scores[2])]
            rows = [dict(i, **{'bleu_4': j}) for i, j in zip(rows, metric_scores[3])]
        if 'meteor' in self.metrics:
            _, metric_scores = self.meteor.compute_score(radiologist, synthetic)
            rows = [dict(i, **{'meteor': j}) for i, j in zip(rows, metric_scores)]
        if 'rouge' in self.metrics:
            _, metric_scores = self.rouge.compute_score(radiologist, synthetic)
            rows = [dict(i, **{'rouge': j}) for i, j in zip(rows, metric_scores)]
        if 'cider' in self.metrics:
            _, metric_scores = self.cider.compute_score(radiologist, synthetic)
            rows = [dict(i, **{'cider': j}) for i, j in zip(rows, metric_scores)]
        if 'spice' in self.metrics:
            _, metric_scores = self.spice.compute_score(radiologist, synthetic)
            rows = [dict(i, **{'spice': j}) for i, j in zip(rows, metric_scores)]

        return rows
