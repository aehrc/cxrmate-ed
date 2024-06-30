import os
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from tools.metrics.nlg import NLGMetric


class MIMICCXRReportGenerationMetric(NLGMetric):
    """
    Torchmetric for metrics for chest X-ray report generation evaluation with the MIMIC-CXR dataset.
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
