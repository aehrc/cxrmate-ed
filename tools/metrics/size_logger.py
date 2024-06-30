import os
import time

import pandas as pd
import torch

from tools.metrics.mimic_cxr import MIMICCXRReportGenerationMetric


class SizeLogger(MIMICCXRReportGenerationMetric):

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
