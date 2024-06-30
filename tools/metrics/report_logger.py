import os
import time

import pandas as pd
import torch

from tools.metrics.mimic_cxr import MIMICCXRReportGenerationMetric


class ReportLogger(MIMICCXRReportGenerationMetric):
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
