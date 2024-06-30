import os
import pandas as pd
import time
import torch

from tools.metrics.mimic_cxr import MIMICCXRReportGenerationMetric


class ReportTokenIdentifiersLogger(MIMICCXRReportGenerationMetric):
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
