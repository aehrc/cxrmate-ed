from typing import Optional
from torchmetrics import Metric
import torch


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
