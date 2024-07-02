
import math

from modules.lightning_modules.ed_cxr.freeze_encoder_partial_warm_start_optimiser import (
    FreezeEncoderPartialWarmStartOptimiser,
)
from modules.transformers.cxrmate_ed.dataset import StudyIDEDStayIDSubset
from modules.transformers.cxrmate_ed.records import EDCXRSubjectRecords


class EDExclusive(FreezeEncoderPartialWarmStartOptimiser):

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """

        edstays_study_ids = self.records.connect.sql('FROM edstays_study_ids').df()['study_id'].to_list()

        if stage == 'fit' or stage is None:
            self.train_set = StudyIDEDStayIDSubset(
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.train_transforms,
                split='train',
                max_images_per_study=5,
                records=self.records,
                study_id_inclusion_list=edstays_study_ids,
            )
            print(f'No. of training examples: {self.train_set.__len__()}.')
            print(
                f'No. of training dicom_ids, study_ids, & subject_ids: {self.train_set.num_dicom_ids},',
                f'{self.train_set.num_study_ids}, & {self.train_set.num_subject_ids}.',
            )

        if stage == 'fit' or stage == 'validate' or stage is None:
            self.val_set = StudyIDEDStayIDSubset(
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.test_transforms,
                split='validate',
                max_images_per_study=5,
                records=self.records,
                study_id_inclusion_list=edstays_study_ids,
            )
            print(f'No. of validation examples: {self.val_set.__len__()}.')
            print(
                f'No. of validation dicom_ids, study_ids, & subject_ids: {self.val_set.num_dicom_ids},',
                f'{self.val_set.num_study_ids}, & {self.val_set.num_subject_ids}.',
            )

        if stage == 'test' or stage is None:
            self.test_set = StudyIDEDStayIDSubset(
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.test_transforms,
                split='test',
                max_images_per_study=5,
                records=self.records,
                study_id_inclusion_list=edstays_study_ids,
            )
            print('No. of test examples: {}.'.format(self.test_set.__len__()))
            print(
                f'No. of test dicom_ids, study_ids, & subject_ids: {self.test_set.num_dicom_ids},',
                f'{self.test_set.num_study_ids}, & {self.test_set.num_subject_ids}.',
            )


class EffectiveInputs(EDExclusive):
    
    def __init__(self, database_path, **kwargs):
        records = EDCXRSubjectRecords(database_path=database_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))

        records.ed_module_tables = {k: records.ed_module_tables[k] for k in ['medrecon', 'edstays', 'triage', 'vitalsign']}
        records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
        records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = ['indication', 'history']

        super().__init__(records=records, **kwargs)


class MinusMedrecon(EDExclusive):
    
    def __init__(self, database_path, **kwargs):
        records = EDCXRSubjectRecords(database_path=database_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))

        records.ed_module_tables = {k: records.ed_module_tables[k] for k in ['edstays', 'triage', 'vitalsign']}
        records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
        records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = ['indication', 'history']

        super().__init__(records=records, **kwargs)
        
        
class MinusEdstays(EDExclusive):
    
    def __init__(self, database_path, **kwargs):
        records = EDCXRSubjectRecords(database_path=database_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))

        records.ed_module_tables = {k: records.ed_module_tables[k] for k in ['medrecon', 'triage', 'vitalsign']}
        records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
        records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = ['indication', 'history']

        super().__init__(records=records, **kwargs)
        
        
class MinusTriage(EDExclusive):
    
    def __init__(self, database_path, **kwargs):
        records = EDCXRSubjectRecords(database_path=database_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))

        records.ed_module_tables = {k: records.ed_module_tables[k] for k in ['medrecon', 'edstays', 'vitalsign']}
        records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
        records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = ['indication', 'history']

        super().__init__(records=records, **kwargs)
        
        
class MinusVitalsign(EDExclusive):
    
    def __init__(self, database_path, **kwargs):
        records = EDCXRSubjectRecords(database_path=database_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))

        records.ed_module_tables = {k: records.ed_module_tables[k] for k in ['medrecon', 'edstays', 'triage']}
        records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
        records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = ['indication', 'history']

        super().__init__(records=records, **kwargs)
        
        
class MinusIndication(EDExclusive):
    
    def __init__(self, database_path, **kwargs):
        records = EDCXRSubjectRecords(database_path=database_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))

        records.ed_module_tables = {k: records.ed_module_tables[k] for k in ['medrecon', 'edstays', 'triage', 'vitalsign']}
        records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
        records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = ['history']

        super().__init__(records=records, **kwargs)
        
        
class MinusHistory(EDExclusive):
    
    def __init__(self, database_path, **kwargs):
        records = EDCXRSubjectRecords(database_path=database_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))

        records.ed_module_tables = {k: records.ed_module_tables[k] for k in ['medrecon', 'edstays', 'triage', 'vitalsign']}
        records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
        records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = ['indication']

        super().__init__(records=records, **kwargs)
        
        
class MinusTimedelta(EDExclusive):
    
    def __init__(self, database_path, **kwargs):
        records = EDCXRSubjectRecords(database_path=database_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))

        records.ed_module_tables = {k: records.ed_module_tables[k] for k in ['medrecon', 'edstays', 'triage', 'vitalsign']}
        records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
        records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = ['indication', 'history']

        super().__init__(records=records, add_time_deltas=False, **kwargs)
        