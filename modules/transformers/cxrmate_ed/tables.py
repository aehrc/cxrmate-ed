from collections import OrderedDict
from typing import Optional

ed_cxr_token_type_ids = {
    'medrecon': 0, 
    'edstays': 1, 
    'triage': 2, 
    'vitalsign': 3, 
    'pyxis': 4, 
    'mimic_cxr_2_0_0_metadata': 5, 
    'medrecon_name': 6, 
    'triage_chiefcomplaint': 7, 
    'triage_pain': 8, 
    'vitalsign_pain': 9, 
    'indication': 10, 
    'history': 11, 
    'findings': 12, 
    'impression': 13, 
    'image': 14,
    'comparison': 15,
    'previous_findings': 16,
    'previous_impression': 17,
    'previous_image': 18,
}

NUM_ED_CXR_TOKEN_TYPE_IDS = max(ed_cxr_token_type_ids.values()) + 1


class TableConfig:
    def __init__(
        self,
        name: str,
        hadm_id_filter: bool = False, 
        stay_id_filter: bool = False, 
        study_id_filter: bool = False,
        subject_id_filter: bool = True,
        load: Optional[bool] = None,
        groupby: Optional[str] = None,
        index_columns: list = [],
        text_columns: list = [],
        value_columns: list = [],
        time_columns: list = [],
        target_sections: list = [],
        use_start_time: bool = False,
        mimic_cxr_sectioned: bool = False,
    ):
        self.name = name
        self.hadm_id_filter = hadm_id_filter 
        self.stay_id_filter = stay_id_filter
        self.study_id_filter = study_id_filter
        self.subject_id_filter = subject_id_filter
        self.load = load
        self.groupby = groupby
        self.index_columns_source = [index_columns] if isinstance(index_columns, str) else index_columns
        self.index_columns = [f'{i}_index' for i in self.index_columns_source]
        self.text_columns = [text_columns] if isinstance(text_columns, str) else text_columns
        self.value_columns = [value_columns] if isinstance(value_columns, str) else value_columns
        self.time_columns = [time_columns] if isinstance(time_columns, str) else time_columns
        self.target_sections = [target_sections] if isinstance(target_sections, str) else target_sections
        self.use_start_time = use_start_time
        self.mimic_cxr_sectioned = mimic_cxr_sectioned

        assert self.time_columns is None or isinstance(self.time_columns, list)

        self.value_column_to_idx = {}
        self.total_indices = None


# ed module:
"""
Order the tables for position_ids based on their order of occurance (for cases where their time deltas are matching). 
The way that they are ordered here is the way that they will be ordered as input.

1. medrecon - the medications which the patient was taking prior to their ED stay.
2. edstays - patient stays are tracked in the edstays table.
3. triage - information collected from the patient at the time of triage.
4. vitalsign - aperiodic vital signs documented for patients during their stay.
5. pyxis - dispensation information for medications provided by the BD Pyxis MedStation (position is interchangable with 4).
"""
ed_module_tables = OrderedDict(
    {
        'medrecon': TableConfig(
            'Medicine reconciliation',
            stay_id_filter=True, 
            load=True,
            index_columns=['gsn', 'ndc', 'etc_rn', 'etccode'],
            text_columns='name',
            groupby='stay_id',
            use_start_time=True,
        ),
        'edstays': TableConfig(
            'ED admissions',
            stay_id_filter=True,
            load=True, 
            index_columns=['gender', 'race', 'arrival_transport'],
            groupby='stay_id',
            time_columns='intime',
        ),
        'triage': TableConfig(  
            'Triage', 
            stay_id_filter=True,
            load=True,
            text_columns=['chiefcomplaint', 'pain'],
            value_columns=['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'acuity'],
            groupby='stay_id',
            use_start_time=True,
        ),
        'vitalsign': TableConfig(
            'Aperiodic vital signs',
            stay_id_filter=True,
            load=True, 
            index_columns=['rhythm'],
            text_columns=['pain'],
            value_columns=['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp'],
            groupby='charttime',
            time_columns='charttime',
        ),
        'pyxis': TableConfig(
            'Dispensation information for medications provided by the BD Pyxis MedStation', 
            stay_id_filter=True,
            load=True,
            index_columns=['med_rn', 'name', 'gsn_rn', 'gsn'],
            groupby='charttime',
            time_columns='charttime',
        ),
        'diagnosis': TableConfig('Diagnosis', stay_id_filter=True, hadm_id_filter=False),
    }
)

# MIMIC-CXR module:
mimic_cxr_tables = OrderedDict(
    {
        'mimic_cxr_2_0_0_metadata': TableConfig(
            'Metadata', 
            study_id_filter=True,
            load=True,
            index_columns=[
                'PerformedProcedureStepDescription', 
                'ViewPosition', 
                'ProcedureCodeSequence_CodeMeaning', 
                'ViewCodeSequence_CodeMeaning',
                'PatientOrientationCodeSequence_CodeMeaning',
            ],
            groupby='study_id',
        ),
        'mimic_cxr_sectioned': TableConfig(
            'Report sections', 
            mimic_cxr_sectioned=True, 
            subject_id_filter=False,
            load=True,
            groupby='study',
            text_columns=['indication', 'history', 'comparison'],
            target_sections=['findings', 'impression'],
        ),
        'mimic_cxr_2_0_0_chexpert': TableConfig('CheXpert', study_id_filter=True),
        'mimic_cxr_2_0_0_split': TableConfig('Split', study_id_filter=True),
    }
) 

