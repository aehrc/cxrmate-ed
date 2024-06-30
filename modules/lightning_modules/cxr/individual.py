import math

from modules.lightning_modules.ed.individual import EDExclusive
from tools.mimic_iv.ed_cxr.records_rev_a import EDCXRSubjectRecords


class Indication(EDExclusive):

    def __init__(self, mimic_iv_duckdb_path, **kwargs):

        records = EDCXRSubjectRecords(database_path=mimic_iv_duckdb_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))
        records.ed_module_tables = {}
        records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
        records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = ['indication']
        super().__init__(records=records, **kwargs)


class History(EDExclusive):

    def __init__(self, mimic_iv_duckdb_path, **kwargs):

        records = EDCXRSubjectRecords(database_path=mimic_iv_duckdb_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))
        records.ed_module_tables = {}
        records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
        records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = ['history']
        super().__init__(records=records, **kwargs)


class Metadata(EDExclusive):

    def __init__(self, mimic_iv_duckdb_path, **kwargs):

        records = EDCXRSubjectRecords(database_path=mimic_iv_duckdb_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))
        records.ed_module_tables = {}
        records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = []
        super().__init__(records=records, **kwargs)
