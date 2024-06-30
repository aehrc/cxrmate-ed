import functools
import os
import re
from collections import OrderedDict
from typing import Dict, List, Optional

import duckdb
import pandas as pd
import torch

from .tables import ed_cxr_token_type_ids, ed_module_tables, mimic_cxr_tables


def mimic_cxr_text_path(dir, subject_id, study_id, ext='txt'):
    return os.path.join(dir, 'p' + str(subject_id)[:2], 'p' + str(subject_id),
                        's' + str(study_id) + '.' + ext)

def format(text):
    # Remove newline, tab, repeated whitespaces, and leading and trailing whitespaces:
    text = re.sub(r'\n|\t', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
        

def df_to_tensor_index_columns(
        df: pd.DataFrame, 
        tensor: torch.Tensor, 
        group_idx_to_y_idx: Dict,
        groupby: str, 
        index_columns: List[str],
    ):
    """
    Converts a dataframe with index columns to a tensor, where each index of the y-axis is determined by the 
    'groupby' column.
    """
    assert len(group_idx_to_y_idx) == tensor.shape[0]
    all_columns = index_columns + [groupby]
    y_indices = [group_idx_to_y_idx[row[groupby]] for _, row in df[all_columns].iterrows() for i in index_columns if row[i] == row[i]]
    x_indices = [row[i] for _, row in df[all_columns].iterrows() for i in index_columns if row[i] == row[i]]
    tensor[y_indices, x_indices] = 1.0
    return tensor


def df_to_tensor_value_columns(
        df: pd.DataFrame, 
        tensor: torch.Tensor, 
        group_idx_to_y_idx: Dict,
        groupby: str, 
        value_columns: List[str], 
        value_column_to_idx: Dict,
    ):
    """
    Converts a dataframe with value columns to a tensor, where each index of the y-axis is determined by the 
    'groupby' column. The x-index is determined by a dictionary using the column name.
    """
    assert len(group_idx_to_y_idx) == tensor.shape[0]
    all_columns = value_columns + [groupby]
    y_indices = [group_idx_to_y_idx[row[groupby]] for _, row in df[all_columns].iterrows() for i in value_columns if row[i] == row[i]]
    x_indices = [value_column_to_idx[i] for _, row in df[all_columns].iterrows() for i in value_columns if row[i] == row[i]]
    element_value = [row[i] for _, row in df[all_columns].iterrows() for i in value_columns if row[i] == row[i]]
    tensor[y_indices, x_indices] = torch.tensor(element_value, dtype=tensor.dtype)
    return tensor


class EDCXRSubjectRecords:
    def __init__(
        self, 
        database_path: str, 
        dataset_dir: Optional[str] = None, 
        reports_dir: Optional[str] = None, 
        token_type_ids_starting_idx: Optional[int] = None,
        time_delta_map = lambda x: x,
        debug: bool = False
    ):

        self.database_path = database_path
        self.dataset_dir = dataset_dir
        self.reports_dir = reports_dir
        self.time_delta_map = time_delta_map
        self.debug = debug

        self.connect = duckdb.connect(self.database_path, read_only=True)

        self.streamlit_flag = False

        self.clear_start_end_times()

        # Module configurations:
        self.ed_module_tables = ed_module_tables
        self.mimic_cxr_tables = mimic_cxr_tables

        lut_info = self.connect.sql("FROM lut_info").df()

        for k, v in (self.ed_module_tables | self.mimic_cxr_tables).items():
            if v.load and (v.value_columns or v.index_columns):
                v.value_column_to_idx = {}
                if v.index_columns:
                    v.total_indices = lut_info[lut_info['table_name'] == k]['end_index'].item() + 1
                else:
                    v.total_indices = 0
                for i in v.value_columns:
                    v.value_column_to_idx[i] = v.total_indices
                    v.total_indices += 1

        # Token type identifiers:
        self.token_type_to_token_type_id = ed_cxr_token_type_ids
        if token_type_ids_starting_idx is not None:
            self.token_type_to_token_type_id = {k: v + token_type_ids_starting_idx for k, v in self.token_type_to_token_type_id.items()}

    def __len__(self):
        return len(self.subject_ids)
    
    def clear_start_end_times(self):
        self.start_time, self.end_time = None, None

    def admission_ed_stay_ids(self, hadm_id):
        if hadm_id:
            return self.connect.sql(f'SELECT stay_id FROM edstays WHERE subject_id = {self.subject_id} AND hadm_id = {hadm_id}').df()['stay_id'].tolist()
        else:
            return self.connect.sql(f'SELECT stay_id FROM edstays WHERE subject_id = {self.subject_id}').df()['stay_id'].tolist()
    
    def subject_study_ids(self):
        mimic_cxr = self.connect.sql(
            f'SELECT study_id, study_datetime FROM mimic_cxr WHERE subject_id = {self.subject_id}',
        ).df()
        if self.start_time and self.end_time:
            mimic_cxr = self.filter_admissions_by_time_span(mimic_cxr, 'study_datetime')
        mimic_cxr = mimic_cxr.drop_duplicates(subset=['study_id']).sort_values(by='study_datetime')
        return dict(zip(mimic_cxr['study_id'], mimic_cxr['study_datetime']))

    def load_ed_module(self, hadm_id=None, stay_id=None, reference_time=None):
        if not self.start_time and stay_id is not None:
            edstay = self.connect.sql(
                f"""
                SELECT intime, outtime
                FROM edstays
                WHERE stay_id = {stay_id}
                """
            ).df()
            self.start_time = edstay['intime'].item()
            self.end_time = edstay['outtime'].item()
        self.load_module(self.ed_module_tables, hadm_id=hadm_id, stay_id=stay_id, reference_time=reference_time)

    def load_mimic_cxr(self, study_id, reference_time=None):
        self.load_module(self.mimic_cxr_tables, study_id=study_id, reference_time=reference_time)
        if self.streamlit_flag:
            self.report_path = mimic_cxr_text_path(self.reports_dir, self.subject_id, study_id, 'txt')

    def load_module(self, module_dict, hadm_id=None, stay_id=None, study_id=None, reference_time=None):
        for k, v in module_dict.items():

            if self.streamlit_flag or v.load:

                query = f"FROM {k}"

                conditions = []
                if hasattr(self, 'subject_id') and v.subject_id_filter:
                    conditions.append(f"subject_id={self.subject_id}")
                if v.hadm_id_filter:
                    assert hadm_id is not None
                    conditions.append(f"hadm_id={hadm_id}")
                if v.stay_id_filter:
                    assert stay_id is not None
                    conditions.append(f"stay_id={stay_id}")
                if v.study_id_filter:
                    assert study_id is not None
                    conditions.append(f"study_id={study_id}")
                if v.mimic_cxr_sectioned:
                    assert study_id is not None
                    conditions.append(f"study='s{study_id}'")
                ands = ['AND'] * (len(conditions) * 2 - 1)
                ands[0::2] = conditions

                if conditions:
                    query += " WHERE ("
                    query += ' '.join(ands) 
                    query += ")"

                df = self.connect.sql(query).df()

                if v.load:

                    columns = [v.groupby] + v.time_columns + v.index_columns + v.text_columns + v.value_columns + v.target_sections

                    # Use the starting time of the stay/admission as the time:
                    if v.use_start_time:
                        df['start_time'] = self.start_time
                        columns += ['start_time']

                    if reference_time is not None:
                        time_column = v.time_columns[-1] if not v.use_start_time else 'start_time'

                        # Remove rows that are after the reference time to maintain causality:
                        df = df[df[time_column] < reference_time]

                if self.streamlit_flag:
                    setattr(self, k, df)

                if v.load:
                    columns = list(dict.fromkeys(columns))  # remove repetitions.
                    df = df.drop(columns=df.columns.difference(columns), axis=1)
                    setattr(self, f'{k}_feats', df)

    def return_ed_module_features(self, stay_id, reference_time=None):
        example_dict = {}
        if stay_id is not None:
            self.load_ed_module(stay_id=stay_id, reference_time=reference_time)
            for k, v in self.ed_module_tables.items():
                if v.load:

                    df = getattr(self, f'{k}_feats')

                    if self.debug:
                        example_dict.setdefault('ed_tables', []).append(k)

                    if not df.empty:
                        
                        assert f'{k}_index_value_feats' not in example_dict

                        # The y-index and the time for each group:
                        time_column = v.time_columns[-1] if not v.use_start_time else 'start_time'
                        group_idx_to_y_idx, group_idx_to_datetime = OrderedDict(), OrderedDict()
                        groups = df.dropna(subset=v.index_columns + v.value_columns + v.text_columns, axis=0, how='all')
                        groups = groups.drop_duplicates(subset=[v.groupby])[list(dict.fromkeys([v.groupby, time_column]))]
                        groups = groups.reset_index(drop=True)
                        for i, row in groups.iterrows():
                            group_idx_to_y_idx[row[v.groupby]] = i
                            group_idx_to_datetime[row[v.groupby]] = row[time_column]

                        if (v.index_columns or v.value_columns) and group_idx_to_y_idx:
                            example_dict[f'{k}_index_value_feats'] = torch.zeros(len(group_idx_to_y_idx), v.total_indices)
                            if v.index_columns:
                                example_dict[f'{k}_index_value_feats'] = df_to_tensor_index_columns(
                                    df=df, 
                                    tensor=example_dict[f'{k}_index_value_feats'], 
                                    group_idx_to_y_idx=group_idx_to_y_idx, 
                                    groupby=v.groupby, 
                                    index_columns=v.index_columns,
                                )
                            if v.value_columns:
                                example_dict[f'{k}_index_value_feats'] = df_to_tensor_value_columns(
                                    df=df, 
                                    tensor=example_dict[f'{k}_index_value_feats'],
                                    group_idx_to_y_idx=group_idx_to_y_idx,
                                    groupby=v.groupby, 
                                    value_columns=v.value_columns,
                                    value_column_to_idx=v.value_column_to_idx
                                )

                            example_dict[f'{k}_index_value_token_type_ids'] = torch.full(
                                [example_dict[f'{k}_index_value_feats'].shape[0]], 
                                self.token_type_to_token_type_id[k], 
                                dtype=torch.long, 
                            )

                            event_times = list(group_idx_to_datetime.values())
                            assert all([i == i for i in event_times])
                            time_delta = [self.compute_time_delta(i, reference_time) for i in event_times]
                            example_dict[f'{k}_index_value_time_delta'] = torch.tensor(time_delta)[:, None]

                            assert example_dict[f'{k}_index_value_feats'].shape[0] == example_dict[f'{k}_index_value_time_delta'].shape[0]

                        if v.text_columns:
                            for j in group_idx_to_y_idx.keys():
                                group_text = df[df[v.groupby] == j]
                                for i in v.text_columns:

                                    column_text = [format(k) for k in list(dict.fromkeys(group_text[i].tolist())) if k is not None]

                                    if column_text:
                        
                                        example_dict.setdefault(f'{k}_{i}', []).append(f"{', '.join(column_text)}.")

                                        event_times = group_text[time_column].iloc[0]
                                        time_delta = self.compute_time_delta(event_times, reference_time, to_tensor=False)
                                        example_dict.setdefault(f'{k}_{i}_time_delta', []).append(time_delta)

        return example_dict

    def return_mimic_cxr_features(self, study_id, reference_time=None):
        example_dict = {}
        if study_id is not None:
            self.load_mimic_cxr(study_id=study_id, reference_time=reference_time)
            for k, v in self.mimic_cxr_tables.items():
                if v.load:

                    df = getattr(self, f'{k}_feats')

                    if self.debug:
                        example_dict.setdefault('mimic_cxr_inputs', []).append(k)

                    if not df.empty:

                        # The y-index for each group:
                        group_idx_to_y_idx = OrderedDict()
                        groups = df.dropna(
                            subset=v.index_columns + v.value_columns + v.text_columns + v.target_sections, 
                            axis=0, 
                            how='all'
                        )
                        groups = groups.drop_duplicates(subset=[v.groupby])[[v.groupby]]
                        groups = groups.reset_index(drop=True)
                        for i, row in groups.iterrows():
                            group_idx_to_y_idx[row[v.groupby]] = i
                        
                        if v.index_columns and group_idx_to_y_idx:

                            example_dict[f'{k}_index_value_feats'] = torch.zeros(len(group_idx_to_y_idx), v.total_indices)
                            if v.index_columns:
                                example_dict[f'{k}_index_value_feats'] = df_to_tensor_index_columns(
                                    df=df, 
                                    tensor=example_dict[f'{k}_index_value_feats'], 
                                    group_idx_to_y_idx=group_idx_to_y_idx, 
                                    groupby=v.groupby, 
                                    index_columns=v.index_columns,
                                )
                        
                            example_dict[f'{k}_index_value_token_type_ids'] = torch.full(
                                [example_dict[f'{k}_index_value_feats'].shape[0]], 
                                self.token_type_to_token_type_id[k], 
                                dtype=torch.long, 
                            )

                    if v.text_columns:
                        for j in group_idx_to_y_idx.keys():
                            group_text = df[df[v.groupby] == j]
                            for i in v.text_columns:
                                column_text = [format(k) for k in list(dict.fromkeys(group_text[i].tolist())) if k is not None]
                                if column_text:
                                    example_dict.setdefault(f'{i}', []).append(f"{', '.join(column_text)}.")

                    if v.target_sections:
                        for j in group_idx_to_y_idx.keys():
                            group_text = df[df[v.groupby] == j]
                            for i in v.target_sections:
                                column_text = [format(k) for k in list(dict.fromkeys(group_text[i].tolist())) if k is not None]
                                assert len(column_text) == 1
                                example_dict[i] = column_text[-1]

        return example_dict

    def compute_time_delta(self, event_time, reference_time, denominator = 3600, to_tensor=True):
        """
        How to we transform time-delta inputs? It appears that minutes are used as the input to 
        a weight matrix in "Self-Supervised Transformer for Sparse and Irregularly Sampled Multivariate 
        Clinical Time-Series". This is almost confirmed by the CVE class defined here:
        https://github.com/sindhura97/STraTS/blob/main/strats_notebook.ipynb, where the input has 
        a size of one.
        """
        time_delta = reference_time - event_time
        time_delta = time_delta.total_seconds() / (denominator)
        assert isinstance(time_delta, float), f'time_delta should be float, not {type(time_delta)}.'
        if time_delta < 0:
            raise ValueError(f'time_delta should be greater than or equal to zero, not {time_delta}.')
        time_delta = self.time_delta_map(time_delta)
        if to_tensor: 
            time_delta = torch.tensor(time_delta)
        return time_delta

    def filter_admissions_by_time_span(self, df, time_column):
        return df[(df[time_column] > self.start_time) & (df[time_column] <= self.end_time)]
    