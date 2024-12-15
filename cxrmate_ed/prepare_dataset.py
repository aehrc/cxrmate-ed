import json
import multiprocessing
import os
import re
import shutil
from glob import glob
from pathlib import Path

import datasets
import duckdb
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from .create_section_files import create_section_files


def mimic_cxr_image_path(dir, subject_id, study_id, dicom_id, ext='dcm'):
    return os.path.join(dir, 'p' + str(subject_id)[:2], 'p' + str(subject_id),
                        's' + str(study_id), str(dicom_id) + '.' + ext)
    
    
def format(text):
    # Remove newline, tab, repeated whitespaces, and leading and trailing whitespaces:
    def remove(text):
        text = re.sub(r'\n|\t', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    if isinstance(text, np.ndarray) or isinstance(text, list):
        return [remove(t) if not pd.isna(t) else t for t in text]
    else:
        if pd.isna(text):
            return text
        return remove(text)
    

def create_lookup_table(df, columns, start_idx):
    df = df.groupby(columns).head(1)[columns].sort_values(by=columns)
    indices = range(start_idx, start_idx + len(df))
    df['index'] = indices
    return df, indices[-1]


def lookup_tables(con, tables):
    luts_dict = {}
    for k, v in tables.items():
        luts_dict[k] = {}
        start_idx = 0
        if 'index_columns' in v:
            for i in v['index_columns']:
                lut, end_idx = create_lookup_table(con.sql(f"SELECT {i} FROM {k}").df(), [i], start_idx)
                start_idx = end_idx + 1
                luts_dict[k][i] = {str(row[i]): int(row['index']) for _, row in lut.iterrows()}
        if 'value_columns' in v:
            for i in v['value_columns']:
                luts_dict[k][i] = start_idx
                start_idx += 1
                
        luts_dict[k]['total'] = start_idx

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lookup_tables.json'), 'w') as file:
        json.dump(luts_dict, file)


def prepare_dataset(physionet_dir, database_dir, num_workers=None):
    
    num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count() // 4
    
    Path(database_dir).mkdir(parents=True, exist_ok=True)

    sectioned_dir = os.path.join(database_dir, 'mimic_cxr_sectioned')
    mimic_cxr_sectioned_path = os.path.join(sectioned_dir, 'mimic_cxr_sectioned.csv')
    if not os.path.exists(mimic_cxr_sectioned_path):
        print(f'{mimic_cxr_sectioned_path} does not exist, creating...')
        
        # Check if reports exist. Reports for the first and last patients are checked only for speed, this comprimises comprehensiveness for speed:
        report_paths = [
            os.path.join(physionet_dir, 'mimic-cxr/2.0.0/files/p10/p10000032/s50414267.txt'),
            os.path.join(physionet_dir, 'mimic-cxr/2.0.0/files/p10/p10000032/s53189527.txt'),
            os.path.join(physionet_dir, 'mimic-cxr/2.0.0/files/p10/p10000032/s53911762.txt'),
            os.path.join(physionet_dir, 'mimic-cxr/2.0.0/files/p10/p10000032/s56699142.txt'),
            os.path.join(physionet_dir, 'mimic-cxr/2.0.0/files/p19/p19999987/s55368167.txt'),
            os.path.join(physionet_dir, 'mimic-cxr/2.0.0/files/p19/p19999987/s58621812.txt'),
            os.path.join(physionet_dir, 'mimic-cxr/2.0.0/files/p19/p19999987/s58971208.txt'),
        ]
        assert all([os.path.isfile(i) for i in report_paths]), f"""The reports do not exist with the following regex: {os.path.join(physionet_dir, 'mimic-cxr/2.0.0/files/p1*/p1*/s*.txt')}.
        "Please download them using wget -r -N -c -np --reject dcm --user <username> --ask-password https://physionet.org/files/mimic-cxr/2.0.0/"""

        print('Extracting sections from reports...')        
        create_section_files(
            reports_path=os.path.join(physionet_dir, 'mimic-cxr', '2.0.0', 'files'),
            output_path=sectioned_dir,
            no_split=True,
        )
                
    csv_paths = []         
    csv_paths.append(glob(os.path.join(physionet_dir, 'mimic-iv-ed', '*', 'ed', 'edstays.csv.gz'))[0])
    csv_paths.append(glob(os.path.join(physionet_dir, 'mimic-iv-ed', '*', 'ed', 'medrecon.csv.gz'))[0])
    csv_paths.append(glob(os.path.join(physionet_dir, 'mimic-iv-ed', '*', 'ed', 'pyxis.csv.gz'))[0])
    csv_paths.append(glob(os.path.join(physionet_dir, 'mimic-iv-ed', '*', 'ed', 'triage.csv.gz'))[0])
    csv_paths.append(glob(os.path.join(physionet_dir, 'mimic-iv-ed', '*', 'ed', 'vitalsign.csv.gz'))[0])
    
    base_names = [os.path.basename(i) for i in csv_paths]

    for i in ['edstays.csv.gz', 'medrecon.csv.gz', 'pyxis.csv.gz', 'triage.csv.gz', 'vitalsign.csv.gz']:
        assert i in base_names, f"""Table {i} is missing from MIMIC-IV-ED.
            Please download the tables from https://physionet.org/content/mimic-iv-ed. Do not decompress them."""
                
    csv_paths.append(glob(os.path.join(physionet_dir, 'mimic-cxr-jpg', '*', 'mimic-cxr-2.0.0-metadata.csv.gz'))[0])
    csv_paths.append(glob(os.path.join(physionet_dir, 'mimic-cxr-jpg', '*', 'mimic-cxr-2.0.0-chexpert.csv.gz'))[0])
    csv_paths.append(glob(os.path.join(physionet_dir, 'mimic-cxr-jpg', '*', 'mimic-cxr-2.0.0-split.csv.gz'))[0])

    base_names = [os.path.basename(i) for i in csv_paths[-3:]]

    for i in ['mimic-cxr-2.0.0-metadata.csv.gz', 'mimic-cxr-2.0.0-chexpert.csv.gz', 'mimic-cxr-2.0.0-split.csv.gz']:
        assert i in base_names, f"""CSV file {i} is missing from MIMIC-CXR-JPG.
            Please download the tables from https://physionet.org/content/mimic-cxr-jpg. Do not decompress them."""
            
    con = duckdb.connect(':memory:')
    for i in csv_paths:
        name = Path(i).stem.replace('.csv', '').replace('.gz', '').replace('-', '_').replace('.', '_')
        print(f'Copying {name} into database...')  
        con.sql(f"CREATE OR REPLACE TABLE {name} AS FROM '{i}';")         

    # DuckDB has trouble reading the sectioned .csv file, read with pandas instead:
    sections = pd.read_csv(mimic_cxr_sectioned_path)

    # Remove the first character from the study column and rename it to study_id:
    con.sql(
        """
        CREATE OR REPLACE TABLE mimic_cxr_sectioned AS 
        SELECT *, CAST(SUBSTR(study, 2) AS INT32) AS study_id 
        FROM sections;
        """
    )

    # Combine StudyDate and StudyTime into a single column and create the studies table:
    con.sql(
        """
        CREATE OR REPLACE TABLE studies AS 
        SELECT *, 
            strptime(
                CAST(StudyDate AS VARCHAR) || ' ' || lpad(split_part(CAST(StudyTime AS VARCHAR), '.', 1), 6, '0'), 
                '%Y%m%d %H%M%S'
            ) AS study_datetime
        FROM mimic_cxr_2_0_0_metadata;
        """
    )
    
    # Load the table configuration:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables.json')
    path = path if os.path.exists(path) else hf_hub_download(repo_id='aehrc/cxrmate-ed', filename='tables.json')
    with open(path, 'r') as f:
        tables = json.load(f)
            
    # Create lookup tables:
    lookup_tables(con, tables)
        
    # Collapse to one row per study, aggregate each studies columns as a list: 
    con.sql(
        """
        CREATE OR REPLACE TABLE studies AS
        SELECT 
            LIST(dicom_id) AS dicom_id, 
            FIRST(subject_id) AS subject_id, 
            study_id,    
            LIST(PerformedProcedureStepDescription) AS PerformedProcedureStepDescription, 
            LIST(ViewPosition) AS ViewPosition, 
            LIST(Rows) AS Rows, 
            LIST(Columns) AS Columns,
            LIST(StudyDate) AS StudyDate, 
            LIST(StudyTime) AS StudyTime, 
            LIST(ProcedureCodeSequence_CodeMeaning) AS ProcedureCodeSequence_CodeMeaning,
            LIST(ViewCodeSequence_CodeMeaning) AS ViewCodeSequence_CodeMeaning,
            LIST(PatientOrientationCodeSequence_CodeMeaning) AS PatientOrientationCodeSequence_CodeMeaning,
            LIST(study_datetime) AS study_datetime,
            MAX(study_datetime) AS latest_study_datetime,
        FROM studies
        GROUP BY study_id;
        """
    )

    # Join and filter the studies that overlap with ED stays:
    con.sql(
        """
        CREATE OR REPLACE TABLE studies AS
        SELECT 
            s.*, 
            e.hadm_id, 
            e.stay_id, 
            e.intime,
            e.outtime,
        FROM studies s
        LEFT JOIN edstays e
        ON s.subject_id = e.subject_id
        AND e.intime < s.latest_study_datetime 
        AND e.outtime > s.latest_study_datetime
        AND s.study_id != 59128861;
        """
    )  # Don't join study 59128861 as it overlaps with two ED stays
    
    
    # Aggregate and add the edstays table:
    con.sql(
        """
        CREATE OR REPLACE TABLE edstays_aggregated AS
        SELECT 
            FIRST(subject_id) AS subject_id, 
            stay_id,    
            LIST(intime) AS intime, 
            LIST(outtime) AS outtime, 
            LIST(gender) AS gender, 
            LIST(race) AS race, 
            LIST(arrival_transport) AS arrival_transport,
            LIST(disposition) AS disposition,
        FROM edstays
        GROUP BY stay_id;
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TABLE studies AS
        SELECT 
            s.*, 
            e.intime AS edstays_intime,
            e.outtime AS edstays_outtime,
            e.gender AS edstays_gender,
            e.race AS edstays_race,
            e.arrival_transport AS edstays_arrival_transport,
            e.disposition AS edstays_disposition,
        FROM studies s
        LEFT JOIN edstays_aggregated e
        ON s.stay_id = e.stay_id;
        """
    )

    # Aggregate and add the triage table:
    con.sql(
        """
        CREATE OR REPLACE TABLE triage_aggregated AS
        SELECT 
            FIRST(subject_id) AS subject_id, 
            stay_id,    
            LIST(temperature) as temperature, 
            LIST(heartrate) AS heartrate, 
            LIST(resprate) AS resprate,
            LIST(o2sat) AS o2sat,
            LIST(sbp) AS sbp,
            LIST(dbp) AS dbp,
            LIST(pain) AS pain,
            LIST(acuity) AS acuity,
            LIST(chiefcomplaint) AS chiefcomplaint,
        FROM triage
        GROUP BY stay_id;
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TABLE studies AS
        SELECT 
            s.*, 
            t.temperature AS triage_temperature,
            t.heartrate AS triage_heartrate,
            t.resprate AS triage_resprate,
            t.o2sat AS triage_o2sat,
            t.sbp AS triage_sbp,
            t.dbp AS triage_dbp,
            t.pain AS triage_pain,
            t.acuity AS triage_acuity,
            t.chiefcomplaint AS triage_chiefcomplaint,
        FROM studies s
        LEFT JOIN triage_aggregated t
        ON s.stay_id = t.stay_id;
        """
    )
    
    # Aggregate and then add the vitalsign table (ensuring no rows with a charttime after the latest study_datetime):
    con.sql(
        """
        CREATE OR REPLACE TABLE vitalsign_causal AS
        SELECT v.*, s.latest_study_datetime, s.study_id,
        FROM vitalsign v
        JOIN studies s ON v.stay_id = s.stay_id
        WHERE v.charttime < s.latest_study_datetime;
        """
    )  # This duplicates the rows for stay_ids that cover multiple study_ids. Hence, the following joins must be on study_id, not stay_id.
    con.sql(
        """
        CREATE OR REPLACE TABLE vitalsign_aggregated AS
        SELECT 
            study_id,
            FIRST(subject_id) AS subject_id, 
            FIRST(stay_id) as stay_id,    
            LIST(charttime) AS charttime, 
            LIST(temperature) as temperature, 
            LIST(heartrate) AS heartrate, 
            LIST(resprate) AS resprate,
            LIST(o2sat) AS o2sat,
            LIST(sbp) AS sbp,
            LIST(dbp) AS dbp,
            LIST(rhythm) AS rhythm,
            LIST(pain) AS pain,
        FROM vitalsign_causal
        GROUP BY study_id;
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TABLE studies AS
        SELECT 
            s.*, 
            v.charttime AS vitalsign_charttime,
            v.temperature AS vitalsign_temperature,
            v.heartrate AS vitalsign_heartrate,
            v.resprate AS vitalsign_resprate,
            v.o2sat AS vitalsign_o2sat,
            v.sbp AS vitalsign_sbp,
            v.dbp AS vitalsign_dbp,
            v.rhythm AS vitalsign_rhythm,
            v.pain AS vitalsign_pain,
        FROM studies s
        LEFT JOIN vitalsign_aggregated v
        ON s.study_id = v.study_id;
        """
    )
    
    # Aggregate and then add the medrecon table:
    con.sql(
        """
        CREATE OR REPLACE TABLE medrecon_aggregated AS
        SELECT 
            FIRST(subject_id) AS subject_id, 
            stay_id,    
            LIST(charttime) AS charttime, 
            LIST(name) as name, 
            LIST(gsn) AS gsn, 
            LIST(ndc) AS ndc,
            LIST(etc_rn) AS etc_rn,
            LIST(etccode) AS etccode,
            LIST(etcdescription) AS etcdescription,
        FROM medrecon
        GROUP BY stay_id;
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TABLE studies AS
        SELECT 
            s.*, 
            m.charttime AS medrecon_charttime,
            m.name AS medrecon_name,
            m.gsn AS medrecon_gsn,
            m.ndc AS medrecon_ndc,
            m.etc_rn AS medrecon_etc_rn,
            m.etccode AS medrecon_etccode,
            m.etcdescription AS medrecon_etcdescription,
        FROM studies s
        LEFT JOIN medrecon_aggregated m
        ON s.stay_id = m.stay_id;
        """
    )
    
    # Aggregate and then add the pyxis table (ensuring no rows with a charttime after the latest study_datetime):
    con.sql(
        """
        CREATE OR REPLACE TABLE pyxis_causal AS
        SELECT p.*, s.latest_study_datetime, s.study_id,
        FROM pyxis p
        JOIN studies s ON p.stay_id = s.stay_id
        WHERE p.charttime < s.latest_study_datetime;
        """
    ) # This duplicates the rows for stay_ids that cover multiple study_ids. Hence, the following joins must be on study_id, not stay_id.
    con.sql(
        """
        CREATE OR REPLACE TABLE pyxis_aggregated AS
        SELECT 
            study_id,
            FIRST(subject_id) AS subject_id, 
            FIRST(stay_id) as stay_id,    
            LIST(charttime) AS charttime, 
            LIST(med_rn) as med_rn, 
            LIST(name) as name, 
            LIST(gsn_rn) AS gsn_rn, 
            LIST(gsn) AS gsn, 
        FROM pyxis_causal
        GROUP BY study_id;
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TABLE studies AS
        SELECT 
            s.*, 
            p.charttime AS pyxis_charttime,
            p.med_rn AS pyxis_med_rn,
            p.name AS pyxis_name,
            p.gsn_rn AS pyxis_gsn_rn,
            p.gsn AS pyxis_gsn,
        FROM studies s
        LEFT JOIN pyxis_aggregated p
        ON s.study_id = p.study_id;
        """
    )
      
    # Add the reports:
    con.sql(
        """
        CREATE OR REPLACE TABLE studies AS
        SELECT s.*, r.findings, r.impression, r.indication, r.history, r.comparison, r.last_paragraph, r.technique,
        FROM studies s
        LEFT JOIN mimic_cxr_sectioned r
        ON s.study_id = r.study_id
        """
    )
      
    # Aggregate and then add the splits:
    con.sql(
        """
        CREATE OR REPLACE TABLE split_aggregated AS
        SELECT 
            study_id,    
            FIRST(split) AS split,  
        FROM mimic_cxr_2_0_0_split
        GROUP BY study_id;
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TABLE studies AS
        SELECT s.*, x.split,
        FROM studies s
        JOIN split_aggregated x
        ON s.study_id = x.study_id;
        """
    )
    
    # Prior studies column:
    con.sql(
        """
        CREATE OR REPLACE TABLE prior_studies AS
        WITH sorted AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY latest_study_datetime) AS rn
            FROM studies
        ),
        aggregated AS (
            SELECT subject_id,
                study_id,
                latest_study_datetime,
                ARRAY_AGG(study_id) OVER (PARTITION BY subject_id ORDER BY rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_study_ids,
                ARRAY_AGG(latest_study_datetime) OVER (PARTITION BY subject_id ORDER BY rn ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_study_datetimes
            FROM sorted
        )
        SELECT * 
        FROM aggregated;
        """
    )
    con.sql(
        """
        CREATE OR REPLACE TABLE studies AS
        SELECT s.*, p.prior_study_ids, p.prior_study_datetimes,
        FROM studies s
        LEFT JOIN prior_studies p
        ON s.study_id = p.study_id
        ORDER BY s.subject_id, s.study_datetime DESC;
        """
    )
    
    # Text columns:
    text_columns = [f'{k}_{j}' if k != 'mimic_cxr_sectioned' else j for k, v in tables.items() if 'text_columns' in v for j in (v['text_columns'] if isinstance(v['text_columns'], list) else [v['text_columns']])] + ['findings', 'impression']
        
    pattern = os.path.join(physionet_dir, 'mimic-cxr-jpg', '*', 'files')
    mimic_cxr_jpg_dir = glob(pattern)
    assert len(mimic_cxr_jpg_dir), f'Multiple directories matched the pattern {pattern}: {mimic_cxr_jpg_dir}. Only one is required.'
    mimic_cxr_jpg_dir = mimic_cxr_jpg_dir[0]
    
    def load_image(row):
        images = []
        for dicom_ids, study_id, subject_id in zip(row['dicom_id'], row['study_id'], row['subject_id']):
            study_images = []
            for dicom_id in dicom_ids: 
                image_path = mimic_cxr_image_path(mimic_cxr_jpg_dir, subject_id, study_id, dicom_id, 'jpg')
                with open(image_path, 'rb') as f:
                    image = f.read()
                study_images.append(image)
            images.append(study_images)
        row['images'] = images
        return row
    
    dataset_dict = {}
    for split in ['test', 'validate', 'train']:
        df = con.sql(f"FROM studies WHERE split = '{split}'").df()
        
        # Format text columns:
        for i in text_columns:
            df[i] = df[i].apply(format)
            
        # Save indices for each split:        
        df[df['findings'].notna() & df['impression'].notna()]['study_id'].to_json(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), f'mimic_cxr_jpg_{split}_study_ids.json'),
            orient='records',
            lines=False,
        )
        df_stay_id = df[df['findings'].notna() & df['impression'].notna() & df['stay_id'].notna()][['study_id', 'stay_id']]
        df_stay_id['stay_id'] = df_stay_id['stay_id'].astype(int)
        df_stay_id['study_id'].to_json(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), f'mimic_iv_ed_mimic_cxr_jpg_{split}_study_ids.json'),
            orient='records',
            lines=False,
        )
        
        if split == 'test':
            pyxis_columns = [col for col in df.columns if col.startswith('pyxis_')]
            df_pyxis = df[df['findings'].notna() & df['impression'].notna() & df['stay_id'].notna()]
            df_pyxis = df_pyxis[~df_pyxis[pyxis_columns].isna().all(axis=1)]    
            df_pyxis['study_id'].to_json(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), f'mimic_iv_ed_mimic_cxr_jpg_pyxis_{split}_study_ids.json'),
                orient='records',
                lines=False,
            )
            
            vitalsign_columns = [col for col in df.columns if col.startswith('vitalsign_')]
            df_vitalsign = df[df['findings'].notna() & df['impression'].notna() & df['stay_id'].notna()]
            df_vitalsign = df_vitalsign[~df_vitalsign[vitalsign_columns].isna().all(axis=1)]    
            df_vitalsign['study_id'].to_json(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), f'mimic_iv_ed_mimic_cxr_jpg_vitalsign_{split}_study_ids.json'),
                orient='records',
                lines=False,
            )
        
        dataset_dict[split] = datasets.Dataset.from_pandas(df)
        cache_dir = os.path.join(database_dir, '.cache')
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        dataset_dict[split] = dataset_dict[split].map(
            load_image,
            num_proc=num_workers,
            writer_batch_size=8,
            batched=True,
            batch_size=8,
            keep_in_memory=False,
            cache_file_name=os.path.join(cache_dir, f'.{split}'),
            load_from_cache_file=False,
        )
        dataset_dict[split].cleanup_cache_files()
        shutil.rmtree(cache_dir)
        
    dataset = datasets.DatasetDict(dataset_dict)
    dataset.save_to_disk(os.path.join(database_dir, 'mimic_iv_ed_mimic_cxr_jpg_dataset'))
    
    con.close()
    

if __name__ == "__main__":
    physionet_dir = '/datasets/work/hb-mlaifsp-mm/work/archive/physionet.org/files'  # Where MIMIC-CXR, MIMIC-CXR-JPG, and MIMIC-IV-ED are stored.
    database_dir = '/scratch3/nic261/database/cxrmate_ed'  # Where the resultant database will be stored.

    prepare_dataset(physionet_dir=physionet_dir, database_dir=database_dir)
