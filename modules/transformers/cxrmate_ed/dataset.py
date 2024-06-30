import os
import struct

import lmdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image, read_image

from data.mimic_cxr.dcm_processing import load_and_preprocess_dcm_uint16
from tools.mimic_iv.ed_cxr.records import EDCXRSubjectRecords
from tools.utils import mimic_cxr_image_path

# Ordered by oblique, lateral, AP, and then PA views so that PA views are closest in position to the generated tokens (and oblique is furtherest).
VIEW_ORDER = ['LPO', 'RAO', 'LAO', 'SWIMMERS', 'XTABLE LATERAL', 'LL', 'LATERAL',  'AP AXIAL', 'AP RLD', 'AP LLD', 'AP', 'PA RLD', 'PA LLD', 'PA']


class StudyIDEDStayIDSubset(Dataset):
    """
    Study ID & ED stay ID subset. Examples are indexed by the study identifier.
    Information from the ED module is added by finding the study_id that is within 
    the timespan of the stay_id for the subject_id. The history and indication 
    sections are also included.
    """
    def __init__(
        self, 
        mimic_iv_duckdb_path, 
        split, 
        dataset_dir=None, 
        max_images_per_study=None,
        transforms=None, 
        images=True,
        columns='study_id, dicom_id, subject_id, findings, impression',
        and_condition='',
        records=None,
        study_id_inclusion_list=None,
        return_images=True,
        ed_module=True,
        extension='jpg',
        images_rocksdb_path=None,
        jpg_lmdb_path=None,
        jpg_rocksdb_path=None,
    ):
        """
        Argument/s:
            mimic_iv_duckdb_path - Path to MIMIC-IV DuckDB database.
            split - 'train', 'validate', or 'test'.
            dataset_dir - Dataset directory.
            max_images_per_study - the maximum number of images per study.
            transforms - torchvision transformations.
            colour_space - PIL target colour space.
            images - flag to return processed images.
            columns - which columns to query on.
            and_condition - AND condition to add to the SQL query.
            records - MIMIC-IV records class instance.
            study_id_inclusion_list - studies not in this list are excluded.
            return_images - return CXR images for the study as tensors.
            ed_module - use the ED module.
            extension - 'jpg' or 'dcm'.
            images_rocksdb_path - path to image RocksDB database.
            jpg_lmdb_path - path to LMDB .jpg database.
            jpg_rocksdb_path - path to RocksDB .jpg database. 
        """
        super(StudyIDEDStayIDSubset, self).__init__()
        self.split = split
        self.dataset_dir = dataset_dir
        self.max_images_per_study = max_images_per_study
        self.transforms = transforms
        self.images = images
        self.columns = columns
        self.and_condition = and_condition
        self.return_images = return_images
        self.ed_module = ed_module
        self.extension = extension
        self.images_rocksdb_path = images_rocksdb_path
        self.jpg_lmdb_path = jpg_lmdb_path
        self.jpg_rocksdb_path = jpg_rocksdb_path
        
        # If max images per study is not set:
        self.max_images_per_study = float('inf') if self.max_images_per_study is None else self.max_images_per_study

        assert self.extension == 'jpg' or self.extension == 'dcm'

        if self.dataset_dir is not None and self.images_rocksdb_path is None:
            if self.extension == 'jpg':
                if 'physionet.org/files/mimic-cxr-jpg/2.0.0/files' not in self.dataset_dir:
                    self.dataset_dir = os.path.join(self.dataset_dir, 'physionet.org/files/mimic-cxr-jpg/2.0.0/files')
            elif self.extension == 'dcm':
                if 'physionet.org/files/mimic-cxr/2.0.0/files' not in self.dataset_dir:
                    self.dataset_dir = os.path.join(self.dataset_dir, 'physionet.org/files/mimic-cxr/2.0.0/files')

        # Open the RocksDB images database:
        if self.images_rocksdb_path is not None:
            import rocksdb

            # Define the column families:
            column_families = {
                b'shape': rocksdb.ColumnFamilyOptions(),
                b'image': rocksdb.ColumnFamilyOptions(),
            }
            
            opts = rocksdb.Options()
            opts.max_open_files = 1e+5
            self.images_db = rocksdb.DB(self.images_rocksdb_path, opts, column_families=column_families, read_only=True)
            
            self.shape_handle = self.images_db.get_column_family(b'shape')
            self.image_handle = self.images_db.get_column_family(b'image')
            
            self.shape_dtype = np.int32
            self.image_dtype = np.uint16

        # Prepare the RocksDB .jpg database:
        if self.jpg_rocksdb_path is not None:
            import rocksdb
            
            opts = rocksdb.Options()
            opts.max_open_files = 1e+5
                    
            self.images_db = rocksdb.DB(self.jpg_rocksdb_path, opts, read_only=True)
            
        # Prepare the LMDB .jpg database:
        if self.jpg_lmdb_path is not None:
            
            print('Loading images using LMDB.')

            # Map size:
            map_size = int(0.65 * (1024 ** 4))
            assert isinstance(map_size, int)
            
            self.env = lmdb.open(self.jpg_lmdb_path, map_size=map_size, lock=False, readonly=True)
            self.txn = self.env.begin(write=False)

        self.records = EDCXRSubjectRecords(database_path=mimic_iv_duckdb_path) if records is None else records

        query = f"""
        SELECT {columns}
        FROM mimic_cxr 
        WHERE split = '{split}' 
        {and_condition}
        ORDER BY study_id
        """

        # For multi-image, the study identifiers make up the training examples:
        df = self.records.connect.sql(query).df()

        # Drop studies that don't have a findings or impression section:
        df = df.dropna(subset=['findings', 'impression'], how='any')

        # This study has two rows in edstays (removed as it causes issues):
        if self.ed_module:
            df = df[df['study_id'] != 59128861]

        # Exclude studies not in list:
        if study_id_inclusion_list is not None:
            df = df[df['study_id'].isin(study_id_inclusion_list)]

        # Example study identifiers for the subset:
        self.examples = df['study_id'].unique().tolist()

        # Record statistics:
        self.num_study_ids = len(self.examples)
        self.num_dicom_ids = len(df['dicom_id'].unique().tolist())
        self.num_subject_ids = len(df['subject_id'].unique().tolist())

    def __len__(self):
        return self.num_study_ids

    def __getitem__(self, index):

        study_id = self.examples[index]

        # Get the study:
        study = self.records.connect.sql(
            f"""
            SELECT dicom_id, study_id, subject_id, study_datetime, ViewPosition
            FROM mimic_cxr 
            WHERE (study_id = {study_id});
            """
        ).df()
        subject_id = study.iloc[0, study.columns.get_loc('subject_id')]
        study_id = study.iloc[0, study.columns.get_loc('study_id')]
        study_datetime = study['study_datetime'].max()

        example_dict = {
            'study_ids': study_id,
            'subject_id': subject_id,
            'index': index,
        }

        example_dict.update(self.records.return_mimic_cxr_features(study_id))

        if self.ed_module:
            edstays = self.records.connect.sql(
                f"""
                SELECT stay_id, intime, outtime
                FROM edstays 
                WHERE (subject_id = {subject_id})
                AND intime < '{study_datetime}'
                AND outtime > '{study_datetime}';
                """
            ).df()

            assert len(edstays) <= 1
            stay_id = edstays.iloc[0, edstays.columns.get_loc('stay_id')] if not edstays.empty else None
            self.records.clear_start_end_times()
            example_dict.update(self.records.return_ed_module_features(stay_id, study_datetime))

            example_dict['stay_ids'] = stay_id

        if self.return_images:
            example_dict['images'], example_dict['image_time_deltas'] = self.get_images(study, study_datetime)

        return example_dict

    def get_images(self, example, reference_time):
        """
        Get the image/s for a given example. 

        Argument/s:
            example - dataframe for the example.
            reference_time - reference_time for time delta.

        Returns:
            The image/s for the example
        """

        # Sample if over max_images_per_study. Only allowed during training:
        if len(example) > self.max_images_per_study:
            assert self.split == 'train'
            example = example.sample(n=self.max_images_per_study, axis=0)

        # Order by ViewPostion:
        example['ViewPosition'] = example['ViewPosition'].astype(pd.CategoricalDtype(categories=VIEW_ORDER, ordered=True))
        
        # Sort the DataFrame based on the categorical column
        example = example.sort_values(by=['study_datetime', 'ViewPosition'])

        # Load and pre-process each CXR:
        images, time_deltas = [], []
        for _, row in example.iterrows():
            images.append(
                self.load_and_preprocess_image(
                    row['subject_id'], 
                    row['study_id'], 
                    row['dicom_id'], 
                ),
            )
            time_deltas.append(self.records.compute_time_delta(row['study_datetime'], reference_time, to_tensor=False))
                
        if self.transforms is not None:
            images = torch.stack(images, 0)
        return images, time_deltas

    def load_and_preprocess_image(self, subject_id, study_id, dicom_id):
        """
        Load and preprocess an image using torchvision.transforms.v2:
            https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py

        Argument/s:
            subject_id - subject identifier.
            study_id - study identifier.
            dicom_id - DICOM identifier.

        Returns:
            image - Tensor of the CXR.
        """

        if self.extension == 'jpg':

            if self.jpg_rocksdb_path is not None:
                
                # Convert to bytes:
                key = bytes(dicom_id, 'utf-8')

                # Retrieve image:
                image = bytearray(self.images_db.get(key))
                image = torch.frombuffer(image, dtype=torch.uint8)
                image = decode_image(image)            

            elif self.jpg_lmdb_path is not None:
                
                # Convert to bytes:
                key = bytes(dicom_id, 'utf-8')

                # Retrieve image:
                image = bytearray(self.txn.get(key))
                image = torch.frombuffer(image, dtype=torch.uint8)
                image = decode_image(image)
                
            else:
                image_file_path = mimic_cxr_image_path(self.dataset_dir, subject_id, study_id, dicom_id, self.extension)
                image = read_image(image_file_path)

        elif self.extension == 'dcm':
            if self.images_rocksdb_path is not None:
                
                key = dicom_id.encode('utf-8')
                
                # Retrieve the serialized image shape associated with the key:
                shape_bytes = self.images_db.get((self.shape_handle, key), key)
                shape = struct.unpack('iii', shape_bytes)

                np.frombuffer(shape_bytes, dtype=self.shape_dtype).reshape(3)

                # Retrieve the serialized image data associated with the key:
                image_bytes = self.images_db.get((self.image_handle, key), key)
                image = np.frombuffer(image_bytes, dtype=self.image_dtype).reshape(*shape)

            else:
                image_file_path = mimic_cxr_image_path(self.dataset_dir, subject_id, study_id, dicom_id, self.extension)
                image = load_and_preprocess_dcm_uint16(image_file_path)
                
            # Convert to a torch tensor:
            image = torch.from_numpy(image)

        if self.transforms is not None:
            image = self.transforms(image)

        return image


if __name__ == '__main__':
    import time

    from tqdm import tqdm

    num_samples = 20
    
    datasets = []
    datasets.append(
        StudyIDEDStayIDSubset(
            dataset_dir='/datasets/work/hb-mlaifsp-mm/work/archive',
            mimic_iv_duckdb_path='/scratch3/nic261/database/mimic_iv_duckdb_rev_b.db',
            split='train',
            extension='jpg',
            ed_module=False,
        ),
    )
    
    datasets.append(
        StudyIDEDStayIDSubset(
            dataset_dir='/scratch3/nic261/datasets',
            mimic_iv_duckdb_path='/scratch3/nic261/database/mimic_iv_duckdb_rev_b.db',
            split='train',
            extension='jpg',
            ed_module=False,
        ),
    )
    
    datasets.append(
        StudyIDEDStayIDSubset(
            jpg_lmdb_path='/scratch3/nic261/database/mimic_cxr_jpg_lmdb_rev_a.db',
            mimic_iv_duckdb_path='/scratch3/nic261/database/mimic_iv_duckdb_rev_b.db',
            split='train',
            extension='jpg',
            ed_module=False,
        ),
    )
    
    datasets.append(
        StudyIDEDStayIDSubset(
            jpg_rocksdb_path='/scratch3/nic261/database/mimic_cxr_jpg_rocksdb.db',
            mimic_iv_duckdb_path='/scratch3/nic261/database/mimic_iv_duckdb_rev_b.db',
            split='train',
            extension='jpg',
            ed_module=False,
        )
    )
    
    assert (datasets[1][0]['images'][0] == datasets[2][0]['images'][0]).all().item()
    assert (datasets[1][5]['images'][0] == datasets[2][5]['images'][0]).all().item()

    for d in datasets:
        start_time = time.time()
        indices = torch.randperm(len(d))[:num_samples]  # Get random indices.
        for i in tqdm(indices):
            _ = d[i]
        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        