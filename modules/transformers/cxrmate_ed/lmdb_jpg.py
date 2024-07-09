import multiprocessing

import duckdb
import lmdb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .dataset import mimic_cxr_image_path


class JPGDataset(Dataset):
    def __init__(self, df, jpg_path):
        self.df = df
        self.jpg_path = jpg_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        
        jpg_path = mimic_cxr_image_path(self.jpg_path, row['subject_id'], row['study_id'], row['dicom_id'], 'jpg')
       
        # Convert key to bytes:
        key = bytes(row['dicom_id'], 'utf-8')

        # Read the .jpg file as bytes:
        with open(jpg_path, 'rb') as f:
            image = f.read()

        return {
            'keys': key,
            'images': image,
        }

def prepare_mimic_cxr_jpg_lmdb(mimic_iv_duckdb_path, mimic_cxr_jpg_path, mimic_cxr_jpg_lmdb_path, map_size_tb, num_workers=None):
    
    num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count()

    connect = duckdb.connect(mimic_iv_duckdb_path, read_only=True)
    df = connect.sql("SELECT DISTINCT ON(dicom_id) subject_id, study_id, dicom_id FROM mimic_cxr").df()
    connect.close()

    # Map size:
    map_size = int(map_size_tb * (1024 ** 4))
    assert isinstance(map_size, int)

    print(f'Map size: {map_size}')
    
    dataset = JPGDataset(df, mimic_cxr_jpg_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=num_workers, 
        shuffle=False, 
        num_workers=num_workers, 
        prefetch_factor=1, 
        collate_fn=lambda x: x,
    )
    
    env = lmdb.open(mimic_cxr_jpg_lmdb_path, map_size=map_size, readonly=False)
    for batch in tqdm(dataloader):
        for i in batch:
            with env.begin(write=True) as txn:
                value = txn.get(b'image_keys')
                if value is None:
                    txn.put(i['keys'], i['images'])
            env.sync()
    env.close()
