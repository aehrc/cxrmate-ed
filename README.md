# CXRMate-ED: The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It

This is the model and data pipeline for the CXRMate-ED model from: https://arxiv.org/pdf/2406.13181.

The abstract from the paper:

"This study investigates the integration of diverse patient data sources into multimodal language models for automated chest X-ray (CXR) report generation. Traditionally, CXR report generation relies solely on CXR images and limited radiology data, overlooking valuable information from patient health records, particularly from emergency departments. Utilising the MIMIC-CXR and MIMIC-IV-ED datasets, we incorporate detailed patient information such as aperiodic vital signs, medications, and clinical history to enhance diagnostic accuracy. We introduce a novel approach to transform these heterogeneous data sources into embeddings that prompt a multimodal language model, significantly enhancing the diagnostic accuracy of generated radiology reports. Our comprehensive evaluation demonstrates the benefits of using a broader set of patient data, underscoring the potential for enhanced diagnostic capabilities and better patient outcomes through the integration of multimodal data in CXR report generation."

## Hugging Face Hub
The model and data pipeline are available on Hugging Face Hub:

https://huggingface.co/aehrc/cxrmate-ed

## MIMIC-CXR & MIMIC-IV-ED Dataset:

MIMIC-CXR, MIMIC-CXR-JPG, and MIMIC-IV-ED must be in the same Physio Net directory. E.g.:

```shell
user@cluster:~$ ls /home/user/physionet.org/files
mimic-cxr  mimic-cxr-jpg  mimic-iv-ed
```

### Download MIMIC-CXR-JPG:
Download the MIMIC-CXR-JPG dataset from https://physionet.org/content/mimic-cxr-jpg, e.g.,
```shell
wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/mimic-cxr-jpg/2.1.0/
```
Note that you must be a credentialised user to access this dataset.

### Download the reports from MIMIC-CXR:
MIMIC-CXR-JPG does not include the radiology reports and are instead included with MIMIC-CXR (the DICOM version of the dataset). To download this dataset and avoid downloading the DICOM files (which are very large), use `--reject dcm` with the wget command from https://physionet.org/content/mimic-cxr, e.g, 
```shell
wget -r -N -c -np --reject dcm --user <username> --ask-password https://physionet.org/files/mimic-cxr/2.0.0/
```
Note that you must be a credentialised user to access this dataset.

### Download MIMIC-IV-ED:
Download the MIMIC-IV-ED dataset from https://physionet.org/content/mimic-iv-ed, e.g.,
```shell
wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/mimic-iv-ed/2.2/
```
Note that you must be a credentialised user to access this dataset.

### Prepare the dataset:
Run the [prepare_dataset.ipynb](https://github.com/aehrc/anon/blob/main/prepare_dataset.ipynb) notebook and change the paths accordingly. It should take roughly 2-3 hours. The most time-consuming tasks are extracting sections from the radiology reports and matching CXR studies to ED stays.

Or, run the following:
```python
import transformers

# Paths:
physionet_dir = '/.../physionet.org/files'  # Where MIMIC-CXR, MIMIC-CXR-JPG, and MIMIC-IV-ED are stored.
database_dir = '/.../database/cxrmate_ed'  # The LMDB database for the JPGs and the DuckDB database for the tables will be saved here.

# Prepare the MIMIC-CXR & MIMIC-IV-ED dataset:
model = transformers.AutoModel.from_pretrained('aehrc/cxrmate-ed', trust_remote_code=True)
model.prepare_data(
    physionet_dir=physionet_dir,
    database_dir=database_dir,
)
```

#### Inference example:

```python
import torch
import transformers
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import os
import pprint
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Device and paths:
device = 'cuda'
physionet_dir = '/datasets/work/hb-mlaifsp-mm/work/archive/physionet.org/files'  # Where MIMIC-CXR, MIMIC-CXR-JPG, and MIMIC-IV-ED are stored.
database_dir = '/scratch3/nic261/database/cxrmate_ed'  # The LMDB database for the JPGs and the DuckDB database for the tables will be saved here.

# Download model checkpoint:
ckpt_name = '...'  # Anonymised for now.
model = transformers.AutoModel.from_pretrained(ckpt_name, trust_remote_code=True).to(device=device)
model.eval()

# Download tokenizer:
tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(ckpt_name)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Image transforms:
image_size = 384
test_transforms = v2.Compose(
    [
        v2.Grayscale(num_output_channels=3),
        v2.Resize(
            size=image_size, 
            antialias=True,
            interpolation=v2.InterpolationMode.BICUBIC,
        ),
        v2.CenterCrop(size=[image_size, image_size]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ]
)

# Prepare the MIMIC-CXR & MIMIC-IV-ED dataset:
model.prepare_data(
    physionet_dir=physionet_dir,
    database_dir=database_dir,
)

# Get the test set dataset & dataloader:
test_set = model.get_dataset(split='test', transforms=test_transforms, database_dir=database_dir)
test_dataloader = DataLoader(
    test_set,
    batch_size=1, 
    num_workers=5,
    shuffle=True,
    collate_fn=model.collate_fn,
    pin_memory=True,
)

# Get an example:
batch = next(iter(test_dataloader))

# Move tensors in the batch to the device:
for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        batch[key] = value.to(device)

# Convert the patient data in the batch into embeddings:
inputs_embeds, attention_mask, token_type_ids, position_ids, bos_token_ids = model.prepare_inputs(tokenizer=tokenizer, **batch)
    
# Generate reports:
output_ids = model.generate(
    input_ids=bos_token_ids,
    decoder_inputs_embeds=inputs_embeds,
    decoder_token_type_ids=token_type_ids,
    prompt_attention_mask=attention_mask,
    prompt_position_ids=position_ids,
    special_token_ids=[tokenizer.sep_token_id],
    token_type_id_sections=model.decoder.config.section_ids,
    max_length=256,
    num_beams=4,
    return_dict_in_generate=True,
)['sequences']

# Findings and impression section:
findings, impression = model.split_and_decode_sections(output_ids, [tokenizer.sep_token_id, tokenizer.eos_token_id], tokenizer)
for i,j in zip(findings, impression):
    print(f'Findings:\t{i}\nImpression:\t{j}\n\n')

```

## Generated reports

Generated reports (findings and impression sections) for the test set are provided in [`test_set_generated_reports`](https://github.com/aehrc/anon/blob/main/test_set_generated_reports).

## Environment

The used packages can be found in `requirements.txt`.

A virtual environment can be created via:
```shell
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt
```

## Training

Training is performed using [`dlhpcstarter`](https://github.com/csiro-mlai/dl_hpc_starter_pack) and [PyTorch Lightning](https://lightning.ai).

There are three stages of training. The first two are with teacher forcing, with the last stage using reinforcement learning.

First, configure the paths at config/paths.

#### Stage 1: Training on CXRs

```shell
dlhpcstarter -t cxrmate_ed -c config/stage_1 --train --test
```

#### Stage 2: Training on CXRs + patient data embeddings

First, set the `warm_start_ckpt_path` to the checkpoint from stage 1, which can be found in the `exp_dir` (defined in `config/paths.yaml`), e.g., `exp_dir/cxrmate_ed/stage_1/trial_0/epoch=7-step=125416-val_report_chexbert_f1_macro=0.351025.ckpt`.

```shell
dlhpcstarter -t cxrmate_ed -c config/stage_2 --train --test
```

#### Stage 3: Reinforcement learning with self-critical sequence training

First, set the `warm_start_ckpt_path` to the checkpoint from stage 2, which can be found in the `exp_dir` (defined in `config/paths.yaml`), e.g., `exp_dir/cxrmate_ed/stage_2/trial_0/epoch=4-step=47750-val_report_chexbert_f1_macro=0.352807.ckpt`.

Note that four GPUs are used with [DDP](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html#distributed-data-parallel) during this stage. This can be modified in config/stage_3.yaml.

```shell
dlhpcstarter -t cxrmate_ed -c config/stage_3 --train --test
```

## To Do:

Revice metrics. The current ones are difficult to install. Only BERTScore and CXR-BERT are easy to get working currently (as they rely on HF Hub).
