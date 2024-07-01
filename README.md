# CXRMate-ED: The Impact of Auxiliary Patient Data on Automated Chest X-Ray Report Generation and How to Incorporate It

This is the model and data pipeline for the CXRMate-ED model from: https://arxiv.org/pdf/2406.13181.

The abstract from the paper:

"This study investigates the integration of diverse patient data sources into multimodal language models for automated chest X-ray (CXR) report generation. Traditionally, CXR report generation relies solely on CXR images and limited radiology data, overlooking valuable information from patient health records, particularly from emergency departments. Utilising the MIMIC-CXR and MIMIC-IV-ED datasets, we incorporate detailed patient information such as aperiodic vital signs, medications, and clinical history to enhance diagnostic accuracy. We introduce a novel approach to transform these heterogeneous data sources into embeddings that prompt a multimodal language model, significantly enhancing the diagnostic accuracy of generated radiology reports. Our comprehensive evaluation demonstrates the benefits of using a broader set of patient data, underscoring the potential for enhanced diagnostic capabilities and better patient outcomes through the integration of multimodal data in CXR report generation."

## Hugging Face Hub
The model and data pipeline are available on Hugging Face Hub:

https://huggingface.co/aehrc/cxrmate-ed

## Example

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
physionet_dir = '/.../physionet.org/files'
dataset_dir = '/.../datasets'
database_path = '/.../database/cxrmate_ed.db'
mimic_cxr_jpg_dir = '/.../physionet.org/files/mimic-cxr-jpg/2.0.0/files'

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
    dataset_dir=dataset_dir,
    database_path=database_path,
)

# Get the test set dataset & dataloader:
test_set = model.get_dataset('test', test_transforms, database_path, mimic_cxr_jpg_dir)
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
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    num_beams=4,
    return_dict_in_generate=True,
    use_cache=True,
)['sequences']

# Findings and impression section:
findings, impression = model.split_and_decode_sections(output_ids, [tokenizer.sep_token_id, tokenizer.eos_token_id], tokenizer)
for i,j in zip(findings, impression):
    print(f'Findings:\t{i}\nImpression:\t{j}\n\n')

```

## Training

Coming soon...