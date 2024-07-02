
import datetime
import math
import os

import torch
import transformers
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import v2

from modules.lightning_modules.ed_cxr.report_gen_rev_e import (
    MIMICIVEDCXRReportGen as MIMICIVEDCXRReportGenRevE,
)
from modules.transformers.cxrmate_ed.configuration_uniformer import (
    UniFormerWithProjectionHeadConfig,
)
from modules.transformers.cxrmate_ed.modelling_cxrmate_ed import (
    MIMICIVEDCXRMultimodalModel,
)
from modules.transformers.cxrmate_ed.modelling_uniformer import (
    MultiUniFormerWithProjectionHead,
)
from modules.transformers.cxrmate_ed.records import EDCXRSubjectRecords
from modules.transformers.cxrmate_ed.tables import NUM_ED_CXR_TOKEN_TYPE_IDS


class MIMICIVEDCXRReportGen(MIMICIVEDCXRReportGenRevE):
    
    def __init__(self, records=None, database_path=None, add_time_deltas=True, **kwargs):
        self.add_time_deltas = add_time_deltas
        records = EDCXRSubjectRecords(database_path=database_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1)) if records is None else records
        super().__init__(records=records, database_path=database_path, **kwargs)
    
    def init_modules(self):
        """
        Initialise torch.nn.Modules.
        """

        index_value_encoder_config = {}
        for k, v in (self.records.ed_module_tables | self.records.mimic_cxr_tables).items():
            if v.load and (v.value_columns or v.index_columns):
                index_value_encoder_config[k] = v.total_indices

        # Decoder tokenizer:
        self.tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained('aehrc/cxrmate-ed')
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Print the special tokens:
        print('Description, Special token, Index')
        for k, v in self.tokenizer.special_tokens_map.items():
            if k != 'additional_special_tokens':
                print(f'{k}, {v}, {getattr(self.tokenizer, k + "_id")}')
            else:
                for i, j in zip(self.tokenizer.additional_special_tokens, self.tokenizer.additional_special_tokens_ids):
                    print(f'additional_special_token, {i}, {j}')

        # Decoder config:
        config_decoder = transformers.LlamaConfig(
            vocab_size=len(self.tokenizer),
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=12,
            num_hidden_layers=6,
            max_position_embeddings=2048,
        )
        config_decoder.is_decoder = True
        config_decoder.index_value_encoder_config = index_value_encoder_config
        config_decoder.index_value_encoder_intermediate_size = 2048
        config_decoder.ed_module_columns = [f'{k}_{i}' for k, v in self.records.ed_module_tables.items() for i in v.text_columns]
        config_decoder.mimic_cxr_columns = [i for _, v in self.records.mimic_cxr_tables.items() for i in v.text_columns]
        config_decoder.token_type_to_token_type_id = self.records.token_type_to_token_type_id
        config_decoder.num_token_types = NUM_ED_CXR_TOKEN_TYPE_IDS
        config_decoder.include_time_delta = True
        config_decoder.time_delta_monotonic_inversion = True
        config_decoder.zero_time_delta_value = self.records.compute_time_delta(
            datetime.datetime.fromtimestamp(0),
            datetime.datetime.fromtimestamp(0), 
            to_tensor=False,
        )
        config_decoder.add_time_deltas = self.add_time_deltas

        # Section embedding identifiers (for report):
        self.section_ids = [
            self.records.token_type_to_token_type_id['findings'], 
            self.records.token_type_to_token_type_id['impression'], 
        ]

        # Add set token identifiers in decoder's config:
        config_decoder.pad_token_id = self.tokenizer.pad_token_id

        # Encoder config:
        config_encoder = UniFormerWithProjectionHeadConfig(
            projection_size=config_decoder.hidden_size,
        )

        # Encoder-to-decoder model:
        if self.warm_start_modules:
            encoder = MultiUniFormerWithProjectionHead.from_pretrained('aehrc/uniformer_base_tl_384', config=config_encoder)
            decoder = transformers.LlamaForCausalLM(config=config_decoder)
            self.encoder_decoder = MIMICIVEDCXRMultimodalModel(encoder=encoder, decoder=decoder)
            
        else:
            config = transformers.VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
            config.decoder.add_cross_attention = False
            self.encoder_decoder = MIMICIVEDCXRMultimodalModel(
                config=config, 
                DefaultEncoderClass=MultiUniFormerWithProjectionHead,
                DefaultDecoderClass=transformers.LlamaForCausalLM,
            )
            
        self.encoder_decoder.config.is_encoder_decoder = False  # Need to make custom VisionEncoderDecoderConfig.

        # Image transformations:
        self.train_transforms = v2.Compose(
            [
                v2.Grayscale(num_output_channels=3),
                v2.Resize(
                    size=config_encoder.image_size, 
                    antialias=True, 
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomCrop(
                    size=[config_encoder.image_size, config_encoder.image_size],
                    pad_if_needed=True,
                ),
                v2.RandomRotation(degrees=5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
        self.test_transforms = v2.Compose(
            [
                v2.Grayscale(num_output_channels=3),
                v2.Resize(
                    size=config_encoder.image_size, 
                    antialias=True,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.CenterCrop(size=[config_encoder.image_size, config_encoder.image_size]),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
