import datetime
import math
import os

import torch
import transformers
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import v2

from modules.lightning_modules.ed.individual import EDExclusive
from modules.transformers.cxrmateed.modelling_mimic_iv_ed_cxr_rev_c import (
    MIMICIVEDCXRMultimodalModel,
)
from modules.transformers.mimic_iv_ed_cxr.modelling_mimic_iv_ed_cxr_rev_c_alt_a import (
    MIMICIVEDCXRMultimodalModel as MIMICIVEDCXRMultimodalModelAltA,
)
from modules.transformers.uniformer.configuration_uniformer import (
    UniFormerWithProjectionHeadConfig,
)
from modules.transformers.uniformer.modelling_uniformer_rev_b import (
    MultiUniFormerWithProjectionHead,
)
from tools.mimic_iv.ed_cxr.records_rev_a import EDCXRSubjectRecords
from tools.mimic_iv.ed_cxr.records_rev_a_alt_a import (
    EDCXRSubjectRecords as EDCXRSubjectRecordsAltA,
)
from tools.mimic_iv.ed_cxr.records_rev_a_alt_b import (
    EDCXRSubjectRecords as EDCXRSubjectRecordsAltB,
)
from tools.mimic_iv.ed_cxr.tables_rev_a import NUM_ED_CXR_TOKEN_TYPE_IDS


class TriageMedrecon(EDExclusive):

    def __init__(self, mimic_iv_duckdb_path, **kwargs):

        records = EDCXRSubjectRecords(database_path=mimic_iv_duckdb_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))
        records.ed_module_tables = {k: records.ed_module_tables[k] for k in ['triage', 'medrecon']}
        records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
        records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = []
        super().__init__(records=records, **kwargs)


class TriageMedreconText(EDExclusive):

    def __init__(self, mimic_iv_duckdb_path, **kwargs):

        records = EDCXRSubjectRecordsAltA(database_path=mimic_iv_duckdb_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))
        records.ed_module_tables = {k: records.ed_module_tables[k] for k in ['triage', 'medrecon']}
        records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
        records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = []
        super().__init__(records=records, **kwargs)

    def init_modules(self):
        """
        Initialise torch.nn.Modules.
        """

        index_value_encoder_config = {}
        for k, v in (self.records.ed_module_tables | self.records.mimic_cxr_tables).items():
            if v.load and (v.value_columns or v.index_columns or v.value_to_embedding_columns or v.index_to_embedding_columns):
                index_value_encoder_config[k] = v.total_indices

        # Decoder tokenizer:
        encoder_decoder_ckpt_name = f'{self.ckpt_zoo_dir}/mimic_iv_tokenizers/bpe_cxr_findings_impression_indication_history_ed_medrecon_vitalsign_triage'
        self.tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(encoder_decoder_ckpt_name)
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
        config_decoder.ed_module_columns = ['triage_value_to_text'] +  config_decoder.ed_module_columns
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
        encoder_ckpt_name = 'uniformer_base_tl_384'

        # Encoder-to-decoder model:
        if self.warm_start_modules:
            encoder = MultiUniFormerWithProjectionHead.from_pretrained(
                f'{self.ckpt_zoo_dir}/{encoder_ckpt_name}', config=config_encoder
            )
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


class TriageMedreconSeperateEmbeddings(EDExclusive):

    def __init__(self, mimic_iv_duckdb_path, **kwargs):

        records = EDCXRSubjectRecordsAltB(database_path=mimic_iv_duckdb_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))
        records.ed_module_tables = {k: records.ed_module_tables[k] for k in ['triage', 'medrecon']}
        records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
        records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = []
        super().__init__(records=records, **kwargs)

    def init_modules(self):
        """
        Initialise torch.nn.Modules.
        """

        index_value_encoder_config = {}
        for k, v in (self.records.ed_module_tables | self.records.mimic_cxr_tables).items():
            if v.load and (v.value_columns or v.index_columns or v.value_to_embedding_columns or v.index_to_embedding_columns):
                index_value_encoder_config[k] = v.total_indices

        # Decoder tokenizer:
        encoder_decoder_ckpt_name = f'{self.ckpt_zoo_dir}/mimic_iv_tokenizers/bpe_cxr_findings_impression_indication_history_ed_medrecon_vitalsign_triage'
        self.tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(encoder_decoder_ckpt_name)
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
        encoder_ckpt_name = 'uniformer_base_tl_384'

        # Encoder-to-decoder model:
        if self.warm_start_modules:
            encoder = MultiUniFormerWithProjectionHead.from_pretrained(
                f'{self.ckpt_zoo_dir}/{encoder_ckpt_name}', config=config_encoder
            )
            decoder = transformers.LlamaForCausalLM(config=config_decoder)
            self.encoder_decoder = MIMICIVEDCXRMultimodalModelAltA(encoder=encoder, decoder=decoder)
            
        else:
            config = transformers.VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
            config.decoder.add_cross_attention = False
            self.encoder_decoder = MIMICIVEDCXRMultimodalModelAltA(
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


# class MedRecon(EDExclusive):

#     def __init__(self, mimic_iv_duckdb_path=None, records=None, **kwargs):
        
#         if records is None:
#             records = EDCXRSubjectRecords(database_path=mimic_iv_duckdb_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1))
#             records.ed_module_tables = {k: records.ed_module_tables[k] for k in ['medrecon']}
#             records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
#             records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = []
#         super().__init__(records=records, **kwargs)