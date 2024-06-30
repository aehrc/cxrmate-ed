import os
from typing import Optional

import torch
import torch.nn.functional as F
import transformers
from lightning.pytorch import LightningModule
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from data.dataset.study_id_ed_stay_id_rev_b import StudyIDEDStayIDSubset
from modules.transformers.mimic_iv_ed_cxr.modelling_mimic_iv_ed_cxr_rev_b import (
    MIMICIVEDCXRMultimodalModel,
)
from modules.transformers.uniformer.configuration_uniformer import (
    UniFormerWithProjectionHeadConfig,
)
from modules.transformers.uniformer.modelling_uniformer_rev_b import (
    MultiUniFormerWithProjectionHead,
)
from tools.metrics.bertscore import BERTScoreRoBERTaLargeMetric
from tools.metrics.chexbert import CheXbertClassificationMetrics
from tools.metrics.coco import COCONLGMIMICCXRMetrics
from tools.metrics.cxr_bert import CXRBERTMetric
from tools.metrics.radgraph import RadGraphMetric
from tools.metrics.report_ids_logger import ReportTokenIdentifiersLogger
from tools.metrics.report_logger import ReportLogger
from tools.metrics.size_logger import SizeLogger
from tools.mimic_iv.ed_cxr.records import EDCXRSubjectRecords
from tools.mimic_iv.ed_cxr.tables import NUM_ED_CXR_TOKEN_TYPE_IDS


class MIMICIVEDCXRReportGen(LightningModule):

    def __init__(
            self,
            warm_start_modules: bool,
            exp_dir_trial: str,
            dataset_dir: str,
            physionet_dir: str,
            mimic_iv_duckdb_path=None,
            images_rocksdb_path=None,
            records=None, 
            num_token_type_ids=None,
            ckpt_zoo_dir: Optional[str] = None,
            mbatch_size: Optional[int] = None,
            decoder_max_len: Optional[int] = None,
            lr: Optional[float] = None,
            num_test_beams: Optional[int] = None,
            max_images_per_study: Optional[int] = None,
            sections_to_evaluate: list = ['report'],
            type_vocab_size: int = 2,
            prefetch_factor: int = 5,
            num_workers: int = 0,
            accumulate_over_dicoms: bool = False,
            nlg_val_metrics: list = ['bleu', 'cider', 'rouge'],
            nlg_test_metrics: list = ['bleu', 'cider', 'rouge', 'meteor'],
            image_dir: Optional[str] = None,
            module_load_apptainer: Optional[str] = None,
            use_radgraph_metric: bool = True,
            debug: bool = False,
            **kwargs,
    ):
        LightningModule.__init__(self)

        self.warm_start_modules = warm_start_modules
        self.exp_dir_trial = exp_dir_trial
        self.dataset_dir = dataset_dir
        self.physionet_dir = physionet_dir
        self.mimic_iv_duckdb_path = mimic_iv_duckdb_path
        self.images_rocksdb_path = images_rocksdb_path
        self.ckpt_zoo_dir = ckpt_zoo_dir
        self.mbatch_size = mbatch_size
        self.decoder_max_len = decoder_max_len
        self.lr = lr
        self.num_test_beams = num_test_beams
        self.max_images_per_study = max_images_per_study
        self.sections_to_evaluate = sections_to_evaluate
        self.type_vocab_size = type_vocab_size
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.accumulate_over_dicoms = accumulate_over_dicoms
        self.image_dir = image_dir
        self.module_load_apptainer = module_load_apptainer
        self.use_radgraph_metric = use_radgraph_metric
        self.debug = debug

        self.ckpt_epoch = 0

        self.records = EDCXRSubjectRecords(database_path=mimic_iv_duckdb_path) if records is None else records
        self.num_token_type_ids = NUM_ED_CXR_TOKEN_TYPE_IDS if num_token_type_ids is None else num_token_type_ids

        # Paths:
        self.merged_csv_path = os.path.join(self.dataset_dir, 'mimic_cxr_merged', 'splits_reports_metadata.csv')
        self.tokenizer_dir =  os.path.join(self.ckpt_zoo_dir, 'mimic-cxr-tokenizers', 'bpe_prompt')
        self.mimic_cxr_dir = os.path.join(self.dataset_dir, 'physionet.org', 'files', 'mimic-cxr-jpg', '2.0.0', 'files')

        """
        Evaluation metrics
        
        These need to be defined correctly in order for them to be placed on the correct device:
        https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#torchmetrics-in-pytorch-lightning
        """
        self.val_metrics, self.test_metrics = [], []
        
        # COCO NLG metrics:
        for i in self.sections_to_evaluate:
            self.val_metrics.append(f'val_{i}_nlg')
            setattr(
                self,
                self.val_metrics[-1],
                COCONLGMIMICCXRMetrics(
                    split=f'val_{i}',
                    metrics=nlg_val_metrics,
                    exp_dir=self.exp_dir_trial,
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                ),
            )
        
        # Cannot deepcopy either SPICE or METEOR:
        for i in self.sections_to_evaluate:
            self.test_metrics.append(f'test_{i}_nlg')
            setattr(
                self,
                self.test_metrics[-1],
                COCONLGMIMICCXRMetrics(
                    split=f'test_{i}',
                    metrics=nlg_test_metrics,
                    exp_dir=self.exp_dir_trial,
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                ),
            )

        # RadGraph:
        # if use_radgraph_metric:
        #     for i in self.sections_to_evaluate:
        #         self.test_metrics.append(f'test_{i}_radgraph')
        #         setattr(
        #             self,
        #             self.test_metrics[-1],
        #             RadGraphMetric(
        #                 mbatch_size=1,
        #                 exp_dir=self.exp_dir_trial,
        #                 split=f'test_{i}',
        #                 accumulate_over_dicoms=self.accumulate_over_dicoms,
        #                 image_dir=self.image_dir,
        #                 module_load_apptainer=self.module_load_apptainer,
        #             ),
        #         )

        # CheXbert metrics:
        for i in self.sections_to_evaluate:
            self.val_metrics.append(f'val_{i}_chexbert')
            setattr(
                self,
                self.val_metrics[-1],
                CheXbertClassificationMetrics(
                    bert_path='bert-base-uncased',
                    checkpoint_path='stanford/chexbert/chexbert.pth',
                    ckpt_dir=self.ckpt_zoo_dir,
                    mbatch_size=1,
                    exp_dir=self.exp_dir_trial,
                    split=f'val_{i}',
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                )
            )
        for i in self.sections_to_evaluate:
            self.test_metrics.append(f'test_{i}_chexbert')
            setattr(
                self,
                self.test_metrics[-1],
                CheXbertClassificationMetrics(
                    bert_path='bert-base-uncased',
                    checkpoint_path='stanford/chexbert/chexbert.pth',
                    ckpt_dir=self.ckpt_zoo_dir,
                    mbatch_size=1,
                    exp_dir=self.exp_dir_trial,
                    split=f'test_{i}',
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                )
            )
        
        # CXR-BERT:
        for i in self.sections_to_evaluate:
            self.val_metrics.append(f'val_{i}_cxr-bert')
            setattr(
                self,
                self.val_metrics[-1],
                CXRBERTMetric(
                    mbatch_size=1,
                    exp_dir=self.exp_dir_trial,
                    split=f'val_{i}',
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                ),
            )
            self.test_metrics.append(f'test_{i}_cxr-bert')
            setattr(
                self,
                self.test_metrics[-1],
                CXRBERTMetric(
                    mbatch_size=1,
                    exp_dir=self.exp_dir_trial,
                    split=f'test_{i}',
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                ),
            )

        # BERTScore:
        for i in self.sections_to_evaluate:
            self.val_metrics.append(f'val_{i}_bertscore')
            setattr(
                self,
                self.val_metrics[-1],
                BERTScoreRoBERTaLargeMetric(
                    mbatch_size=1,
                    exp_dir=self.exp_dir_trial,
                    split=f'val_{i}',
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                    num_workers=self.num_workers,
                ),
            )
            self.test_metrics.append(f'test_{i}_bertscore')
            setattr(
                self,
                self.test_metrics[-1],
                BERTScoreRoBERTaLargeMetric(
                    mbatch_size=1,
                    exp_dir=self.exp_dir_trial,
                    split=f'test_{i}',
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                    num_workers=self.num_workers,
                ),
            )

        # Report logging:
        self.val_report_logger = ReportLogger(
            exp_dir=self.exp_dir_trial, split='val_reports', track_dicom_id=self.accumulate_over_dicoms,
        )
        self.test_report_logger = ReportLogger(
            exp_dir=self.exp_dir_trial, split='test_reports', track_dicom_id=self.accumulate_over_dicoms,
        )
        self.val_report_ids_logger = ReportTokenIdentifiersLogger(
            exp_dir=self.exp_dir_trial, split='val_report_ids', track_dicom_id=self.accumulate_over_dicoms,
        )
        self.test_report_ids_logger = ReportTokenIdentifiersLogger(
            exp_dir=self.exp_dir_trial, split='test_report_ids', track_dicom_id=self.accumulate_over_dicoms,
        )
        self.test_prompt_size_logger = SizeLogger(
            exp_dir=self.exp_dir_trial, split='test_prompt_size', track_dicom_id=self.accumulate_over_dicoms,
        )

        # Initialise modules:
        self.init_modules()

    def init_modules(self):
        """
        Initialise torch.nn.Modules.
        """

        index_value_encoder_config = {}
        for k, v in (self.records.ed_module_tables | self.records.mimic_cxr_tables).items():
            if v.load and (v.value_columns or v.index_columns):
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

        # Section embedding identifiers (for report):
        self.section_ids = [
            self.records.token_type_to_token_type_id['findings'], 
            self.records.token_type_to_token_type_id['impression'], 
        ]

        # Add set token identifiers in decoder's config:
        config_decoder.img_token_id = self.records.token_type_to_token_type_id['image']
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
        
    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """

        if stage == 'fit' or stage is None:
            self.train_set = StudyIDEDStayIDSubset(
                mimic_iv_duckdb_path=self.mimic_iv_duckdb_path,
                images_rocksdb_path=self.images_rocksdb_path,
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.train_transforms,
                split='train',
                max_images_per_study=5,
                records=self.records,
            )
            print(f'No. of training examples: {self.train_set.__len__()}.')
            print(
                f'No. of training dicom_ids, study_ids, & subject_ids: {self.train_set.num_dicom_ids},',
                f'{self.train_set.num_study_ids}, & {self.train_set.num_subject_ids}.',
            )

        if stage == 'fit' or stage == 'validate' or stage is None:
            self.val_set = StudyIDEDStayIDSubset(
                mimic_iv_duckdb_path=self.mimic_iv_duckdb_path,
                images_rocksdb_path=self.images_rocksdb_path,
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.test_transforms,
                split='validate',
                max_images_per_study=5,
                records=self.records,
            )
            print(f'No. of validation examples: {self.val_set.__len__()}.')
            print(
                f'No. of validation dicom_ids, study_ids, & subject_ids: {self.val_set.num_dicom_ids},',
                f'{self.val_set.num_study_ids}, & {self.val_set.num_subject_ids}.',
            )

        if stage == 'test' or stage is None:
            self.test_set = StudyIDEDStayIDSubset(
                mimic_iv_duckdb_path=self.mimic_iv_duckdb_path,
                images_rocksdb_path=self.images_rocksdb_path,
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.test_transforms,
                split='test',
                max_images_per_study=5,
                records=self.records,
            )
            print('No. of test examples: {}.'.format(self.test_set.__len__()))
            print(
                f'No. of test dicom_ids, study_ids, & subject_ids: {self.test_set.num_dicom_ids},',
                f'{self.test_set.num_study_ids}, & {self.test_set.num_subject_ids}.',
            )

    def train_dataloader(self, shuffle=True):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        return DataLoader(
            self.train_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#val-dataloader
        """
        return DataLoader(
            self.val_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-dataloader
        """
        return DataLoader(
            self.test_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    @staticmethod
    def collate_fn(batch):
        keys = set().union(*(d.keys() for d in batch))
        batch = {j: [i.setdefault(j, None) for i in batch] for j in keys}
        batch['images'] = torch.nn.utils.rnn.pad_sequence(batch['images'], batch_first=True, padding_value=0.0)

        for k in keys:
            if 'index_value_feats' in k:

                total_indices = next(i for i in batch[k] if i is not None).shape[-1]
                batch[k] = [i if i is not None else torch.empty(0, total_indices) for i in batch[k]]
                batch[k] = torch.nn.utils.rnn.pad_sequence(batch[k], batch_first=True, padding_value=-1)  # Pad value of -1 is not ideal. Need to use something else.
                token_type_id_name = k.replace('_feats', '_token_type_ids')
                batch[token_type_id_name] = [i if i is not None else torch.empty(0, dtype=torch.long) for i in batch[token_type_id_name]]
                batch[token_type_id_name] = torch.nn.utils.rnn.pad_sequence(
                    batch[token_type_id_name], batch_first=True, padding_value=0,
                )
                mask_name = k.replace('_feats', '_mask')
                batch[mask_name] = (batch[k] != -1).any(dim=-1).int()
            
            if 'time_delta' in k and 'index_value' in k:
                batch[k] = [i if i is not None else torch.empty(0, 1) for i in batch[k]]
                batch[k] = torch.nn.utils.rnn.pad_sequence(batch[k], batch_first=True, padding_value=0)

        return batch
    
    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        optimiser = {'optimizer': torch.optim.AdamW(self.parameters(), lr=self.lr)}
        return optimiser
    
    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """

        # Tokenize the report (fingins and impression sections):
        tokenized_report = self.encoder_decoder.tokenize_report_teacher_forcing(
            batch['findings'], batch['impression'], self.tokenizer, self.decoder_max_len,
        )

        # Prepare the features from the tables:
        inputs_embeds, attention_mask, token_type_ids, position_ids, _ = self.encoder_decoder.prepare_inputs(
            tokenizer=self.tokenizer, 
            tokenized_report=tokenized_report, 
            sep_token_id=self.tokenizer.sep_token_id, 
            section_ids=self.section_ids, 
            **batch,
        )

        # Teacher forcing: labels are given as input:
        y_hat = self.encoder_decoder.forward(
            decoder_inputs_embeds=inputs_embeds,
            decoder_attention_mask=attention_mask,
            decoder_token_type_ids=token_type_ids,
            decoder_position_ids=position_ids,
            return_dict=True,
        ).logits

        # Add padding to account for non-text positions in prompt:
        tokenized_report['label_ids'] = F.pad(
            tokenized_report['label_ids'],
            (y_hat.shape[1] - tokenized_report['label_ids'].shape[1], 0, 0, 0),
            'constant',
            self.tokenizer.pad_token_id,
        )

        # Loss:
        loss = F.cross_entropy(
            y_hat.permute([0, 2, 1]), 
            tokenized_report['label_ids'], 
            ignore_index=self.tokenizer.pad_token_id,
        )

        # Logging:
        self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True, batch_size=batch['images'].size()[0])

        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """

        # Prepare the features from the tables:
        inputs_embeds, attention_mask, token_type_ids, position_ids, bos_token_ids = self.encoder_decoder.prepare_inputs(tokenizer=self.tokenizer, **batch)

        # Greedy search:
        output_ids = self.encoder_decoder.generate(
            input_ids=bos_token_ids,
            decoder_inputs_embeds=inputs_embeds,
            decoder_token_type_ids=token_type_ids,
            prompt_attention_mask=attention_mask,
            prompt_position_ids=position_ids,
            special_token_ids=[self.tokenizer.sep_token_id],
            token_type_id_sections=self.section_ids,
            max_length=self.decoder_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id, 
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
        )['sequences']

        # Log report token identifier:
        self.val_report_ids_logger.update(output_ids, study_ids=batch['study_ids'])

        # Findings and impression sections:
        findings, impression = self.encoder_decoder.split_and_decode_sections(
            output_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )

        # Log reports:
        self.val_report_logger.update(findings, impression, study_ids=batch['study_ids'])

        # Evaluate:
        for i in self.val_metrics:
            if 'findings' in i:
                getattr(self, i).update(
                    findings, batch['findings'], study_ids=batch['study_ids'],
                )
            elif 'impression' in i:
                getattr(self, i).update(
                    impression, batch['impression'], study_ids=batch['study_ids'],
                )
            elif 'report' in i:
                getattr(self, i).update(
                    [f'{i} {j}' for i, j in zip(findings, impression)],
                    [f'{i} {j}' for i, j in zip(batch['findings'], batch['impression'])],
                    study_ids=batch['study_ids'],
                )
            else:
                raise ValueError(f'{i} must contain findings, impression, or report')

    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-validation-epoch-end
        """

        if (self.current_epoch > 0) or not hasattr(self, 'ckpt_epoch'):
            self.ckpt_epoch = self.current_epoch

        # Save reports:
        self.val_report_logger.compute(self.ckpt_epoch)
        self.val_report_logger.reset()
        self.val_report_ids_logger.compute(self.ckpt_epoch)
        self.val_report_ids_logger.reset()

        scores = {'epoch': float(self.ckpt_epoch)}
        for i in self.val_metrics:
            output = getattr(self, i).compute(self.ckpt_epoch)
            if isinstance(output, dict):
                for k, v in output.items():
                    scores.update({k: v})
            else:
                scores.update({i: output})

        self.log_dict(scores, on_step=False, on_epoch=True)
        [getattr(self, i).reset() for i in self.val_metrics]

    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """

        # Prepare the features from the tables:
        inputs_embeds, attention_mask, token_type_ids, position_ids, bos_token_ids = self.encoder_decoder.prepare_inputs(tokenizer=self.tokenizer, **batch)
            
        # Beam search:
        output_ids = self.encoder_decoder.generate(
            input_ids=bos_token_ids,
            decoder_inputs_embeds=inputs_embeds,
            decoder_token_type_ids=token_type_ids,
            prompt_attention_mask=attention_mask,
            prompt_position_ids=position_ids,
            special_token_ids=[self.tokenizer.sep_token_id],
            token_type_id_sections=self.section_ids,
            max_length=self.decoder_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=self.num_test_beams,
            return_dict_in_generate=True,
            use_cache=True,
        )['sequences']

        # Log report token identifier:
        self.test_report_ids_logger.update(output_ids, study_ids=batch['study_ids'])

        # Findings and impression sections:
        findings, impression = self.encoder_decoder.split_and_decode_sections(
            output_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )

        # Log reports:
        self.test_report_logger.update(findings, impression, study_ids=batch['study_ids'])

        # Log prompt size:
        self.test_prompt_size_logger.update(attention_mask.sum(dim=1).tolist(), study_ids=batch['study_ids'])

        # Evaluate:
        for i in self.test_metrics:
            if 'findings' in i:
                getattr(self, i).update(
                    findings, batch['findings'], study_ids=batch['study_ids'],
                )
            elif 'impression' in i:
                getattr(self, i).update(
                    impression, batch['impression'], study_ids=batch['study_ids'],
                )
            elif 'report' in i:
                getattr(self, i).update(
                    [f'{i} {j}' for i, j in zip(findings, impression)],
                    [f'{i} {j}' for i, j in zip(batch['findings'], batch['impression'])],
                    study_ids=batch['study_ids'],
                )
            else:
                raise ValueError(f'{i} must contain findings, impression, or report')
            
            
    def on_test_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-test-epoch-end
        """

        # Save reports:
        self.test_report_logger.compute(self.ckpt_epoch)
        self.test_report_logger.reset()
        self.test_report_ids_logger.compute(self.ckpt_epoch)
        self.test_report_ids_logger.reset()
        output = self.test_prompt_size_logger.compute(self.ckpt_epoch)
        self.test_prompt_size_logger.reset()
        
        scores = {'epoch': float(self.ckpt_epoch), **output}
        for i in self.test_metrics:
            output = getattr(self, i).compute(self.ckpt_epoch)
            if isinstance(output, dict):
                for k, v in output.items():
                    scores.update({k: v})
            else:
                scores.update({i: output})

        self.log_dict(scores, on_step=False, on_epoch=True)
        [getattr(self, i).reset() for i in self.test_metrics]
