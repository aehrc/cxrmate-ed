import os
from typing import Optional

import torch
import torch.nn.functional as F
import transformers
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader

from .configuration_cxrmate_ed import CXRMateEDConfig
from .metrics import (
    AbsenceOfRepeatedNGramesMetric,
    BERTScoreRoBERTaLargeMetric,
    BLEUMetric,
    CheXbertMetric,
    CXRBERTMetric,
    GREENMetric,
    RadGraphMetric,
    ReportLogger,
    ReportTokenIdentifiersLogger,
    ROUGELMetric,
    SizeLogger,
)
from .modelling_cxrmate_ed import CXRMateEDModel
from .rewards import ARNReward, BERTScoreReward, CXRBERTReward


class MIMICIVEDCXRReportGen(LightningModule):

    def __init__(
            self,
            warm_start_modules: bool,
            exp_dir_trial: str,
            history: int,
            add_time_deltas: bool,
            tables_filter: list,
            prompt_report_sections_filter: list,
            database_dir=None,
            ckpt_zoo_dir: Optional[str] = None,
            mbatch_size: Optional[int] = None,
            decoder_max_len: Optional[int] = None,
            lr: Optional[float] = None,
            num_test_beams: Optional[int] = None,
            max_train_images_per_study: Optional[int] = None,
            type_vocab_size: int = 2,
            prefetch_factor: int = 5,
            num_workers: int = 0,
            accumulate_over_dicoms: bool = False,
            no_repeat_ngram_size: Optional[int] = None,
            **kwargs,
    ):
        LightningModule.__init__(self)

        self.warm_start_modules = warm_start_modules
        self.exp_dir_trial = exp_dir_trial
        self.history = history
        self.add_time_deltas = add_time_deltas
        self.tables_filter = tables_filter
        self.prompt_report_sections_filter = prompt_report_sections_filter
        self.database_dir = database_dir
        self.ckpt_zoo_dir = ckpt_zoo_dir
        self.mbatch_size = mbatch_size
        self.decoder_max_len = decoder_max_len
        self.lr = lr
        self.num_test_beams = num_test_beams
        self.max_train_images_per_study = max_train_images_per_study
        self.type_vocab_size = type_vocab_size
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.accumulate_over_dicoms = accumulate_over_dicoms
        self.no_repeat_ngram_size = no_repeat_ngram_size

        self.ckpt_epoch = 0

        """
        Evaluation metrics
        
        These need to be defined correctly in order for them to be placed on the correct device:
        https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#torchmetrics-in-pytorch-lightning
        """
        self.val_metrics, self.test_metrics = [], []
        
        for i in ['findings', 'impression', 'report']:
            
            # Absence of Repeat N-Grams metric:
            self.val_metrics.append(f'val_{i}_aor_ngram')
            setattr(
                self,
                self.val_metrics[-1],
                AbsenceOfRepeatedNGramesMetric(
                    split=f'val_{i}',
                    exp_dir=self.exp_dir_trial,
                    accumulate_over_dicoms=accumulate_over_dicoms,
                ),
            )
            self.test_metrics.append(f'test_{i}_aor_ngram')
            setattr(
                self,
                self.test_metrics[-1],
                AbsenceOfRepeatedNGramesMetric(
                    split=f'test_{i}',
                    exp_dir=self.exp_dir_trial,
                    accumulate_over_dicoms=accumulate_over_dicoms,
                ),
            )
            
            # BLEU metric:
            self.val_metrics.append(f'val_{i}_bleu')
            setattr(
                self,
                self.val_metrics[-1],
                BLEUMetric(
                    split=f'val_{i}',
                    exp_dir=self.exp_dir_trial,
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                ),
            )
            self.test_metrics.append(f'test_{i}_bleu')
            setattr(
                self,
                self.test_metrics[-1],
                BLEUMetric(
                    split=f'test_{i}',
                    exp_dir=self.exp_dir_trial,
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                ),
            )

            # ROUGE-L metric:
            self.val_metrics.append(f'val_{i}_rouge_l')
            setattr(
                self,
                self.val_metrics[-1],
                ROUGELMetric(
                    split=f'val_{i}',
                    exp_dir=self.exp_dir_trial,
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                ),
            )
            self.test_metrics.append(f'test_{i}_rouge_l')
            setattr(
                self,
                self.test_metrics[-1],
                ROUGELMetric(
                    split=f'test_{i}',
                    exp_dir=self.exp_dir_trial,
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                ),
            )
        
            # CheXbert metric:
            self.val_metrics.append(f'val_{i}_chexbert')
            setattr(
                self,
                self.val_metrics[-1],
                CheXbertMetric(
                    mbatch_size=1,
                    exp_dir=self.exp_dir_trial,
                    split=f'val_{i}',
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                )
            )
            self.test_metrics.append(f'test_{i}_chexbert')
            setattr(
                self,
                self.test_metrics[-1],
                CheXbertMetric(
                    mbatch_size=1,
                    exp_dir=self.exp_dir_trial,
                    split=f'test_{i}',
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                )
            )
        
            # CXR-BERT:
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

            # RadGraph:
            # self.test_metrics.append(f'test_{i}_rg')
            # setattr(
            #     self,
            #     self.test_metrics[-1],
            #     RadGraphMetric(
            #         mbatch_size=1,
            #         exp_dir=self.exp_dir_trial,
            #         split=f'test_{i}',
            #         accumulate_over_dicoms=self.accumulate_over_dicoms,
            #     ),
            # )

            # GREEN:
            # self.test_metrics.append(f'test_{i}_green')
            # setattr(
            #     self,
            #     self.test_metrics[-1],
            #     GREENMetric(
            #         mbatch_size=self.mbatch_size,
            #         exp_dir=self.exp_dir_trial,
            #         split=f'test_{i}',
            #         accumulate_over_dicoms=self.accumulate_over_dicoms,
            #     ),
            # )

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

        # Encoder config:
        config_encoder = transformers.AutoConfig.from_pretrained(
            'aehrc/uniformer_base_tl_384',
            trust_remote_code=True,
        )
        
        config = CXRMateEDConfig(
            vision_config=config_encoder,
            text_config=config_decoder,
            add_time_deltas=self.add_time_deltas,
            history = self.history,
            tables_filter=self.tables_filter,
            prompt_report_sections_filter=self.prompt_report_sections_filter,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        self.model = CXRMateEDModel(config=config)
        if self.warm_start_modules:
            self.model.image_encoder.encoder.load_state_dict(
                transformers.AutoModel.from_pretrained(
                    'aehrc/uniformer_base_tl_384', 
                    config=config_encoder,
                    trust_remote_code=True,
                ).state_dict()
            )
        self.model.train()

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """

        self.train_set, self.val_set, self.test_set = self.model.get_dataset(
            self.database_dir,
            self.max_train_images_per_study,
        )

        if stage == 'fit' or stage is None:
            print(f'No. of training examples: {self.train_set.__len__()}.')

        if stage == 'fit' or stage == 'validate' or stage is None:
            print(f'No. of validation examples: {self.val_set.__len__()}.')

        if stage == 'test' or stage is None:
            print(f'No. of test examples: {self.test_set.__len__()}.')

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
            collate_fn=self.model.collate_fn,
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
            collate_fn=self.model.collate_fn,
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
            collate_fn=self.model.collate_fn,
            pin_memory=True,
        )
    
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

        # Tokenize the report (findings and impression sections):
        tokenized_report = self.model.tokenize_report_teacher_forcing(
            batch['findings'], batch['impression'], self.tokenizer, self.decoder_max_len,
        )

        # Prepare the features from the tables:
        inputs_embeds, attention_mask, token_type_ids, position_ids, _ = self.model.prepare_inputs(
            tokenizer=self.tokenizer, 
            tokenized_report=tokenized_report, 
            sep_token_id=self.tokenizer.sep_token_id, 
            **batch,
        )

        # Teacher forcing: labels are given as input:
        y_hat = self.model.forward(
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
        inputs_embeds, attention_mask, token_type_ids, position_ids, bos_token_ids = self.model.prepare_inputs(tokenizer=self.tokenizer, **batch)

        # Greedy search:
        output_ids = self.model.generate(
            input_ids=bos_token_ids,
            decoder_inputs_embeds=inputs_embeds,
            decoder_token_type_ids=token_type_ids,
            prompt_attention_mask=attention_mask,
            prompt_position_ids=position_ids,
            special_token_ids=[self.tokenizer.sep_token_id],
            max_length=self.decoder_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id, 
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )['sequences']

        # Log report token identifier:
        self.val_report_ids_logger.update(output_ids, study_ids=batch['study_id'])

        # Findings and impression sections:
        findings, impression = self.model.split_and_decode_sections(
            output_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )

        # Log reports:
        self.val_report_logger.update(findings, impression, study_ids=batch['study_id'])

        # Evaluate:
        for i in self.val_metrics:
            if 'findings' in i:
                getattr(self, i).update(
                    findings, batch['findings'], study_ids=batch['study_id'],
                )
            elif 'impression' in i:
                getattr(self, i).update(
                    impression, batch['impression'], study_ids=batch['study_id'],
                )
            elif 'report' in i:
                getattr(self, i).update(
                    [f'{i} {j}' for i, j in zip(findings, impression)],
                    [f'{i} {j}' for i, j in zip(batch['findings'], batch['impression'])],
                    study_ids=batch['study_id'],
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
        inputs_embeds, attention_mask, token_type_ids, position_ids, bos_token_ids = self.model.prepare_inputs(tokenizer=self.tokenizer, **batch)
            
        # Beam search:
        output_ids = self.model.generate(
            input_ids=bos_token_ids,
            decoder_inputs_embeds=inputs_embeds,
            decoder_token_type_ids=token_type_ids,
            prompt_attention_mask=attention_mask,
            prompt_position_ids=position_ids,
            special_token_ids=[self.tokenizer.sep_token_id],
            max_length=self.decoder_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=self.num_test_beams,
            return_dict_in_generate=True,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            use_cache=True,
        )['sequences']

        # Log report token identifier:
        self.test_report_ids_logger.update(output_ids, study_ids=batch['study_id'])

        # Findings and impression sections:
        findings, impression = self.model.split_and_decode_sections(
            output_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )

        # Log reports:
        self.test_report_logger.update(findings, impression, study_ids=batch['study_id'])

        # Log prompt size:
        self.test_prompt_size_logger.update(attention_mask.sum(dim=1).tolist(), study_ids=batch['study_id'])

        # Evaluate:
        for i in self.test_metrics:
            if 'findings' in i:
                getattr(self, i).update(
                    findings, batch['findings'], study_ids=batch['study_id'],
                )
            elif 'impression' in i:
                getattr(self, i).update(
                    impression, batch['impression'], study_ids=batch['study_id'],
                )
            elif 'report' in i:
                getattr(self, i).update(
                    [f'{i} {j}' for i, j in zip(findings, impression)],
                    [f'{i} {j}' for i, j in zip(batch['findings'], batch['impression'])],
                    study_ids=batch['study_id'],
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


class Images(MIMICIVEDCXRReportGen):

    def __init__(self, **kwargs):
        
        kwargs['tables_filter'] = ['mimic_cxr_sectioned']
        kwargs['prompt_report_sections_filter'] = []
        kwargs['add_time_deltas'] = False

        super().__init__(**kwargs)

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """

        self.train_set, self.val_set, self.test_set = self.model.get_stage_1_dataset(
            self.database_dir,
            self.max_train_images_per_study,
        )

        if stage == 'fit' or stage is None:
            print(f'No. of training examples: {self.train_set.__len__()}.')

        if stage == 'fit' or stage == 'validate' or stage is None:
            print(f'No. of validation examples: {self.val_set.__len__()}.')

        if stage == 'test' or stage is None:
            print(f'No. of test examples: {self.test_set.__len__()}.')


class FreezeEncoderPartialWarmStartOptimiser(MIMICIVEDCXRReportGen):

    def __init__(self, allow_warm_start_optimiser_partial, warm_start_ckpt_path, **kwargs):
        super().__init__(**kwargs)

        self.allow_warm_start_optimiser_partial = allow_warm_start_optimiser_partial
        self.warm_start_ckpt_path = warm_start_ckpt_path
        self.warm_start_optimiser_partial = False

        # Freeze encoder:
        for n, p in self.model.named_parameters():
            if 'encoder.uniformer' in n:
                p.requires_grad = False

    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        new_model_params = []
        for k in self.model.tables.keys():
            if self.model.luts[k]['total'] > 0:   
                new_model_params += getattr(self.model, f'{k}_index_value_encoder').parameters()
        if hasattr(self.model, 'time_delta_encoder'):
            new_model_params += self.model.time_delta_encoder.parameters()

        if self.allow_warm_start_optimiser_partial and self.warm_start_optimiser_partial:
            base_model_param_groups = [
                {
                    'params': list(self.model.image_encoder.parameters()) + \
                        list(self.model.language_model.parameters()) + \
                        list(self.model.token_type_embeddings.parameters()),
                    'lr': self.lr,
                },
            ]
            optimiser = torch.optim.AdamW(base_model_param_groups)
            checkpoint = torch.load(self.warm_start_ckpt_path, map_location=self.device)
            optimiser_states = checkpoint['optimizer_states']
            assert len(optimiser_states) == 1
            optimiser_states = optimiser_states[0]
            optimiser.load_state_dict(optimiser_states)

            new_model_param_group = {'params': new_model_params, 'lr': self.lr}
            optimiser.add_param_group(new_model_param_group)
        else:
            param_groups = [
                {
                    'params': list(self.model.image_encoder.parameters()) + \
                        list(self.model.language_model.parameters()) + \
                        list(self.model.token_type_embeddings.parameters()),
                    'lr': self.lr,
                },
                {'params': new_model_params, 'lr': self.lr},
            ]

            optimiser = torch.optim.AdamW(param_groups)
        return {'optimizer': optimiser}


class VitalsignExclusive(FreezeEncoderPartialWarmStartOptimiser):
    
    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """

        self.train_set, self.val_set, self.test_set = self.model.get_dataset(
            self.database_dir,
            self.max_train_images_per_study,
            test_study_id_json_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mimic_iv_ed_mimic_cxr_jpg_vitalsign_test_study_ids.json'),
            test_set_only=True,
        )
        
        assert stage == 'test'
        
        if stage == 'test' or stage is None:
            print(f'No. of test examples: {self.test_set.__len__()}.')
            
            
class PyxisExclusive(FreezeEncoderPartialWarmStartOptimiser):
    
    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """

        self.train_set, self.val_set, self.test_set = self.model.get_dataset(
            self.database_dir,
            self.max_train_images_per_study,
            test_study_id_json_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mimic_iv_ed_mimic_cxr_jpg_pyxis_test_study_ids.json'),
            test_set_only=True,
        )
        
        assert stage == 'test'
        
        if stage == 'test' or stage is None:
            print(f'No. of test examples: {self.test_set.__len__()}.')


class SCST(FreezeEncoderPartialWarmStartOptimiser):
    
    def __init__(
        self,
        num_warmup_steps,
        scst_sample_top_p: float = 1.0,
        scst_sample_top_k: float = 50,
        scst_sample_temperature: float = 1.0,
        **kwargs,
    ):
        """
        Argument/s:
            scst_sample_top_p - only the most probable tokens with probabilities that add up to top_p or higher are
                considered during sampling.
            scst_sample_top_k - only the top-k ranked tokens are considered during sampling.
            scst_sample_temperature - the sharpness of the softmax probability distribution during sampling.
            kwargs - keyword arguments.
        """
        super(SCST, self).__init__(**kwargs)

        self.num_warmup_steps = num_warmup_steps
        self.scst_sample_top_p = scst_sample_top_p
        self.scst_sample_top_k = scst_sample_top_k
        self.scst_sample_temperature = scst_sample_temperature

        # Freeze the encoder:
        for p in self.model.image_encoder.parameters():
            p.requires_grad = False

    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """

        optimiser = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimiser,
            num_warmup_steps=self.num_warmup_steps,
        )
        return {
            'optimizer': optimiser,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
        
    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """
        
        # Prepare the features from the tables:
        inputs_embeds, attention_mask, token_type_ids, position_ids, bos_token_ids = self.model.prepare_inputs(tokenizer=self.tokenizer, **batch)
          
        # Samples:
        sample = self.model.generate.__wrapped__(  # Use __wrapped__ to avoid the torch.no_grad() decorator of generate().
            self.model,
            input_ids=bos_token_ids,
            prompt_attention_mask=attention_mask,
            prompt_position_ids=position_ids,
            decoder_inputs_embeds=inputs_embeds,
            decoder_token_type_ids=token_type_ids,
            special_token_ids=[self.tokenizer.sep_token_id],
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id, 
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
            do_sample=True,
            output_scores=True,
            top_p=self.scst_sample_top_p,
            top_k=self.scst_sample_top_k,
            temperature=self.scst_sample_temperature,
            max_new_tokens=self.decoder_max_len - 1,
        )

        # Sample logits:
        logits = torch.stack(sample['scores'], dim=-1)

        # Convert token indices into strings for the reward function:
        findings, impression = self.model.split_and_decode_sections(
            sample['sequences'],
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )
        sample_str = [' '.join(filter(None, [i, j])) for i, j in zip(findings, impression)]

        # Sampled token identifiers:
        generated_start_idx = 1
        sampled_token_ids = sample['sequences'][:, generated_start_idx:]

        # Sequence length:
        mask = sampled_token_ids == self.tokenizer.pad_token_id
        seq_len = torch.sum(torch.logical_not(mask), dim=-1).float()

        # Log sequence length:
        self.log_dict({'seq_len': torch.mean(seq_len)}, on_step=True, on_epoch=True, batch_size=seq_len.size()[0])
        
        # Sample reward:
        labels = [[' '.join(filter(None, [i, j]))] for i, j in zip(batch['findings'], batch['impression'])]
        reward = self.reward(sample_str, labels)

        # Baseline:
        baseline_ids = self.model.generate(
            input_ids=bos_token_ids,
            prompt_attention_mask=attention_mask,
            prompt_position_ids=position_ids,
            decoder_inputs_embeds=inputs_embeds,
            decoder_token_type_ids=token_type_ids,
            special_token_ids=[self.tokenizer.sep_token_id],
            max_length=self.decoder_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id, 
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
        )['sequences']
        baseline_findings, baseline_impression = self.model.split_and_decode_sections(
            baseline_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )
        baseline = self.reward(
            [' '.join(filter(None, [i, j]))for i, j in zip(baseline_findings, baseline_impression)], labels,
        ).to(self.device)
        reward = reward - baseline

        # Loss:
        loss = self.reinforce_loss(logits, sampled_token_ids, reward)

        # Log the rewards:
        self.log_dict(
            {'reward': torch.mean(reward), 'baseline': torch.mean(baseline)},
            on_step=True,
            on_epoch=True,
            batch_size=batch['images'].size()[0],
        )

        # Log the loss:
        self.log_dict({'scst_loss': loss}, on_step=True, on_epoch=True, batch_size=batch['images'].size()[0])

        return loss
    
    def reinforce_loss(self, logits: torch.Tensor, sampled_token_ids: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """
        Loss for the REINFORCE algorithm from https://doi.org/10.1007/BF00992696. It is detailed for
        gradient descent in https://doi.org/10.1109/cvpr.2017.131.
        
        PyTorch implementation:
            https://pytorch.org/docs/stable/distributions.html#score-function

        Argument/s
            logits - logits from the language model head.
            sampled_token_ids - sampled token indices.
            reward - reward for each batch element.

        Returns:
            REINFORCE loss for gradient descent.
        """
        
        # Probabilities:
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + (probs == 0)*torch.finfo(logits.dtype).smallest_normal)

        # Negative log-likelihood of each sampled token:
        loss = torch.nn.functional.nll_loss(
            input=log_probs,
            target=sampled_token_ids,
            ignore_index=self.tokenizer.pad_token_id,
            reduction='none',
        )

        # Negative sequence log-likelihood:
        loss = loss.sum(dim=-1)

        # Reward:
        loss = loss * reward

        # Mean over mini-batch elements:
        loss = loss.mean()
        
        return loss


class SCSTSections(SCST):
    
    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """
        
        # Prepare the features from the tables:
        inputs_embeds, attention_mask, token_type_ids, position_ids, bos_token_ids = self.model.prepare_inputs(tokenizer=self.tokenizer, **batch)
          
        # Samples:
        sample = self.model.generate.__wrapped__(  # Use __wrapped__ to avoid the torch.no_grad() decorator of generate().
            self.model,
            input_ids=bos_token_ids,
            prompt_attention_mask=attention_mask,
            prompt_position_ids=position_ids,
            decoder_inputs_embeds=inputs_embeds,
            decoder_token_type_ids=token_type_ids,
            special_token_ids=[self.tokenizer.sep_token_id],
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id, 
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
            do_sample=True,
            output_scores=True,
            top_p=self.scst_sample_top_p,
            top_k=self.scst_sample_top_k,
            temperature=self.scst_sample_temperature,
            max_new_tokens=self.decoder_max_len - 1,
        )

        # Sample logits:
        logits = torch.stack(sample['scores'], dim=-1)

        # Convert token indices into strings for the reward function:
        findings, impression = self.model.split_and_decode_sections(
            sample['sequences'],
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )

        # Sampled token identifiers:
        generated_start_idx = 1
        sampled_token_ids = sample['sequences'][:, generated_start_idx:]

        # Sequence length:
        mask = sampled_token_ids == self.tokenizer.pad_token_id
        seq_len = torch.sum(torch.logical_not(mask), dim=-1).float()

        # Log sequence length:
        self.log_dict({'seq_len': torch.mean(seq_len)}, on_step=True, on_epoch=True, batch_size=seq_len.size()[0])
        
        # Section labels:
        findings_labels = [[i] for i in batch['findings']]
        impression_labels = [[j] for j in batch['impression']]

        
        # Compute rewards separately for findings and impression:
        findings_reward = self.reward(findings, findings_labels)
        impression_reward = self.reward(impression, impression_labels)

        # Baseline:
        baseline_ids = self.model.generate(
            input_ids=bos_token_ids,
            prompt_attention_mask=attention_mask,
            prompt_position_ids=position_ids,
            decoder_inputs_embeds=inputs_embeds,
            decoder_token_type_ids=token_type_ids,
            special_token_ids=[self.tokenizer.sep_token_id],
            max_length=self.decoder_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id, 
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
        )['sequences']
        baseline_findings, baseline_impression = self.model.split_and_decode_sections(
            baseline_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )
        
        # Compute baseline rewards separately for findings and impression
        baseline_findings_reward = self.reward(baseline_findings, findings_labels).to(self.device)
        baseline_impression_reward = self.reward(baseline_impression, impression_labels).to(self.device)
        
        # Subtract baseline rewards from the sampled rewards:
        findings_reward = findings_reward - baseline_findings_reward
        impression_reward = impression_reward - baseline_impression_reward

        # Combine rewards for the final loss:
        reward = (findings_reward + impression_reward) / 2

        # Loss:
        loss = self.reinforce_loss(logits, sampled_token_ids, reward)

        # Log the rewards:
        self.log_dict(
            {
                'findings_reward': torch.mean(findings_reward), 
                'impression_reward': torch.mean(impression_reward), 
                'baseline_findings': torch.mean(baseline_findings_reward),
                'baseline_impression': torch.mean(baseline_impression_reward)
            },
            on_step=True, 
            on_epoch=True, 
            batch_size=batch['images'].size()[0],
        )

        # Log the loss:
        self.log_dict({'scst_loss': loss}, on_step=True, on_epoch=True, batch_size=batch['images'].size()[0])

        return loss    


class SCSTSectionsWeighted(SCST):
    
    def __init__(
        self,
        reward_section_weights,
        **kwargs,
    ):
        super(SCSTSectionsWeighted, self).__init__(**kwargs)
        self.reward_section_weights = reward_section_weights
    
    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """
        
        # Prepare the features from the tables:
        inputs_embeds, attention_mask, token_type_ids, position_ids, bos_token_ids = self.model.prepare_inputs(tokenizer=self.tokenizer, **batch)
          
        # Samples:
        sample = self.model.generate.__wrapped__(  # Use __wrapped__ to avoid the torch.no_grad() decorator of generate().
            self.model,
            input_ids=bos_token_ids,
            prompt_attention_mask=attention_mask,
            prompt_position_ids=position_ids,
            decoder_inputs_embeds=inputs_embeds,
            decoder_token_type_ids=token_type_ids,
            special_token_ids=[self.tokenizer.sep_token_id],
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id, 
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
            do_sample=True,
            output_scores=True,
            top_p=self.scst_sample_top_p,
            top_k=self.scst_sample_top_k,
            temperature=self.scst_sample_temperature,
            max_new_tokens=self.decoder_max_len - 1,
        )

        # Sample logits:
        logits = torch.stack(sample['scores'], dim=-1)

        # Convert token indices into strings for the reward function:
        findings, impression = self.model.split_and_decode_sections(
            sample['sequences'],
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )

        # Sampled token identifiers:
        generated_start_idx = 1
        sampled_token_ids = sample['sequences'][:, generated_start_idx:]

        # Sequence length:
        mask = sampled_token_ids == self.tokenizer.pad_token_id
        seq_len = torch.sum(torch.logical_not(mask), dim=-1).float()

        # Log sequence length:
        self.log_dict({'seq_len': torch.mean(seq_len)}, on_step=True, on_epoch=True, batch_size=seq_len.size()[0])
        
        # Section labels:
        findings_labels = [[i] for i in batch['findings']]
        impression_labels = [[j] for j in batch['impression']]

        
        # Compute rewards separately for findings and impression:
        findings_reward = self.reward(findings, findings_labels)
        impression_reward = self.reward(impression, impression_labels)

        # Baseline:
        baseline_ids = self.model.generate(
            input_ids=bos_token_ids,
            prompt_attention_mask=attention_mask,
            prompt_position_ids=position_ids,
            decoder_inputs_embeds=inputs_embeds,
            decoder_token_type_ids=token_type_ids,
            special_token_ids=[self.tokenizer.sep_token_id],
            max_length=self.decoder_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id, 
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
        )['sequences']
        baseline_findings, baseline_impression = self.model.split_and_decode_sections(
            baseline_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )
        
        # Compute baseline rewards separately for findings and impression
        baseline_findings_reward = self.reward(baseline_findings, findings_labels).to(self.device)
        baseline_impression_reward = self.reward(baseline_impression, impression_labels).to(self.device)
        
        # Subtract baseline rewards from the sampled rewards:
        findings_reward = findings_reward - baseline_findings_reward
        impression_reward = impression_reward - baseline_impression_reward

        # Combine rewards for the final loss:
        reward = (self.reward_section_weights[0] * findings_reward) + (self.reward_section_weights[1] * impression_reward)

        # Loss:
        loss = self.reinforce_loss(logits, sampled_token_ids, reward)

        # Log the rewards:
        self.log_dict(
            {
                'findings_reward': torch.mean(findings_reward), 
                'impression_reward': torch.mean(impression_reward), 
                'baseline_findings': torch.mean(baseline_findings_reward),
                'baseline_impression': torch.mean(baseline_impression_reward)
            },
            on_step=True, 
            on_epoch=True, 
            batch_size=batch['images'].size()[0],
        )

        # Log the loss:
        self.log_dict({'scst_loss': loss}, on_step=True, on_epoch=True, batch_size=batch['images'].size()[0])

        return loss


class EASTSectionsWeighted(SCSTSectionsWeighted):
    
    def __init__(self, entropy_weight, **kwargs):
        SCSTSectionsWeighted.__init__(self, **kwargs)
        self.entropy_weight = entropy_weight
    
    def reinforce_loss(self, logits: torch.Tensor, sampled_token_ids: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """
        Loss for the REINFORCE algorithm from https://doi.org/10.1007/BF00992696. It is detailed for
        gradient descent in https://doi.org/10.1109/cvpr.2017.131. The entropy term is added to the loss, 
        forming Entropy-Augmented SCST (EAST) https://aclanthology.org/2024.bionlp-1.8/.
        
        PyTorch implementation:
            https://pytorch.org/docs/stable/distributions.html#score-function

        Argument/s
            logits - logits from the language model head.
            sampled_token_ids - sampled token indices.
            reward - reward for each batch element.

        Returns:
            REINFORCE loss for gradient descent.
        """
        
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + (probs == 0)*torch.finfo(logits.dtype).smallest_normal)

        # Negative log-likelihood of each sampled token:
        loss = torch.nn.functional.nll_loss(
            input=log_probs,
            target=sampled_token_ids,
            ignore_index=self.tokenizer.pad_token_id,
            reduction='none',
        )

        # Negative sequence log-likelihood:
        loss = loss.sum(dim=-1)

        # Reward:
        loss = loss * reward

        # Padding mask:
        padding_mask = (sampled_token_ids != self.tokenizer.pad_token_id).unsqueeze(1)

        # Masked per-token entropy:
        entropy = -(probs * log_probs) * padding_mask

        # Summation over the token distribution for each output token:
        entropy = entropy.sum(dim=1)

        # Mean over the sequence:
        entropy = entropy.sum(dim=1) / padding_mask.squeeze(dim=1).sum(dim=1)

        # Add entropy term:      
        loss = loss - (self.entropy_weight * entropy) 

        # Mean over mini-batch elements:
        loss = loss.mean()

        return loss
        
        
class SCSTCXRBERTReward(SCST):
    
    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward = CXRBERTReward(device=self.device)


class SCSTCXRBERTBERTScoreReward(SCST):
    
    def __init__(self, weights=[0.5, 0.5], **kwargs):
        super(SCSTCXRBERTBERTScoreReward, self).__init__(**kwargs)
        self.weights = weights
        
    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward_cxrbert = CXRBERTReward(device=self.device)
        self.reward_bertscore = BERTScoreReward(device=self.device, num_workers=self.num_workers)

    def reward(self, predictions, labels):
        reward_cxrbert = self.reward_cxrbert(predictions, labels)
        reward_bertscore = self.reward_bertscore(predictions, labels)
        
        reward_cxrbert = self.weights[0]*reward_cxrbert
        reward_bertscore = self.weights[1]*reward_bertscore

        # Composite reward:
        reward = reward_cxrbert + reward_bertscore
        
        return reward


class SCSTSectionsCXRBERTBERTScoreReward(SCSTSections):
    
    def __init__(self, weights=[0.5, 0.5], **kwargs):
        super(SCSTSectionsCXRBERTBERTScoreReward, self).__init__(**kwargs)
        self.weights = weights
        
    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward_cxrbert = CXRBERTReward(device=self.device)
        self.reward_bertscore = BERTScoreReward(device=self.device, num_workers=self.num_workers)

    def reward(self, predictions, labels):
        reward_cxrbert = self.reward_cxrbert(predictions, labels)
        reward_bertscore = self.reward_bertscore(predictions, labels)
        
        reward_cxrbert = self.weights[0]*reward_cxrbert
        reward_bertscore = self.weights[1]*reward_bertscore

        # Composite reward:
        reward = reward_cxrbert + reward_bertscore
        
        return reward


class SCSTSectionsWeightedCXRBERTBERTScoreReward(SCSTSectionsWeighted):
    
    def __init__(self, weights=[0.5, 0.5], **kwargs):
        super(SCSTSectionsWeightedCXRBERTBERTScoreReward, self).__init__(**kwargs)
        self.weights = weights
        
    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward_cxrbert = CXRBERTReward(device=self.device)
        self.reward_bertscore = BERTScoreReward(device=self.device, num_workers=self.num_workers)

    def reward(self, predictions, labels):
        reward_cxrbert = self.reward_cxrbert(predictions, labels)
        reward_bertscore = self.reward_bertscore(predictions, labels)
        
        reward_cxrbert = self.weights[0]*reward_cxrbert
        reward_bertscore = self.weights[1]*reward_bertscore

        # Composite reward:
        reward = reward_cxrbert + reward_bertscore
        
        return reward


class SCSTSectionsWeightedCXRBERTBERTScoreARNReward(SCSTSectionsWeighted):
    
    def __init__(self, reward_no_repeat_ngram_size, weights=[0.45, 0.45, 0.1], **kwargs):
        SCSTSectionsWeighted.__init__(self, **kwargs)
        self.reward_no_repeat_ngram_size = reward_no_repeat_ngram_size
        self.weights = weights
        
    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward_cxrbert = CXRBERTReward(device=self.device)
        self.reward_bertscore = BERTScoreReward(device=self.device, num_workers=self.num_workers)
        self.reward_ngram = ARNReward(n=self.reward_no_repeat_ngram_size, device=self.device, tokenizer=self.tokenizer)

    def reward(self, predictions, labels):
        reward_cxrbert = self.reward_cxrbert(predictions, labels)
        reward_bertscore = self.reward_bertscore(predictions, labels)
        reward_ngram = self.reward_ngram(predictions)
        
        reward_cxrbert = self.weights[0]*reward_cxrbert
        reward_bertscore = self.weights[1]*reward_bertscore
        reward_ngram = self.weights[2]*reward_ngram

        # Composite reward:
        reward = reward_cxrbert + reward_bertscore + reward_ngram
        
        return reward


class EASTSectionsWeightedCXRBERTBERTScoreARNReward(EASTSectionsWeighted):
    
    def __init__(self, reward_no_repeat_ngram_size, weights=[0.45, 0.45, 0.1], **kwargs):
        EASTSectionsWeighted.__init__(self, **kwargs)
        self.reward_no_repeat_ngram_size = reward_no_repeat_ngram_size
        self.weights = weights
        
    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward_cxrbert = CXRBERTReward(device=self.device)
        self.reward_bertscore = BERTScoreReward(device=self.device, num_workers=self.num_workers)
        self.reward_ngram = ARNReward(n=self.reward_no_repeat_ngram_size, device=self.device, tokenizer=self.tokenizer)

    def reward(self, predictions, labels):
        reward_cxrbert = self.reward_cxrbert(predictions, labels)
        reward_bertscore = self.reward_bertscore(predictions, labels)
        reward_ngram = self.reward_ngram(predictions)
        
        reward_cxrbert = self.weights[0]*reward_cxrbert
        reward_bertscore = self.weights[1]*reward_bertscore
        reward_ngram = self.weights[2]*reward_ngram

        # Composite reward:
        reward = reward_cxrbert + reward_bertscore + reward_ngram
        
        return reward

