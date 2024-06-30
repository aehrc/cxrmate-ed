import math
import os
from glob import glob
from pathlib import Path
from typing import Optional, Tuple, Union

import duckdb
import pandas as pd
import streamlit as st
import torch
import transformers
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, VisionEncoderDecoderModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.vision_encoder_decoder.configuration_vision_encoder_decoder import (
    VisionEncoderDecoderConfig,
)
from transformers.utils import logging

from .create_section_files import create_section_files
from .dataset import StudyIDEDStayIDSubset
from .modelling_uniformer import MultiUniFormerWithProjectionHead
from .records import EDCXRSubjectRecords
from .tables import ed_module_tables, mimic_cxr_tables

logger = logging.get_logger(__name__)


def create_lookup_table(df, columns, start_idx):
    df = df.groupby(columns).head(1)[columns].sort_values(by=columns)
    indices = range(start_idx, start_idx + len(df))
    df['index'] = indices
    return df, indices[-1]


class FNNEncoder(torch.nn.Module):
    def __init__(self, num_features, intermediate_size, decoder_hidden_size):
        super().__init__()
        self.up_proj = torch.nn.Linear(num_features, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, decoder_hidden_size, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class MIMICIVEDCXRMultimodalModel(VisionEncoderDecoderModel):

    config_class = VisionEncoderDecoderConfig
    base_model_prefix = "vision_encoder_decoder"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(        
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        DefaultEncoderClass = MultiUniFormerWithProjectionHead,
        DefaultDecoderClass = transformers.LlamaForCausalLM,
    ):

        if decoder:
            assert not decoder.config.add_cross_attention, '"add_cross_attention" must be False for the given decoder'
            assert decoder.config.is_decoder, '"is_decoder" must be True for the given decoder'

        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        config.tie_word_embeddings = False
        config.is_encoder_decoder = False

        # Initialize with config:
        PreTrainedModel.__init__(self, config)

        # Encoder:
        if encoder is None:
            encoder = DefaultEncoderClass(config=config.encoder)

        # Decoder:
        if decoder is None:
            assert not config.decoder.add_cross_attention
            decoder = DefaultDecoderClass(config=config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        assert config.decoder.is_decoder
        assert not config.decoder.is_encoder_decoder
        assert 'pad_token_id' in self.decoder.config.__dict__
        assert 'time_delta_monotonic_inversion' in self.decoder.config.__dict__
        assert 'zero_time_delta_value' in self.decoder.config.__dict__
        assert 'add_time_deltas' in self.decoder.config.__dict__

        assert isinstance(self.decoder.config.time_delta_monotonic_inversion, bool)
        assert isinstance(self.decoder.config.zero_time_delta_value, float)

        for k, v in self.decoder.config.index_value_encoder_config.items():
            setattr(
                self, 
                f'{k}_index_value_encoder', 
                FNNEncoder(
                    num_features=v, 
                    intermediate_size=self.decoder.config.index_value_encoder_intermediate_size, 
                    decoder_hidden_size=self.decoder.config.hidden_size,
                ),
            )
        if self.decoder.config.add_time_deltas:
            self.time_delta_encoder = FNNEncoder(
                num_features=1, 
                intermediate_size=self.decoder.config.index_value_encoder_intermediate_size, 
                decoder_hidden_size=self.decoder.config.hidden_size,
            )
        self.token_type_embeddings = torch.nn.Embedding(self.decoder.config.num_token_types, self.decoder.config.hidden_size)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        r"""
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the image encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co. An
                      example is `google/vit-base-patch16-224-in21k`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the text decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, *optional*):
                All remaning positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import VisionEncoderDecoderModel

        >>> # initialize a vit-bert from a pretrained ViT and a pretrained BERT model. Note that the cross-attention layers will be randomly initialized
        >>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "google/vit-base-patch16-224-in21k", "google-bert/bert-base-uncased"
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./vit-bert")
        >>> # load fine-tuned model
        >>> model = VisionEncoderDecoderModel.from_pretrained("./vit-bert")
        ```"""

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = transformers.AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = transformers.AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = transformers.AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = False

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = transformers.AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        
        config.is_encoder_decoder = False

        return cls(encoder=encoder, decoder=decoder, config=config)

    def forward(
        self,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        decoder_token_type_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
  
        assert decoder_position_ids is not None
        assert decoder_attention_mask is not None
        assert decoder_attention_mask.dtype == torch.long, f'The dtype for {decoder_attention_mask} was {decoder_attention_mask.dtype}. It should be torch.long'
        assert decoder_token_type_ids is not None

        if decoder_inputs_embeds is None:
            decoder_inputs_embeds = self.decoder.get_input_embeddings()(decoder_input_ids)
        decoder_inputs_embeds += self.token_type_embeddings(decoder_token_type_ids)

        # Generation:
        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Loss:
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        special_token_ids,
        prompt_attention_mask,
        prompt_position_ids,
        token_type_id_sections=None,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        """
        Modification of: 
            https://github.com/huggingface/transformers/blob/main/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L660
        """

        report_attention_mask = (input_ids != self.decoder.config.pad_token_id).long()

        if past_key_values is None:
            
            # 4D attention mask:
            decoder_attention_mask = self.create_4d_attention_mask_mixed_causality(prompt_attention_mask, report_attention_mask)

            # Position identifiers accounting for padding:
            report_position_ids = report_attention_mask.cumsum(-1) + prompt_position_ids.max(dim=1).values[:, None]
            report_position_ids.masked_fill_(report_attention_mask == 0, 1)
            decoder_position_ids = torch.cat([prompt_position_ids, report_position_ids], dim=1)

            # `inputs_embeds` are only to be used in the 1st generation step:
            inputs_embeds = torch.cat([kwargs['decoder_inputs_embeds'], self.decoder.get_input_embeddings()(input_ids)], dim=1)

            decoder_token_type_ids = self.token_ids_to_token_type_ids(input_ids, special_token_ids, token_type_id_sections)
            decoder_token_type_ids = torch.cat(
                [
                    kwargs['decoder_token_type_ids'],
                    decoder_token_type_ids,
                ], 
                dim=1,
            )  # Add image token type identifiers.

            input_dict = {
                'decoder_input_ids': input_ids, 
                'decoder_inputs_embeds': inputs_embeds, 
                'decoder_token_type_ids': decoder_token_type_ids,
            }
        else:
            
            # 4D attention mask:
            decoder_attention_mask = self.create_4d_attention_mask_mixed_causality_past_key_values(prompt_attention_mask, report_attention_mask)

            # Position identifiers accounting for padding:
            decoder_position_ids = report_attention_mask.cumsum(-1) + prompt_position_ids.max(dim=1).values[:, None]
            decoder_position_ids.masked_fill_(report_attention_mask == 0, 1)
            
            # Always place token_ids_to_token_type_ids_past_key_values before input_ids = input_ids[:, remove_prefix_length:]:
            decoder_token_type_ids = self.token_ids_to_token_type_ids_past_key_values(input_ids, special_token_ids, token_type_id_sections)
            decoder_position_ids = decoder_position_ids[:, -1:]

            past_length = past_key_values[0][0].shape[2]

            # Some generation methods only pass the last input ID:
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Keep only the final ID:
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

            input_dict = {'decoder_input_ids': input_ids, 'decoder_token_type_ids': decoder_token_type_ids}

        input_dict.update(
            {
                'decoder_attention_mask': decoder_attention_mask,
                'decoder_position_ids': decoder_position_ids,
                'past_key_values': past_key_values,
                'use_cache': use_cache,
            }
        )
        return input_dict
        
    def token_ids_to_token_type_ids(self, token_ids, special_token_ids, token_type_id_sections=None):
        """
        Extract token type identifiers from the token identifiers.

        Argument/s:
            token_ids - token identifiers.
            special_token_ids - special token identifiers that indicate the separation between sections.
            token_type_id_section - token type identifier for each section.

        Returns:
            token_type_ids - token type identifiers.
        """

        token_type_id_sections = token_type_id_sections if token_type_id_sections is not None else list(range(len(special_token_ids) + 1))

        mbatch_size, seq_len = token_ids.shape
        token_type_ids = torch.full_like(token_ids, token_type_id_sections[0], dtype=torch.long, device=token_ids.device)

        for i, j in enumerate(special_token_ids):
            # Find first occurrence of special tokens that indicate the boundary between sections:
            cols = (token_ids == j).int().argmax(dim=1)
            rows = torch.arange(mbatch_size, device=token_ids.device)

            # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer.create_token_type_ids_from_sequences.example
            cols += 1

            # Ensure that the column index is not out of bounds. If 0, then token_id not present.
            # This is safe as index 0 is always a special token (now equal to 1 due to +1):
            rows = rows[torch.logical_and(cols != 1, cols < seq_len)]
            cols = cols[torch.logical_and(cols != 1, cols < seq_len)]

            # Indices to that correspond to the second sequence:
            if rows.nelement() != 0:
                ids = torch.stack([
                    torch.stack([x, z]) for (x, y) in zip(rows, cols) for z in torch.arange(
                        y, seq_len, device=token_ids.device,
                    )
                ])

                token_type_ids[ids[:, 0], ids[:, 1]] = token_type_id_sections[i + 1]

        return token_type_ids

    def token_ids_to_token_type_ids_past_key_values(self, token_ids, special_token_ids, token_type_id_sections=None):
        """
        Extract token type identifiers from the token identifiers if past != None. Make sure to input all the
        token_ids (e.g., do not input input_ids = input_ids[:, remove_prefix_length:] from prepare_inputs_for_generation).

        Argument/s:
            token_ids - token identifiers.
            special_token_ids - special token identifiers that indicate the separation between sections.

        Returns:
            token_type_ids - token type identifiers.
        """

        token_type_id_sections = token_type_id_sections if token_type_id_sections is not None else list(range(len(special_token_ids) + 1))
        token_type_ids = torch.full([token_ids.shape[0], 1], token_type_id_sections[0], dtype=torch.long, device=token_ids.device)

        # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer.create_token_type_ids_from_sequences.example
        token_ids = token_ids[:, :-1]

        for i, j in enumerate(special_token_ids):

            # Find first occurrence of special token, which indicates the boundary between sections:
            exists = torch.any(token_ids == j, dim=1, keepdim=True)
            token_type_ids[exists] = token_type_id_sections[i + 1]

        return token_type_ids
    
    def tokenize_report_teacher_forcing(self, findings: str, impression: str, tokenizer: PreTrainedTokenizerFast, max_len: int):
        """
        Tokenize the reports and creates the inputs and targets for teacher forcing.

        Argument/s:
            findings - findings sections.
            impression - impression sections.
            return_token_type_ids - return the token type identifiers.
            tokenizer - Hugging Face tokenizer.
            max_len - maximum number of tokens.

        Returns:
            decoder_input_ids - the token identifiers for the input of the decoder.
            decoder_attention_mask - the attention mask for the decoder_input_ids.
            label_ids - the label token identifiers for the decoder.
        """

        # Prepare the sections for the tokenizer by placing special tokens between each section:
        reports = [f'{tokenizer.bos_token}{i}{tokenizer.sep_token}{j}{tokenizer.eos_token}' for i, j in
                  zip(findings, impression)]

        # Tokenize the report:
        tokenized = tokenizer(
            reports,
            padding='longest',
            truncation=True,
            max_length=max_len + 1,  # +1 to account for the bias between input and target.
            return_tensors='pt',
            return_token_type_ids=False,
            add_special_tokens=False,
        ).to(self.device)

        # Modify for language modelling:
        batch_dict = {

            # Labels for the decoder (shifted right by one for autoregression):
            'label_ids': tokenized['input_ids'][:, 1:].detach().clone(),

            # Remove last token identifier to match the sequence length of the labels:
            'decoder_input_ids': tokenized['input_ids'][:, :-1],

            # Attention mask for the decoder_input_ids (remove first token so that the eos_token_id is not considered):
            'decoder_attention_mask': tokenized['attention_mask'][:, 1:],
        }

        return batch_dict

    def tokenize_report_teacher_forcing_rev_a(self, tokenizer: PreTrainedTokenizerFast, max_len: int, findings: Optional[str] = None, impression: Optional[str] = None, reports: Optional[str] = None):
        """
        Tokenize the reports and creates the inputs and targets for teacher forcing.

        Argument/s:
            tokenizer - Hugging Face tokenizer.
            max_len - maximum number of tokens.
            findings - findings sections.
            impression - impression sections.
            reports - prepared reports, with special tokens and report sections.

        Returns:
            decoder_input_ids - the token identifiers for the input of the decoder.
            decoder_attention_mask - the attention mask for the decoder_input_ids.
            label_ids - the label token identifiers for the decoder.
        """

        # Prepare the sections for the tokenizer by placing special tokens between each section:
        if reports is None:
            assert findings and impression, "If 'reports' is not defined, 'findings' and 'impression' need to be defined." 
            reports = [f'{tokenizer.bos_token}{i}{tokenizer.sep_token}{j}{tokenizer.eos_token}' for i, j in
                    zip(findings, impression)]

        # Tokenize the report:
        tokenized = tokenizer(
            reports,
            padding='longest',
            truncation=True,
            max_length=max_len + 1,  # +1 to account for the bias between input and target.
            return_tensors='pt',
            return_token_type_ids=False,
            add_special_tokens=False,
        ).to(self.device)

        # Modify for language modelling:
        batch_dict = {

            # Labels for the decoder (shifted right by one for autoregression):
            'label_ids': tokenized['input_ids'][:, 1:].detach().clone(),

            # Remove last token identifier to match the sequence length of the labels:
            'decoder_input_ids': tokenized['input_ids'][:, :-1],

            # Attention mask for the decoder_input_ids (remove first token so that the eos_token_id is not considered):
            'decoder_attention_mask': tokenized['attention_mask'][:, 1:],
        }

        return batch_dict

    def split_and_decode_sections(self, token_ids, special_token_ids, tokenizer: PreTrainedTokenizerFast):
        """
        Split the token identifiers into sections, then convert the token identifiers into strings.

        Argument/s:
            token_ids - token identifiers.
            special_token_ids - special token identifiers that indicate the end of each section.
            tokenizer - Hugging Face tokenizer.

        Returns:
            token_type_ids - token type identifiers.
        """

        _, seq_len = token_ids.shape

        # The number of sections is the same as the number of special_token_ids:
        num_sections = len(special_token_ids)

        sections = {k: [] for k in range(num_sections)}

        for i in token_ids:
            prev_col = 0
            for j, k in enumerate(special_token_ids):

                # The maximum sequence length was exceeded, thus no more tokens:
                if prev_col >= seq_len:
                    sections[j].append('')
                    continue

                # Find first occurrence of special tokens that indicate the boundary between sections:
                col = (i == k).int().argmax().item()

                # If equal to 0, token was not found, set the column to the sequence length (as the decoder exceeded
                # the maximum sequence length):
                if col == 0:
                    col = seq_len

                # Extract section token identifiers:
                section_token_ids = i[prev_col:col]
                prev_col = col
                section_string = tokenizer.decode(section_token_ids, skip_special_tokens=True)

                sections[j].append(section_string)

        return tuple(sections.values())
    
    def tokenize_text_columns(self, tokenizer: PreTrainedTokenizerFast, **kwargs):
        """
        Tokenize the text columns from MIMIC-IV ED and MIMIC-CXR (excluding the findings and impression sections).
        Time deltas for the input_ids are also prepared here.

        Argument/s:
            tokenizer - Hugging Face tokenizer.

        Returns:
            ed - dictionary containing the input_ids, token_type_ids, attention_mask and time_deltas for the ED module columns.
            cxr - dictionary containing the input_ids, token_type_ids, and attention_mask for MIMIC-CXR columns.
        """

        batch_size = len(kwargs['index'])

        tokenized = {
            'input_ids': {i: [] for i in range(batch_size)},
            'token_type_ids': {i: [] for i in range(batch_size)},
            'time_delta': {i: [] for i in range(batch_size)},
            'attention_mask': torch.empty(batch_size, 0, 1, device=self.device),
        }
        
        for i in self.decoder.config.ed_module_columns + self.decoder.config.mimic_cxr_columns + ['previous_findings', 'previous_impression']: 
            if i in kwargs:
                if f'{i}_time_delta' not in kwargs:
                    kwargs[f'{i}_time_delta'] = [[self.decoder.config.zero_time_delta_value for _ in j] if j is not None else None for j in kwargs[i]]
                for x, (y, z) in enumerate(zip(kwargs[i], kwargs[f'{i}_time_delta'])):
                    if y is not None:
                        assert isinstance(y, list)
                        assert isinstance(z, list)
                        for text, time_delta in zip(y, z):
                            tokenized['input_ids'][x].append(
                                tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device=self.device)
                            )
                            tokenized['token_type_ids'][x].append(
                                torch.full(
                                    (1, tokenized['input_ids'][x][-1].shape[-1]), 
                                    self.decoder.config.token_type_to_token_type_id[i], 
                                    dtype=torch.long,
                                    device=self.device,
                                )
                            )
                            tokenized['time_delta'][x].append(
                                torch.full(
                                    (1, tokenized['input_ids'][x][-1].shape[-1]), 
                                    time_delta, 
                                    dtype=torch.float32,
                                    device=self.device,
                                )
                            )

        tokenized['input_ids'] = [torch.cat(j, dim=1).T if j else torch.empty(0, 1, dtype=torch.long, device=self.device) for j in tokenized['input_ids'].values()]
        tokenized['token_type_ids'] = [torch.cat(j, dim=1).T if j else torch.empty(0, 1, dtype=torch.long, device=self.device) for j in tokenized['token_type_ids'].values()]
        tokenized['time_delta'] = [torch.cat(j, dim=1).T if j else torch.empty(0, 1, device=self.device) for j in tokenized['time_delta'].values()]

        tokenized['input_ids'] = torch.nn.utils.rnn.pad_sequence(
            tokenized['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
        )[:, :, 0]
        tokenized['token_type_ids'] = torch.nn.utils.rnn.pad_sequence(
            tokenized['token_type_ids'], batch_first=True, padding_value=0,
        )[:, :, 0]

        tokenized['attention_mask'] = (tokenized['input_ids'] != tokenizer.pad_token_id).int()
        
        tokenized['time_delta'] = torch.nn.utils.rnn.pad_sequence(
            tokenized['time_delta'], batch_first=True, padding_value=0,
        )

        return tokenized
    
    def prepare_inputs(
        self, 
        images, 
        tokenizer: PreTrainedTokenizerFast, 
        tokenized_report=None, 
        sep_token_id=None, 
        section_ids=None, 
        **batch,
    ):
        """
        Tokenize the text columns from MIMIC-IV ED and MIMIC-CXR (excluding the findings and impression sections).

        Argument/s:
            images - images.
            tokenizer - Hugging Face tokenizer.
            tokenized_report - if training/teacher forcing, input the tokenized_report dict to include it in the prepared inputs.
            separator_token_id - separator token identifier.
            section_ids - section identifiers for the findings and impression sections.

        Returns:
            inputs_embeds - input embeddings.
            attention_mask - attention mask.
            token_type_ids - token type identifiers.
            position_ids - position identifiers.
            bos_token_ids - bos_token_ids for generation.
        """

        input_ids = []
        inputs_embeds = []
        token_type_ids = []
        attention_mask = []
        time_delta = []
        position_ids = None
        bos_token_ids = None

        # Index and value columns:
        batch_size = len(batch['index'])
        for k in self.decoder.config.index_value_encoder_config.keys():
            if f'{k}_index_value_feats' not in batch:
                batch[f'{k}_index_value_feats'] = torch.empty(batch_size, 0, self.decoder.config.index_value_encoder_config[k], device=self.device)
            inputs_embeds.append(
                getattr(self, f'{k}_index_value_encoder')(batch[f'{k}_index_value_feats'])
            )
            token_type_ids.append(batch[f'{k}_index_value_token_type_ids'] if f'{k}_index_value_token_type_ids' in batch else torch.empty(batch_size, 0, dtype=torch.long, device=self.device))
            attention_mask.append(batch[f'{k}_index_value_mask'] if f'{k}_index_value_mask' in batch else torch.empty(batch_size, 0, dtype=torch.long, device=self.device))
            if f'{k}_time_delta' in batch:
                time_delta.append(batch[f'{k}_time_delta'])
            else:
                time_delta_index_value = torch.zeros(*batch[f'{k}_index_value_mask'].shape, 1, device=self.device) if f'{k}_index_value_mask' in batch else torch.empty(batch_size, 0, 1, device=self.device)
                time_delta.append(time_delta_index_value)    

        # Tokenize text columns for prompt:
        tokenized = self.tokenize_text_columns(tokenizer, **batch)
        input_ids.append(tokenized['input_ids'])
        token_type_ids.append(tokenized['token_type_ids'])
        attention_mask.append(tokenized['attention_mask'])
        time_delta.append(tokenized['time_delta'])

        # Image encoder:
        encoder_outputs = self.encoder(images)  
        inputs_embeds.append(encoder_outputs[0])
        inputs_per_image = encoder_outputs[0].shape[-2] // images.shape[1]
        padded_image_time_deltas = [i + [self.decoder.config.zero_time_delta_value] * (images.shape[1] - len(i)) for i in batch['image_time_deltas']]
        time_delta_image_features = torch.tensor(padded_image_time_deltas, device=self.device).repeat_interleave(inputs_per_image, dim=1)
        token_type_ids.append(
            torch.where(
                time_delta_image_features == self.decoder.config.zero_time_delta_value, 
                self.decoder.config.token_type_to_token_type_id['image'],
                self.decoder.config.token_type_to_token_type_id['previous_image'],
            ),
        )
        attention_mask.append(encoder_outputs[1])
        time_delta.append(time_delta_image_features[:, :, None])

        # Compute embeddings from token identifiers:
        input_ids = torch.cat(input_ids, dim=1)
        inputs_embeds.append(self.decoder.get_input_embeddings()(input_ids))
        
        # Concatentate time deltas and input embeddings before adding time delta embedding to prompt:
        time_delta = torch.cat(time_delta, dim=1)
        inputs_embeds = torch.cat(inputs_embeds, dim=1)

        # Add time delta embeddings to prompt:
        if time_delta.shape[1] > 0 and self.decoder.config.add_time_deltas:
            time_delta = time_delta.to(dtype=inputs_embeds.dtype)
            inputs_embeds += self.time_delta_encoder(time_delta)
            
        # Concatentate the attention mask:
        attention_mask = torch.cat(attention_mask, dim=1)
        
        # Position identifiers:   
        position_ids = self.position_ids_from_time_deltas_and_attention_mask(time_delta, attention_mask)
    
        # Tokenize report:
        if tokenized_report is not None:
            inputs_embeds = torch.cat([inputs_embeds, self.decoder.get_input_embeddings()(tokenized_report['decoder_input_ids'])], dim=1)
            
            report_token_type_ids = self.token_ids_to_token_type_ids(
                token_ids=tokenized_report['decoder_input_ids'], 
                special_token_ids=[sep_token_id],
                token_type_id_sections=section_ids,
            )
            token_type_ids.append(report_token_type_ids)
           
            # Position identifiers accounting for padding:
            report_position_ids = tokenized_report['decoder_attention_mask'].cumsum(-1) + position_ids.max(dim=1).values[:, None]
            report_position_ids.masked_fill_(tokenized_report['decoder_attention_mask'] == 0, 1)
            position_ids = torch.cat([position_ids, report_position_ids], dim=1)
            
            # 4D attention mask:
            attention_mask = self.create_4d_attention_mask_mixed_causality(attention_mask, tokenized_report['decoder_attention_mask'])
            # attention_mask_diagonal = torch.diagonal(attention_mask[:, 0], dim1=1, dim2=2)

        else:
            
            # BOS token identifiers for inference/generation:
            bos_token_ids = torch.full((encoder_outputs[0].shape[0], 1), tokenizer.bos_token_id, dtype=torch.long, device=self.device) 
            
        # Concatentate the token type identifiers:
        token_type_ids = torch.cat(token_type_ids, dim=1)

        assert inputs_embeds.shape[1] == attention_mask.shape[-1]
        assert inputs_embeds.shape[1] == token_type_ids.shape[1]

        return inputs_embeds, attention_mask, token_type_ids, position_ids, bos_token_ids
    
    @staticmethod
    def create_4d_attention_mask_mixed_causality(non_causal_2d_attention_mask, causal_2d_attention_mask):
    
        prompt_seq_len = non_causal_2d_attention_mask.shape[-1] 
        report_seq_len = causal_2d_attention_mask.shape[-1]
        
        non_causal_2d_attention_mask = non_causal_2d_attention_mask[:, None, None, :]
        causal_2d_attention_mask = causal_2d_attention_mask[:, None, None, :]
    
        # Upper left of attention matrix:
        upper_left = non_causal_2d_attention_mask.expand(-1, -1, prompt_seq_len, -1)
        upper_left = upper_left * non_causal_2d_attention_mask
        upper_left = upper_left * non_causal_2d_attention_mask.permute(0, 1, 3, 2)
        
        causal_mask = torch.tril(
            torch.ones(
                (
                    report_seq_len, 
                    report_seq_len,
                ), 
                dtype=torch.long, 
                device=causal_2d_attention_mask.device,
            ),
        )   
        
        # Lower right of attention matrix:
        lower_right = causal_2d_attention_mask.expand(-1, -1, report_seq_len, -1)
        lower_right = lower_right * causal_2d_attention_mask.permute(0, 1, 3, 2)
        lower_right = lower_right * causal_mask
        
        # Upper right of attention matrix:
        upper_right = torch.zeros(
            causal_2d_attention_mask.shape[0], 
            1, 
            prompt_seq_len, 
            report_seq_len, 
            dtype=torch.long, 
            device=causal_2d_attention_mask.device,
        )
        
        # Lower left of attention matrix:
        lower_left = non_causal_2d_attention_mask.expand(-1, -1, report_seq_len, -1)
        lower_left = lower_left * causal_2d_attention_mask.permute(0, 1, 3, 2)
            
        left = torch.cat((upper_left, lower_left), dim=2)
        right = torch.cat((upper_right, lower_right), dim=2)

        mixed_causality_4d_attention_mask = torch.cat((left, right), dim=-1)
        return mixed_causality_4d_attention_mask
    
    @staticmethod
    def create_4d_attention_mask_mixed_causality_past_key_values(non_causal_2d_attention_mask, causal_2d_attention_mask):
    
        non_causal_2d_attention_mask = non_causal_2d_attention_mask[:, None, None, :]
        causal_2d_attention_mask = causal_2d_attention_mask[:, None, None, :]

        mixed_causality_4d_attention_mask = torch.cat((non_causal_2d_attention_mask, causal_2d_attention_mask), dim=-1)
        return mixed_causality_4d_attention_mask
    
    def position_ids_from_time_deltas_and_attention_mask(self, time_deltas, attention_mask):
        _, col_indices = torch.sort(torch.where(attention_mask == 1, time_deltas[:, :, 0], torch.finfo(time_deltas.dtype).min), descending=not self.decoder.config.time_delta_monotonic_inversion)

        num_rows, num_cols, _ = time_deltas.shape

        row_indices = torch.arange(num_rows, device=time_deltas.device).view(-1, 1).repeat(1, num_cols).view(-1)
        position_ids = torch.zeros_like(col_indices, device=time_deltas.device)
        position_ids[row_indices, col_indices.flatten()] = torch.arange(num_cols, device=time_deltas.device)[None, :].expand(num_rows, -1).flatten()
        position_ids.masked_fill_(attention_mask == 0, 1)  # Following: https://github.com/huggingface/transformers/blob/c5f0288bc7d76f65996586f79f69fba8867a0e67/src/transformers/models/llama/modeling_llama.py#L1285
        
        return position_ids
    
    @staticmethod
    def prepare_data(physionet_dir, database_path, dataset_dir=None):
        
        dataset_dir = physionet_dir if dataset_dir is None else dataset_dir
        
        sectioned_dir = os.path.join(dataset_dir, 'mimic_cxr_sectioned')

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
            
        if not os.path.exists(database_path):
            
            connect = duckdb.connect(database_path)

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
                assert i in base_names, f"""CSV file {i} is missing from MIMIC-IV-ED.
                    Please download the tables from https://physionet.org/content/mimic-cxr-jpg. Do not decompress them."""

            for i in csv_paths:
                name = Path(i).stem.replace('.csv', '').replace('.gz', '').replace('-', '_').replace('.', '_')
                print(f'Copying {name} into database...')  
                connect.sql(f"CREATE OR REPLACE TABLE {name} AS FROM '{i}';")         

            # MIMIC-CXR report sections:
            print(f'Copying mimic_cxr_sectioned into database...')  
            connect.sql(f"CREATE OR REPLACE TABLE mimic_cxr_sectioned AS FROM '{mimic_cxr_sectioned_path}';")   
            connect.sql("ALTER TABLE mimic_cxr_sectioned RENAME COLUMN column0 TO study;")
            connect.sql("ALTER TABLE mimic_cxr_sectioned RENAME COLUMN column1 TO impression;")
            connect.sql("ALTER TABLE mimic_cxr_sectioned RENAME COLUMN column2 TO findings;")
            connect.sql("ALTER TABLE mimic_cxr_sectioned RENAME COLUMN column3 TO indication;")
            connect.sql("ALTER TABLE mimic_cxr_sectioned RENAME COLUMN column4 TO history;")
            connect.sql("ALTER TABLE mimic_cxr_sectioned RENAME COLUMN column5 TO last_paragraph;")
            connect.sql("ALTER TABLE mimic_cxr_sectioned RENAME COLUMN column6 TO comparison;")
            connect.sql("DELETE FROM mimic_cxr_sectioned WHERE study='study';")

            splits = connect.sql("FROM mimic_cxr_2_0_0_split").df()
            reports = connect.sql("FROM mimic_cxr_sectioned").df()
            metadata = connect.sql("FROM mimic_cxr_2_0_0_metadata").df()
            chexpert = connect.sql("FROM mimic_cxr_2_0_0_chexpert").df()

            # Create datetime column:
            metadata['StudyTime'] = metadata['StudyTime'].astype(int)
            metadata['study_datetime'] = pd.to_datetime(
                metadata.apply(lambda x: f'{x["StudyDate"]} {x["StudyTime"]:06}', axis=1), 
                format='%Y%m%d %H%M%S',
            )
            reports.rename(columns={'study': 'study_id'}, inplace=True)
            reports.study_id = reports.study_id.str[1:].astype('int32')
            df = pd.merge(splits, reports, on='study_id')
            df = pd.merge(df, metadata, on=['dicom_id', 'study_id', 'subject_id'])
            df = pd.merge(df, chexpert, on=['study_id', 'subject_id'])

            connect.sql(f"CREATE OR REPLACE TABLE mimic_cxr AS SELECT * FROM df")
        
            # Create lookup tables:
            for k, v in (ed_module_tables | mimic_cxr_tables).items():
                if v.load and v.index_columns:
                    start_idx = 0
                    for i in v.index_columns_source:
                        lut_name = f'{k}_{i}_lut'
                        table = k
                        lut, end_idx = create_lookup_table(connect.sql(f"SELECT {i} FROM {table}").df(), [i], start_idx)
                        start_idx = end_idx + 1
                        lut = lut.rename(columns={'index': f'{i}_index'})

                        print(f'Creating {lut_name}...')

                        connect.sql(f"CREATE OR REPLACE TABLE {lut_name} AS SELECT * FROM lut")

                        if f'{i}_index' in connect.sql(f"FROM {k} LIMIT 0").df().columns:
                            connect.sql(
                                f"""
                                ALTER TABLE {k}
                                DROP COLUMN {i}_index;
                                """
                            )

                        connect.sql(
                            f"""
                                CREATE OR REPLACE TABLE {k} AS
                                SELECT {k}.*, {lut_name}.{i}_index
                                FROM {k} LEFT JOIN {lut_name}
                                ON {k}.{i} = {lut_name}.{i}
                            """
                        )
                
                    connect.sql(
                        f"""
                            CREATE TABLE IF NOT EXISTS lut_info (table_name VARCHAR PRIMARY KEY, start_index INT, end_index INT);
                            INSERT OR REPLACE INTO lut_info VALUES ('{k}', {0}, {end_idx});
                        """
                    )
                    
            table_studies = {
                'edstays': [],
                'triage': [],
                'medrecon': [],
                'vitalsign': [],
                'pyxis': [],
            }
            stay_id_tables = ['triage']
            stay_id_charttime_tables = ['medrecon', 'vitalsign', 'pyxis']

            df = connect.sql(f"FROM mimic_cxr").df()
            
            # DICOM identifiers can have different datetimes, so use most recent datetime for the study:
            df = df.sort_values(by='study_datetime', ascending=False)
            df = df.groupby('study_id').first().reset_index()

            for _, row in tqdm(df.iterrows(), total=df.shape[0]):
                edstays = connect.sql(
                    f"""
                    SELECT stay_id, intime, outtime
                    FROM edstays 
                    WHERE (subject_id = {row['subject_id']})
                    AND intime < '{row['study_datetime']}'
                    AND outtime > '{row['study_datetime']}';
                    """
                ).df()

                if len(edstays) > 0:

                    for i in edstays['stay_id'].to_list():
                        table_studies['edstays'].append({'study_id': row['study_id'], 'stay_id': i})
                        for j in stay_id_tables:
                            table = connect.sql(
                                f"""
                                SELECT stay_id
                                FROM {j} 
                                WHERE (stay_id = {i});
                                """
                            ).df()
                            
                            for k in table['stay_id'].to_list():
                                table_studies[j].append({'study_id': row['study_id'], 'stay_id': k})

                        for j in stay_id_charttime_tables:
                            table = connect.sql(
                                f"""
                                SELECT stay_id
                                FROM {j} 
                                WHERE (stay_id = {i})
                                AND charttime < '{row['study_datetime']}';
                                """
                            ).df()

                            for k in table['stay_id'].to_list():
                                table_studies[j].append({'study_id': row['study_id'], 'stay_id': k})

            for k, v in table_studies.items():
                df = pd.DataFrame(v)
                df = df.drop_duplicates(subset=['study_id', 'stay_id'])
                connect.sql(f"CREATE TABLE {k}_study_ids AS SELECT * FROM df")

    @staticmethod                
    def get_dataset(split, transforms, database_path, mimic_cxr_jpg_dir, max_images_per_study=5, records=None):
        
        if records is None:
            
            # This is the setup for CXRs + all effective inputs - medicine reconciliation:
            records = EDCXRSubjectRecords(database_path=database_path, time_delta_map=lambda x: 1 / math.sqrt(x + 1)) 
            
            records.ed_module_tables = {k: records.ed_module_tables[k] for k in ['edstays', 'triage', 'vitalsign']}
            records.mimic_cxr_tables = {k: records.mimic_cxr_tables[k] for k in ['mimic_cxr_sectioned']}
            records.mimic_cxr_tables['mimic_cxr_sectioned'].text_columns = ['indication', 'history']
        
        dataset = StudyIDEDStayIDSubset(
                mimic_iv_duckdb_path=database_path,
                dataset_dir=mimic_cxr_jpg_dir,
                transforms=transforms,
                split=split,
                max_images_per_study=max_images_per_study,
                records=records,
            )
        print(f'No. of examples: {dataset.__len__()}.')
        print(
            f'No. of training dicom_ids, study_ids, & subject_ids: {dataset.num_dicom_ids},',
            f'{dataset.num_study_ids}, & {dataset.num_subject_ids}.',
        )
        return dataset

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