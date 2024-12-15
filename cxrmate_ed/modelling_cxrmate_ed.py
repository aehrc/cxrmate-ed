import json
import math
import os
import random
from typing import Optional, Tuple, Union

import datasets
import torch
import transformers
from huggingface_hub import hf_hub_download
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset
from torchvision.io import decode_image
from torchvision.transforms import v2
from transformers import PreTrainedTokenizerFast
from transformers.modeling_outputs import ModelOutput, Seq2SeqLMOutput
from transformers.utils import check_min_version, logging

from .configuration_cxrmate_ed import CXRMateEDConfig
from .dataset import PriorsDataset
from .prepare_dataset import prepare_dataset
from .utils import compute_time_delta

logger = logging.get_logger(__name__)      

# Ordered by oblique, lateral, AP, and then PA views so that PA views are closest in position to the generated tokens (and oblique is furtherest).
VIEW_ORDER = [None, 'LPO', 'RAO', 'LAO', 'SWIMMERS', 'XTABLE LATERAL', 'LL', 'LATERAL',  'AP AXIAL', 'AP RLD', 'AP LLD', 'AP', 'PA RLD', 'PA LLD', 'PA']


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


class ProjectionHead(torch.nn.Module):

    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()

        # Layer normalisation before projection:
        self.layer_norm = torch.nn.LayerNorm(input_size, eps=1e-6)

        # No bias as following layer normalisation with bias:
        self.projection = torch.nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.projection(x)
        return x


class CXRStudyImagesEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder_config):
        super().__init__()

        self.encoder = encoder
        self.config = encoder.config
        self.adapter = ProjectionHead(self.config.embed_dim[-1], decoder_config.hidden_size)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Flatten the batch and study_id dimensions:
        assert len(pixel_values.shape) == 5, 'pixel_values must be B, S, C, H, W, where S is the max number of images for a study in the batch.'
        last_hidden_state = self.encoder(pixel_values.view(-1, *pixel_values.shape[2:])).last_hidden_state

        # Flatten h x w:
        last_hidden_state = torch.flatten(last_hidden_state, 2) if last_hidden_state.dim() > 3 else last_hidden_state
        
        # Project the features for each spatial position to the decoder's hidden size using the adapter network:
        last_hidden_state = self.adapter(last_hidden_state)
        
        # Concatenate the features for each chest X-ray:
        last_hidden_state = last_hidden_state.view(pixel_values.shape[0], -1, last_hidden_state.shape[-1])

        # Derive the attention mask from the pixel values:
        mask = (pixel_values[:, :, 0, 0, 0] != 0.0)[:, :, None]
        attention_mask = torch.ones(
            [last_hidden_state.shape[0], pixel_values.shape[1], last_hidden_state.shape[1] // pixel_values.shape[1]], 
            dtype=torch.long,
            device=mask.device,
        )
        attention_mask = attention_mask * mask
        attention_mask = attention_mask.view(attention_mask.shape[0], -1)

        if not return_dict:
            return last_hidden_state

        return ModelOutput(last_hidden_state=last_hidden_state, attention_mask=attention_mask)


class CXRMateEDModel(transformers.LlavaForConditionalGeneration):

    config_class = CXRMateEDConfig

    def __init__(self, config: CXRMateEDConfig):
        
        check_min_version("4.46.0.dev0")

        super(transformers.LlavaPreTrainedModel, self).__init__(config)
        
        self.config = config
        
        self.vocab_size = config.text_config.vocab_size
        
        self.image_encoder = transformers.AutoModel.from_config(self.config.vision_config, trust_remote_code=True)
        
        self.language_model = transformers.AutoModelForCausalLM.from_config(
            config.text_config,
            attn_implementation=config._attn_implementation,
        )
        
        self.image_encoder = CXRStudyImagesEncoder(self.image_encoder, config.text_config)
    
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables.json')
        path = path if os.path.exists(path) else hf_hub_download(repo_id='aehrc/cxrmate-ed', filename='tables.json')
        with open(path, 'r') as f:
            self.tables = json.load(f)
        
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lookup_tables.json')
        path = path if os.path.exists(path) else hf_hub_download(repo_id='aehrc/cxrmate-ed', filename='lookup_tables.json')
        with open(path, 'r') as f:
            self.luts = json.load(f)
        
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'token_type_ids.json')
        path = path if os.path.exists(path) else hf_hub_download(repo_id='aehrc/cxrmate-ed', filename='token_type_ids.json')
        with open(path, 'r') as f:
            self.token_type_to_token_type_id = json.load(f)

        self.tables = {k: self.tables[k] for k in self.config.tables_filter}
        self.tables['mimic_cxr_sectioned']['text_columns'] = self.config.prompt_report_sections_filter

        for k in self.tables.keys():  
            if self.luts[k]['total'] > 0:        
                setattr(
                    self, 
                    f'{k}_index_value_encoder', 
                    FNNEncoder(
                        num_features=self.luts[k]['total'], 
                        intermediate_size=self.config.index_value_encoder_intermediate_size, 
                        decoder_hidden_size=self.config.text_config.hidden_size,
                    ),
                )
                            
        if self.config.add_time_deltas:
            self.time_delta_encoder = FNNEncoder(
                num_features=1, 
                intermediate_size=self.config.index_value_encoder_intermediate_size, 
                decoder_hidden_size=self.config.text_config.hidden_size,
            )
            
        self.token_type_embeddings = torch.nn.Embedding(max(self.token_type_to_token_type_id.values()) + 1, self.config.text_config.hidden_size)

        self.time_delta_map = lambda x: 1 / math.sqrt(x + 1)
        self.zero_time_delta_value = self.time_delta_map(0)
        
        self.inf_time_delta_value = self.time_delta_map(float('inf'))
        
        # Image transformations:
        self.train_transforms = v2.Compose(
            [
                v2.Grayscale(num_output_channels=3),
                v2.Resize(
                    size=self.config.vision_config.image_size, 
                    antialias=True, 
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomCrop(
                    size=[self.config.vision_config.image_size, self.config.vision_config.image_size],
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
                    size=self.config.vision_config.image_size, 
                    antialias=True,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.CenterCrop(size=[self.config.vision_config.image_size, self.config.vision_config.image_size]),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
        
        self.post_init()

    def forward(
        self,
        decoder_position_ids: torch.LongTensor,
        decoder_attention_mask: torch.FloatTensor,
        decoder_token_type_ids: torch.LongTensor,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
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
  
        if decoder_inputs_embeds is None:
            decoder_inputs_embeds = self.language_model.get_input_embeddings()(decoder_input_ids)
        decoder_inputs_embeds += self.token_type_embeddings(decoder_token_type_ids)

        if decoder_attention_mask.dim() == 4:
            assert decoder_attention_mask.dtype == decoder_inputs_embeds.dtype, f'The dtype for {decoder_attention_mask} was {decoder_attention_mask.dtype}. It should be {decoder_inputs_embeds.dtype}'
        else:
            assert decoder_attention_mask.dtype == torch.long, f'The dtype for {decoder_attention_mask} was {decoder_attention_mask.dtype}. It should be torch.long'

        # Generation:
        decoder_outputs = self.language_model(
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
            loss = loss_fct(logits.reshape(-1, self.vocab_size), labels.reshape(-1))

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
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        """
        Modification of: 
            https://github.com/huggingface/transformers/blob/main/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L660
        """

        report_attention_mask = (input_ids != self.config.pad_token_id).long()

        if len(past_key_values) == 0:
            
            # 4D attention mask:
            decoder_attention_mask = self.create_4d_attention_mask_mixed_causality(
                prompt_attention_mask, report_attention_mask, dtype=kwargs['decoder_inputs_embeds'].dtype,
            )
            
            # Position identifiers accounting for padding:
            report_position_ids = report_attention_mask.cumsum(-1) + prompt_position_ids.max(dim=1).values[:, None]
            report_position_ids.masked_fill_(report_attention_mask == 0, 1)
            decoder_position_ids = torch.cat([prompt_position_ids, report_position_ids], dim=1)

            # `inputs_embeds` are only to be used in the 1st generation step:
            inputs_embeds = torch.cat([kwargs['decoder_inputs_embeds'], self.language_model.get_input_embeddings()(input_ids)], dim=1)

            decoder_token_type_ids = self.token_ids_to_token_type_ids(
                input_ids, special_token_ids, 
                [self.token_type_to_token_type_id['findings'], self.token_type_to_token_type_id['impression']],
            )
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
            decoder_attention_mask = self.create_4d_attention_mask_mixed_causality_past_key_values(
                prompt_attention_mask, report_attention_mask, dtype=kwargs['decoder_inputs_embeds'].dtype,
            )

            # Position identifiers accounting for padding:
            decoder_position_ids = report_attention_mask.cumsum(-1) + prompt_position_ids.max(dim=1).values[:, None]
            decoder_position_ids.masked_fill_(report_attention_mask == 0, 1)
            
            # Always place token_ids_to_token_type_ids_past_key_values before input_ids = input_ids[:, remove_prefix_length:]:
            decoder_token_type_ids = self.token_ids_to_token_type_ids_past_key_values(
                input_ids, 
                special_token_ids, 
                [self.token_type_to_token_type_id['findings'], self.token_type_to_token_type_id['impression']],
            )
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
        
    def token_ids_to_token_type_ids(self, token_ids, special_token_ids, token_type_id_sections):
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

    def token_ids_to_token_type_ids_past_key_values(self, token_ids, special_token_ids, token_type_id_sections):
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

    def prepare_inputs(
        self, 
        images, 
        tokenizer: PreTrainedTokenizerFast, 
        tokenized_report=None, 
        sep_token_id=None, 
        **batch,
    ):
        """
        Tokenize the text columns from MIMIC-IV ED and MIMIC-CXR (excluding the findings and impression sections).

        Argument/s:
            images - images.
            tokenizer - Hugging Face tokenizer.
            tokenized_report - if training/teacher forcing, input the tokenized_report dict to include it in the prepared inputs.
            separator_token_id - separator token identifier.
            
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
        batch_size = images.shape[0]
        for k, v in self.tables.items():
            if 'index_columns' in v or 'value_columns' in v:
                if f'{k}_index_value_feats' not in batch:
                    batch[f'{k}_index_value_feats'] = torch.empty(batch_size, 0, self.luts[k]['total'], device=self.device)
                inputs_embeds.append(
                    getattr(self, f'{k}_index_value_encoder')(batch[f'{k}_index_value_feats'])
                )
                token_type_ids.append(batch[f'{k}_index_value_token_type_ids'] if f'{k}_index_value_token_type_ids' in batch else torch.empty(batch_size, 0, dtype=torch.long, device=self.device))
                attention_mask.append(batch[f'{k}_index_value_mask'] if f'{k}_index_value_mask' in batch else torch.empty(batch_size, 0, dtype=torch.long, device=self.device))
                if f'{k}_index_value_time_delta' in batch:
                    time_delta.append(batch[f'{k}_index_value_time_delta'])
                else:
                    time_delta_index_value = torch.zeros(*batch[f'{k}_index_value_mask'].shape, 1, device=self.device) if f'{k}_index_value_mask' in batch else torch.empty(batch_size, 0, 1, device=self.device)
                    time_delta.append(time_delta_index_value)    

        # Tokenize text columns for prompt:
        tokenized = self.tokenize_text_prompt(tokenizer, **batch)
        input_ids.append(tokenized['input_ids'])
        token_type_ids.append(tokenized['token_type_ids'])
        attention_mask.append(tokenized['attention_mask'])
        time_delta.append(tokenized['time_delta'])

        # Image encoder:
        encoder_outputs = self.image_encoder(images)  
        inputs_embeds.append(encoder_outputs[0])
        
        inputs_per_image = encoder_outputs[0].shape[-2] // images.shape[1]
        time_delta_image_features = torch.tensor(batch['image_time_deltas'], device=self.device).repeat_interleave(inputs_per_image, dim=1)
        token_type_ids.append(
            torch.where(
                torch.logical_or(
                    time_delta_image_features == self.zero_time_delta_value, 
                    time_delta_image_features == self.inf_time_delta_value,
                ),
                self.token_type_to_token_type_id['image'],
                self.token_type_to_token_type_id['prior_image'],
            ),
        )
        attention_mask.append(encoder_outputs[1])
        time_delta.append(time_delta_image_features[:, :, None])

        # Compute embeddings from token identifiers:
        input_ids = torch.cat(input_ids, dim=1)
        inputs_embeds.append(self.language_model.get_input_embeddings()(input_ids))
        
        # Concatentate time deltas and input embeddings before adding time delta embedding to prompt:
        time_delta = torch.cat(time_delta, dim=1)
        inputs_embeds = torch.cat(inputs_embeds, dim=1)

        # Add time delta embeddings to prompt:
        if time_delta.shape[1] > 0 and self.config.add_time_deltas:
            time_delta = time_delta.to(dtype=inputs_embeds.dtype)
            inputs_embeds += self.time_delta_encoder(time_delta)
            
        # Concatentate the attention mask:
        attention_mask = torch.cat(attention_mask, dim=1)
        
        # Position identifiers:   
        position_ids = self.position_ids_from_time_deltas_and_attention_mask(time_delta, attention_mask)
    
        # Tokenize report:
        if tokenized_report is not None:
            inputs_embeds = torch.cat([inputs_embeds, self.language_model.get_input_embeddings()(tokenized_report['decoder_input_ids'])], dim=1)
            
            report_token_type_ids = self.token_ids_to_token_type_ids(
                token_ids=tokenized_report['decoder_input_ids'], 
                special_token_ids=[sep_token_id],
                token_type_id_sections=[self.token_type_to_token_type_id['findings'], self.token_type_to_token_type_id['impression']],
            )
            token_type_ids.append(report_token_type_ids)
           
            # Position identifiers accounting for padding:
            report_position_ids = tokenized_report['decoder_attention_mask'].cumsum(-1) + position_ids.max(dim=1).values[:, None]
            report_position_ids.masked_fill_(tokenized_report['decoder_attention_mask'] == 0, 1)
            position_ids = torch.cat([position_ids, report_position_ids], dim=1)
            
            # 4D attention mask:
            attention_mask = self.create_4d_attention_mask_mixed_causality(attention_mask, tokenized_report['decoder_attention_mask'], dtype=inputs_embeds.dtype)
            # attention_mask = self.create_4d_attention_mask_mixed_causality(attention_mask, tokenized_report['decoder_attention_mask'])
            # attention_mask_diagonal = torch.diagonal(attention_mask[:, 0], dim1=1, dim2=2)

        else:
            
            # BOS token identifiers for inference/generation:
            bos_token_ids = torch.full((encoder_outputs[0].shape[0], 1), tokenizer.bos_token_id, dtype=torch.long, device=self.device) 
            
        # Concatentate the token type identifiers:
        token_type_ids = torch.cat(token_type_ids, dim=1)

        assert inputs_embeds.shape[1] == attention_mask.shape[-1]
        assert inputs_embeds.shape[1] == token_type_ids.shape[1]

        return inputs_embeds, attention_mask, token_type_ids, position_ids, bos_token_ids

    def tokenize_text_prompt(self, tokenizer: PreTrainedTokenizerFast, **kwargs):
        """
        Tokenize the text columns from MIMIC-IV ED and MIMIC-CXR (excluding the findings and impression sections).
        Time deltas for the input_ids are also prepared here.

        Argument/s:
            tokenizer - Hugging Face tokenizer.

        Returns:
            ed - dictionary containing the input_ids, token_type_ids, attention_mask and time_deltas for the ED module columns.
            cxr - dictionary containing the input_ids, token_type_ids, and attention_mask for MIMIC-CXR columns.
        """

        batch_size = len(kwargs['study_id'])

        tokenized = {
            'input_ids': {i: [] for i in range(batch_size)},
            'token_type_ids': {i: [] for i in range(batch_size)},
            'time_delta': {i: [] for i in range(batch_size)},
            'attention_mask': torch.empty(batch_size, 0, 1, device=self.device),
        }
        
        prompt_text_columns = [f'{k}_{j}' if k != 'mimic_cxr_sectioned' else j for k, v in self.tables.items() if 'text_columns' in v for j in (v['text_columns'] if isinstance(v['text_columns'], list) else [v['text_columns']])] + ['prior_findings', 'prior_impression']
        
        for i in prompt_text_columns: 
            if i in kwargs:
                if f'{i}_time_delta' not in kwargs:
                    kwargs[f'{i}_time_delta'] = [[self.zero_time_delta_value for _ in j] if j is not None else None for j in kwargs[i]]
                for x, (y, z) in enumerate(zip(kwargs[i], kwargs[f'{i}_time_delta'])):
                    if y is not None:
                        assert isinstance(y, list)
                        assert isinstance(z, list)
                        for text, time_delta in zip(y, z):
                            if text is not None:
                                tokenized['input_ids'][x].append(
                                    tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device=self.device)
                                )
                                tokenized['token_type_ids'][x].append(
                                    torch.full(
                                        (1, tokenized['input_ids'][x][-1].shape[-1]), 
                                        self.token_type_to_token_type_id[i], 
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
    
    def position_ids_from_time_deltas_and_attention_mask(self, time_deltas, attention_mask):
        mask_value = torch.finfo(time_deltas.dtype).max if self.config.time_delta_monotonic_inversion else torch.finfo(time_deltas.dtype).min
        
        masked_time_deltas = torch.where(attention_mask == 1, time_deltas[:, :, 0], mask_value)
        _, col_indices = torch.sort(masked_time_deltas, descending=not self.config.time_delta_monotonic_inversion)
        
        num_rows, num_cols, _ = time_deltas.shape

        row_indices = torch.arange(num_rows, device=time_deltas.device).view(-1, 1).repeat(1, num_cols).view(-1)
        position_ids = torch.zeros_like(col_indices, device=time_deltas.device)
        position_ids[row_indices, col_indices.flatten()] = torch.arange(num_cols, device=time_deltas.device)[None, :].expand(num_rows, -1).flatten()
        position_ids.masked_fill_(attention_mask == 0, 1)  # Following: https://github.com/huggingface/transformers/blob/c5f0288bc7d76f65996586f79f69fba8867a0e67/src/transformers/models/llama/modeling_llama.py#L1285
        
        return position_ids
    
    def prepare_index_value_feats(self, table, batch):
               
        index_value_columns = (self.tables[table].get('index_columns', []) + self.tables[table].get('value_columns', []))
        index_value_columns = [f'{table}_{i}' for i in index_value_columns] if table != 'mimic_cxr_2_0_0_metadata' else index_value_columns
        
        # Map to indices with lookup table:
        if 'index_columns' in self.tables[table]:
            for i in self.tables[table]['index_columns']:
                k = f'{table}_{i}' if not table == 'mimic_cxr_2_0_0_metadata' else i
                batch[k] = [
                    [self.luts[table][i][str(k)] if k is not None else None for k in j] if j is not None else None for j in batch[k]
                ]
        
        batch_index_value_feats_list = []
        batch_token_type_ids_list = []
        batch_time_deltas_list = []
        
        for batch_idx in range(len(batch['study_id'])):
        
            if any([batch[k][batch_idx] for k in index_value_columns]):

                num_rows = [len(batch[i][batch_idx]) for i in index_value_columns]
                assert all(x == num_rows[0] for x in num_rows)
                num_rows = num_rows[0]

                # The y-index and the datetime for each group:
                if isinstance(batch[self.tables[table]['groupby']][batch_idx], list):            
                    y_indices = [d.setdefault(x, len(d)) for d in [{}] for x in batch[self.tables[table]['groupby']][batch_idx]]
                    datetime = [j for i, j in enumerate(batch[self.tables[table]['time_column']][batch_idx]) if j not in batch[self.tables[table]['time_column']][batch_idx][:i]]
                    assert len(set(y_indices)) == len(datetime)
                else:
                    y_indices = [0] * num_rows
                    datetime = batch[self.tables[table]['time_column']][batch_idx] if 'time_column' in self.tables[table] else [batch['latest_study_datetime'][batch_idx]]
                    
                time_deltas = torch.tensor([compute_time_delta(i, batch['latest_study_datetime'][batch_idx], self.time_delta_map, to_tensor=False) for i in datetime])[:, None]
                                                    
                tensor = torch.zeros(max(y_indices) + 1, self.luts[table]['total'])
                
                # Index columns to feats:
                if 'index_columns' in self.tables[table]:

                    for i in self.tables[table]['index_columns']:
                        k = f'{table}_{i}' if not table == 'mimic_cxr_2_0_0_metadata' else i
                        y_indices_column = [y_idx for y_idx, x_idx in zip(y_indices, batch[k][batch_idx]) if x_idx is not None]
                        x_indices_column = [x_idx for x_idx in batch[k][batch_idx] if x_idx is not None]

                        tensor[y_indices_column, x_indices_column] = 1.0
                        
                if 'value_columns' in self.tables[table]:
                    for i in self.tables[table]['value_columns']:
                        
                        k = f'{table}_{i}' if not table == 'mimic_cxr_2_0_0_metadata' else i
                        y_indices_column = [y_idx for y_idx, value in zip(y_indices, batch[k][batch_idx]) if value is not None]
                        x_indices_column = [self.luts[table][i] for value in batch[k][batch_idx] if value is not None]
                        values = [value for value in batch[k][batch_idx] if value is not None]

                        tensor[y_indices_column, x_indices_column] = torch.tensor(values, dtype=tensor.dtype)
                        assert not torch.isnan(tensor).any()
            else:
                tensor = torch.empty(0, self.luts[table]['total'])
                time_deltas = torch.empty(0, 1)
                
            batch_index_value_feats_list.append(tensor)
            batch_token_type_ids_list.append(torch.full(
                    [tensor.shape[0]], 
                    self.token_type_to_token_type_id[table], 
                    dtype=torch.long, 
                )
            )
            batch_time_deltas_list.append(time_deltas)
            
            assert tensor.shape[0] == batch_token_type_ids_list[-1].shape[0]
            assert tensor.shape[0] == time_deltas.shape[0]
            
        batch_index_value_feats = torch.nn.utils.rnn.pad_sequence(batch_index_value_feats_list, batch_first=True, padding_value=-1)  # Pad value of -1 is not ideal. Need to use something else.
        batch_token_type_ids = torch.nn.utils.rnn.pad_sequence(batch_token_type_ids_list, batch_first=True, padding_value=0)
        batch_time_deltas = torch.nn.utils.rnn.pad_sequence(batch_time_deltas_list, batch_first=True, padding_value=0)

        batch_mask = (batch_index_value_feats != -1).any(dim=-1).int()
                
        return batch_index_value_feats, batch_token_type_ids, batch_time_deltas, batch_mask

    def prepare_text_prompt(self, table, column, batch):

        key = f'{table}_{column}' if not table == 'mimic_cxr_sectioned' else column

        batch_text_list = []
        batch_time_deltas_list = []

        for batch_idx in range(len(batch['study_id'])):
            if batch[key][batch_idx]:

                num_rows = len(batch[key][batch_idx])

                # The y-index and the datetime for each group:
                if isinstance(batch[self.tables[table]['groupby']][batch_idx], list):            
                    y_indices = [d.setdefault(x, len(d)) for d in [{}] for x in batch[self.tables[table]['groupby']][batch_idx]]
                    datetime = [j for i, j in enumerate(batch[self.tables[table]['time_column']][batch_idx]) if j not in batch[self.tables[table]['time_column']][batch_idx][:i]]
                    assert len(set(y_indices)) == len(datetime)
                else:
                    y_indices = [0] * num_rows
                    datetime = batch[self.tables[table]['time_column']][batch_idx] if 'time_column' in self.tables[table] else [batch['latest_study_datetime'][batch_idx]]
                    
                # Remove None values:  
                text_rows = batch[key][batch_idx] if isinstance(batch[key][batch_idx], list) else [batch[key][batch_idx]]                                                         
                y_indices = [i for i, j in zip(y_indices, text_rows) if j is not None]
                text_rows = [i for i in text_rows if i is not None]
                datetime = [datetime[i] for i in set(y_indices)]
                if text_rows:
                                                                   
                    # Those in the same group (or those with the same y-index) get joined as the same string:                
                    batch_text_list.append([', '.join([text_rows[j] for j in range(len(y_indices)) if y_indices[j] == k]) + '.' for k in set(y_indices)])
                    batch_time_deltas_list.append([compute_time_delta(i, batch['latest_study_datetime'][batch_idx], self.time_delta_map, to_tensor=False) for i in datetime])
            
                    assert len(batch_time_deltas_list[-1]) == len(batch_text_list[-1])
                else:
                    batch_text_list.append([])
                    batch_time_deltas_list.append([])
            else:
                batch_text_list.append([])
                batch_time_deltas_list.append([])

        return batch_text_list, batch_time_deltas_list

    @staticmethod
    def create_4d_attention_mask_mixed_causality(non_causal_2d_attention_mask, causal_2d_attention_mask, dtype):
    
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
        
        mixed_causality_4d_attention_mask = mixed_causality_4d_attention_mask.to(dtype=dtype)
        mixed_causality_4d_attention_mask[mixed_causality_4d_attention_mask == 0] = torch.finfo(mixed_causality_4d_attention_mask.dtype).min
        mixed_causality_4d_attention_mask[mixed_causality_4d_attention_mask == 1] = 0.0
        
        return mixed_causality_4d_attention_mask
    
    @staticmethod
    def create_4d_attention_mask_mixed_causality_past_key_values(non_causal_2d_attention_mask, causal_2d_attention_mask, dtype):
    
        non_causal_2d_attention_mask = non_causal_2d_attention_mask[:, None, None, :]
        causal_2d_attention_mask = causal_2d_attention_mask[:, None, None, :]

        mixed_causality_4d_attention_mask = torch.cat((non_causal_2d_attention_mask, causal_2d_attention_mask), dim=-1)
        
        mixed_causality_4d_attention_mask = mixed_causality_4d_attention_mask.to(dtype=dtype)
        mixed_causality_4d_attention_mask[mixed_causality_4d_attention_mask == 0] = torch.finfo(mixed_causality_4d_attention_mask.dtype).min
        mixed_causality_4d_attention_mask[mixed_causality_4d_attention_mask == 1] = 0.0
        
        return mixed_causality_4d_attention_mask

    @staticmethod
    def collate_fn(batch):
        keys = set().union(*(d.keys() for d in batch))
        batch = {j: [i.setdefault(j, None) for i in batch] for j in keys}
        batch = {k: torch.stack(v) if isinstance(v[0], torch.Tensor) else v for k, v in batch.items()}
        return batch
    
    @staticmethod
    def prepare_dataset(physionet_dir: str, database_dir: str):
        
        prepare_dataset(physionet_dir=physionet_dir, database_dir=database_dir)

    def get_dataset(self, database_dir, max_train_images_per_study=None, study_id_split='mimic_iv_ed_mimic_cxr_jpg', test_set_only=False):
        
        dataset_path = os.path.join(database_dir, 'mimic_iv_ed_mimic_cxr_jpg_dataset')
        
        assert max_train_images_per_study is not None or test_set_only, 'max_train_images_per_study must be defined if training.'
        
        def train_set_transform(batch):
            
            # Randomly select max_train_images_per_study if the number of images for a study exceeds max_train_images_per_study.
            keys = ['images', 'dicom_id'] 
            keys = keys + self.tables['mimic_cxr_2_0_0_metadata']['index_columns'] if 'mimic_cxr_2_0_0_metadata' in self.tables else keys
            for i in range(len(batch['images'])):
                if len(batch['images'][i]) > max_train_images_per_study:
                    paired = list(zip(*(batch[key][i] for key in keys)))
                    sampled_pairs = random.sample(paired, max_train_images_per_study)
                    unzipped_samples = zip(*sampled_pairs)
                    for key, values in zip(keys, unzipped_samples):
                        batch[key][i] = list(values)
            
            batch['images'] = [[decode_image(torch.frombuffer(bytearray(j), dtype=torch.uint8)) for j in i] for i in batch['images']]
            
            # Sort based on ViewPosition:
            batch['images'] = [list(zip(*sorted(zip(i, v), key=lambda x: VIEW_ORDER.index(x[1]))))[0] for i, v in zip(batch['images'], batch['ViewPosition'])]
            batch['images'] = [torch.stack([self.train_transforms(j) for j in i]) for i in batch['images']]
            max_size = max(i.shape[0] for i in batch['images'])
            
            batch['image_time_deltas'] = [[self.zero_time_delta_value if j < i.shape[0] else self.inf_time_delta_value for j in range(max_size)] for i in batch['images']]
            batch['images'] = torch.nn.utils.rnn.pad_sequence(batch['images'], batch_first=True, padding_value=0.0)
            
            for k, v in self.tables.items():
                if 'index_columns' in v or 'value_columns' in v:
                    batch[f'{k}_index_value_feats'],  batch[f'{k}_index_value_token_type_ids'], batch[f'{k}_index_value_time_delta'], batch[f'{k}_index_value_mask'] = self.prepare_index_value_feats(k, batch)
            
            for k, v in self.tables.items():
                if 'text_columns' in v:
                    for i in v['text_columns']:
                        key = f'{k}_{i}' if not k == 'mimic_cxr_sectioned' else i
                        batch[key], batch[f'{key}_time_delta'] = self.prepare_text_prompt(k, i, batch)

            return batch

        def test_set_transform(batch):
            batch['images'] = [[decode_image(torch.frombuffer(bytearray(j), dtype=torch.uint8)) for j in i] for i in batch['images']]
            
            # Sort based on ViewPosition:
            batch['images'] = [list(zip(*sorted(zip(i, v), key=lambda x: VIEW_ORDER.index(x[1]))))[0] for i, v in zip(batch['images'], batch['ViewPosition'])]
            batch['images'] = [torch.stack([self.test_transforms(j) for j in i]) for i in batch['images']]
            max_size = max(i.shape[0] for i in batch['images'])
            batch['image_time_deltas'] = [[self.zero_time_delta_value if j < i.shape[0] else self.inf_time_delta_value for j in range(max_size)] for i in batch['images']]
            batch['images'] = torch.nn.utils.rnn.pad_sequence(batch['images'], batch_first=True, padding_value=0.0)
            
            for k, v in self.tables.items():
                if 'index_columns' in v or 'value_columns' in v:
                    batch[f'{k}_index_value_feats'],  batch[f'{k}_index_value_token_type_ids'], batch[f'{k}_index_value_time_delta'], batch[f'{k}_index_value_mask'] = self.prepare_index_value_feats(k, batch)
            
            for k, v in self.tables.items():
                if 'text_columns' in v:
                    for i in v['text_columns']:
                        key = f'{k}_{i}' if not k == 'mimic_cxr_sectioned' else i
                        batch[key], batch[f'{key}_time_delta'] = self.prepare_text_prompt(k, i, batch)
            
            return batch

        dataset = datasets.load_from_disk(dataset_path)

        # Train set:
        if not test_set_only:
            
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{study_id_split}_train_study_ids.json')
            path = path if os.path.exists(path) else hf_hub_download(repo_id='aehrc/cxrmate-ed', filename=f'{study_id_split}_train_study_ids.json')
            with open(path, 'r') as f:
                study_ids = json.load(f)
            train_set = dataset['train']
            train_set_study_ids = train_set['study_id']
            index_map = {study_id: idx for idx, study_id in enumerate(train_set_study_ids)}
            indices = [index_map[study_id] for study_id in study_ids if study_id in index_map]
            indices.sort()        
            train_set = PriorsDataset(train_set, self.config.history, self.time_delta_map)
            train_set.set_transform(train_set_transform)
            train_set = Subset(train_set, indices)
        else:
            train_set = None

        # Validation set:
        if not test_set_only:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{study_id_split}_validate_study_ids.json')
            path = path if os.path.exists(path) else hf_hub_download(repo_id='aehrc/cxrmate-ed', filename=f'{study_id_split}_validate_study_ids.json')
            with open(path, 'r') as f:
                study_ids = json.load(f)
            val_set = dataset['validate']
            val_set_study_ids = val_set['study_id']
            index_map = {study_id: idx for idx, study_id in enumerate(val_set_study_ids)}
            indices = [index_map[study_id] for study_id in study_ids if study_id in index_map]
            indices.sort()    
            val_set = PriorsDataset(val_set, self.config.history, self.time_delta_map)    
            val_set.set_transform(test_set_transform)
            val_set = Subset(val_set, indices)
        else:
            val_set = None

        # Test set:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{study_id_split}_test_study_ids.json')
        path = path if os.path.exists(path) else hf_hub_download(repo_id='aehrc/cxrmate-ed', filename=f'{study_id_split}_test_study_ids.json')
        with open(path, 'r') as f:
            study_ids = json.load(f)
        test_set = dataset['test']
        test_set_study_ids = test_set['study_id']
        index_map = {study_id: idx for idx, study_id in enumerate(test_set_study_ids)}
        indices = [index_map[study_id] for study_id in study_ids if study_id in index_map]
        indices.sort()        
        test_set = PriorsDataset(test_set, self.config.history, self.time_delta_map)    
        test_set.set_transform(test_set_transform)
        test_set = Subset(test_set, indices)
        
        if not test_set_only:
            return train_set, val_set, test_set
        else:
            return test_set
    
    def get_stage_1_dataset(self, database_dir, max_train_images_per_study):
        
        dataset_path = os.path.join(database_dir, 'mimic_iv_ed_mimic_cxr_jpg_dataset')
        
        def train_set_transform(batch):
            
            # Randomly select max_train_images_per_study if the number of images for a study exceeds max_train_images_per_study.
            for i in range(len(batch['images'])):
                if len(batch['images'][i]) > max_train_images_per_study:
                    paired = list(zip(batch['images'][i], batch['ViewPosition'][i]))
                    sampled_pairs = random.sample(paired, max_train_images_per_study)
                    batch['images'][i], batch['ViewPosition'][i] = zip(*sampled_pairs)

            batch['images'] = [[decode_image(torch.frombuffer(bytearray(j), dtype=torch.uint8)) for j in i] for i in batch['images']]
            
            # Sort based on ViewPosition:
            batch['images'] = [list(zip(*sorted(zip(i, v), key=lambda x: VIEW_ORDER.index(x[1]))))[0] for i, v in zip(batch['images'], batch['ViewPosition'])]           
            batch['images'] = [torch.stack([self.train_transforms(j) for j in i]) for i in batch['images']]
            max_size = max(i.shape[0] for i in batch['images'])
            batch['image_time_deltas'] = [[self.zero_time_delta_value if j < i.shape[0] else self.inf_time_delta_value for j in range(max_size)] for i in batch['images']]
            batch['images'] = torch.nn.utils.rnn.pad_sequence(batch['images'], batch_first=True, padding_value=0.0)
            
            return batch

        def test_set_transform(batch):
            batch['images'] = [[decode_image(torch.frombuffer(bytearray(j), dtype=torch.uint8)) for j in i] for i in batch['images']]
            
            # Sort based on ViewPosition:
            batch['images'] = [list(zip(*sorted(zip(i, v), key=lambda x: VIEW_ORDER.index(x[1]))))[0] for i, v in zip(batch['images'], batch['ViewPosition'])]
            batch['images'] = [torch.stack([self.test_transforms(j) for j in i]) for i in batch['images']]
            max_size = max(i.shape[0] for i in batch['images'])
            batch['image_time_deltas'] = [[self.zero_time_delta_value if j < i.shape[0] else self.inf_time_delta_value for j in range(max_size)] for i in batch['images']]
            batch['images'] = torch.nn.utils.rnn.pad_sequence(batch['images'], batch_first=True, padding_value=0.0)
            
            return batch

        dataset = datasets.load_from_disk(dataset_path)

        # Train set:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'mimic_cxr_jpg_train_study_ids.json')
        path = path if os.path.exists(path) else hf_hub_download(repo_id='aehrc/cxrmate-ed', filename='mimic_cxr_jpg_train_study_ids.json')
        with open(path, 'r') as f:
            study_ids = json.load(f)
        train_set = dataset['train']
        train_set_study_ids = train_set['study_id']
        index_map = {study_id: idx for idx, study_id in enumerate(train_set_study_ids)}
        indices = [index_map[study_id] for study_id in study_ids if study_id in index_map]
        indices.sort()        
        train_set = PriorsDataset(train_set, self.config.history, self.time_delta_map)
        train_set.set_transform(train_set_transform)
        train_set = Subset(train_set, indices)

        # Validation set:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'mimic_cxr_jpg_validate_study_ids.json')
        path = path if os.path.exists(path) else hf_hub_download(repo_id='aehrc/cxrmate-ed', filename='mimic_cxr_jpg_validate_study_ids.json')
        with open(path, 'r') as f:
            study_ids = json.load(f)
        val_set = dataset['validate']
        val_set_study_ids = val_set['study_id']
        index_map = {study_id: idx for idx, study_id in enumerate(val_set_study_ids)}
        indices = [index_map[study_id] for study_id in study_ids if study_id in index_map]
        indices.sort()        
        val_set = PriorsDataset(val_set, self.config.history, self.time_delta_map)    
        val_set.set_transform(test_set_transform)
        val_set = Subset(val_set, indices)

        # Test set:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'mimic_cxr_jpg_test_study_ids.json')
        path = path if os.path.exists(path) else hf_hub_download(repo_id='aehrc/cxrmate-ed', filename='mimic_cxr_jpg_test_study_ids.json')
        with open(path, 'r') as f:
            study_ids = json.load(f)
        test_set = dataset['test']
        test_set_study_ids = test_set['study_id']
        index_map = {study_id: idx for idx, study_id in enumerate(test_set_study_ids)}
        indices = [index_map[study_id] for study_id in study_ids if study_id in index_map]
        indices.sort()        
        test_set = PriorsDataset(test_set, self.config.history, self.time_delta_map)    
        test_set.set_transform(test_set_transform)
        test_set = Subset(test_set, indices)
        
        return train_set, val_set, test_set
    
    def get_dataset_all_test_set_studies(self, database_dir):
        
        dataset_path = os.path.join(database_dir, 'mimic_iv_ed_mimic_cxr_jpg_dataset')

        def test_set_transform(batch):
            batch['images'] = [[decode_image(torch.frombuffer(bytearray(j), dtype=torch.uint8)) for j in i] for i in batch['images']]
            
            # Sort based on ViewPosition:
            batch['images'] = [list(zip(*sorted(zip(i, v), key=lambda x: VIEW_ORDER.index(x[1]))))[0] for i, v in zip(batch['images'], batch['ViewPosition'])]
            batch['images'] = [torch.stack([self.test_transforms(j) for j in i]) for i in batch['images']]
            max_size = max(i.shape[0] for i in batch['images'])
            batch['image_time_deltas'] = [[self.zero_time_delta_value if j < i.shape[0] else self.inf_time_delta_value for j in range(max_size)] for i in batch['images']]
            batch['images'] = torch.nn.utils.rnn.pad_sequence(batch['images'], batch_first=True, padding_value=0.0)
            
            for k, v in self.tables.items():
                if 'index_columns' in v or 'value_columns' in v:
                    batch[f'{k}_index_value_feats'],  batch[f'{k}_index_value_token_type_ids'], batch[f'{k}_index_value_time_delta'], batch[f'{k}_index_value_mask'] = self.prepare_index_value_feats(k, batch)
            
            for k, v in self.tables.items():
                if 'text_columns' in v:
                    for i in v['text_columns']:
                        key = f'{k}_{i}' if not k == 'mimic_cxr_sectioned' else i
                        batch[key], batch[f'{key}_time_delta'] = self.prepare_text_prompt(k, i, batch)
            
            return batch

        dataset = datasets.load_from_disk(dataset_path)

        # Test set:
        test_set = dataset['test']
        test_set = PriorsDataset(test_set, self.config.history, self.time_delta_map)    
        test_set.set_transform(test_set_transform)
        
        return test_set
    