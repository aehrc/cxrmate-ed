import transformers
from transformers.models.auto import CONFIG_MAPPING


class CXRMateEDConfig(transformers.PretrainedConfig):

    model_type = 'cxrmate-ed'

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        index_value_encoder_intermediate_size: int = 2048,
        include_time_delta: bool = True,
        time_delta_monotonic_inversion: bool = True,
        add_time_deltas: bool = True,
        history: int = 0,
        tables_filter: list = ['mimic_cxr_sectioned', 'triage', 'medrecon'],
        prompt_report_sections_filter: list = ['indication', 'history'],
        pad_token_id: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_config = vision_config
        
        self.index_value_encoder_intermediate_size = index_value_encoder_intermediate_size
        self.include_time_delta = include_time_delta
        self.time_delta_monotonic_inversion = time_delta_monotonic_inversion
        self.add_time_deltas = add_time_deltas
        self.history = history
        self.tables_filter = tables_filter
        self.prompt_report_sections_filter = prompt_report_sections_filter
        self.pad_token_id = pad_token_id      

        if isinstance(vision_config, dict):
            vision_config = transformers.AutoConfig.from_pretrained(
                'aehrc/uniformer_base_tl_384',
                trust_remote_code=True,
                **vision_config,
            )
            
        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config['model_type'] = text_config['model_type'] if 'model_type' in text_config else 'llama'
            text_config = CONFIG_MAPPING[text_config['model_type']](**text_config)

        self.text_config = text_config
