import torch

from modules.lightning_modules.ed_cxr.report_gen_rev_f import MIMICIVEDCXRReportGen


class FreezeEncoderPartialWarmStartOptimiser(MIMICIVEDCXRReportGen):

    def __init__(self, allow_warm_start_optimiser_partial, warm_start_ckpt_path, **kwargs):
        super().__init__(**kwargs)

        self.allow_warm_start_optimiser_partial = allow_warm_start_optimiser_partial
        self.warm_start_ckpt_path = warm_start_ckpt_path
        self.warm_start_optimiser_partial = False

        # Freeze encoder:
        for n, p in self.encoder_decoder.named_parameters():
            if 'encoder.uniformer' in n:
                p.requires_grad = False

    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        new_model_params = []
        for k in self.encoder_decoder.decoder.config.index_value_encoder_config.keys():
            new_model_params += getattr(self.encoder_decoder, f'{k}_index_value_encoder').parameters()
        if hasattr(self.encoder_decoder, 'time_delta_encoder'):
            new_model_params += self.encoder_decoder.time_delta_encoder.parameters()

        if self.allow_warm_start_optimiser_partial and self.warm_start_optimiser_partial:
            base_model_param_groups = [
                {
                    'params': list(self.encoder_decoder.encoder.parameters()) + \
                        list(self.encoder_decoder.decoder.parameters()) + \
                        list(self.encoder_decoder.token_type_embeddings.parameters()),
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
                    'params': list(self.encoder_decoder.encoder.parameters()) + \
                        list(self.encoder_decoder.decoder.parameters()) + \
                        list(self.encoder_decoder.token_type_embeddings.parameters()),
                    'lr': self.lr,
                },
                {'params': new_model_params, 'lr': self.lr},
            ]

            optimiser = torch.optim.AdamW(param_groups)
        return {'optimizer': optimiser}
