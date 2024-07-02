import torch
import transformers

from modules.lightning_modules.ed_cxr.ablation import MinusMedrecon


class SCST(MinusMedrecon):
    
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
        for p in self.encoder_decoder.encoder.parameters():
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
        inputs_embeds, attention_mask, token_type_ids, position_ids, bos_token_ids = self.encoder_decoder.prepare_inputs(tokenizer=self.tokenizer, **batch)
          
        # Samples:
        sample = self.encoder_decoder.generate.__wrapped__(  # Use __wrapped__ to avoid the torch.no_grad() decorator of generate().
            self.encoder_decoder,
            input_ids=bos_token_ids,
            prompt_attention_mask=attention_mask,
            prompt_position_ids=position_ids,
            decoder_inputs_embeds=inputs_embeds,
            decoder_token_type_ids=token_type_ids,
            special_token_ids=[self.tokenizer.sep_token_id],
            token_type_id_sections=self.section_ids,
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
        findings, impression = self.encoder_decoder.split_and_decode_sections(
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
        baseline_ids = self.encoder_decoder.generate(
            input_ids=bos_token_ids,
            prompt_attention_mask=attention_mask,
            prompt_position_ids=position_ids,
            decoder_inputs_embeds=inputs_embeds,
            decoder_token_type_ids=token_type_ids,
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
        baseline_findings, baseline_impression = self.encoder_decoder.split_and_decode_sections(
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

        # Update and log scores for each metric:
        self.log_dict(
            {'reward': torch.mean(reward), 'baseline': torch.mean(baseline)},
            on_step=True,
            on_epoch=True,
            batch_size=batch['images'].size()[0],
        )

        # Logging
        self.log_dict({'scst_loss': loss}, on_step=True, on_epoch=True, batch_size=batch['images'].size()[0])

        # Update and log scores for each validation metric.
        return loss
    
    def reinforce_loss(self, logits: torch.Tensor, sampled_token_ids: torch.Tensor,
                       reward: torch.Tensor) -> torch.Tensor:
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
