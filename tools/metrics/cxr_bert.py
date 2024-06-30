import torch
from transformers import AutoModel, AutoTokenizer

from modules.transformers.microsoft.modelling_cxrbert import CXRBertModel
from tools.metrics.mimic_cxr import MIMICCXRReportGenerationMetric


class CXRBERTMetric(MIMICCXRReportGenerationMetric):
    """
    CXR-BERT similarity for MIMIC-CXR. If multiple reports are generated per study_id, each error type is
    summed over the dicom_ids.
    """

    def __init__(self, **kwargs):
        super().__init__(metric_name='cxrbert', **kwargs)

    def init_metric(self):

        # Load the model and tokenizer
        ckpt_name = 'microsoft/BiomedVLP-CXR-BERT-specialized'
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(ckpt_name, trust_remote_code=True).to(self.device)

        self.model.eval()

    def cleanup_metric(self):
        del self.tokenizer, self.model

    def metric_scoring(self, batch):

        y_hat = [i['synthetic'] for i in batch]
        y = [i['radiologist'] for i in batch]
        study_ids = [i['study_id'] for i in batch]
        if self.accumulate_over_dicoms:
            dicom_ids = [i['dicom_id'] for i in batch]

        # Tokenize and compute the sentence embeddings
        tokenizer_output = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=y_hat,
            add_special_tokens=True,
            padding='longest',
            return_tensors='pt',
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        )

        prediction_embeddings = self.model(
            input_ids=tokenizer_output.input_ids.to(self.device),
            attention_mask=tokenizer_output.attention_mask.to(self.device),
            output_cls_projected_embedding=True,
            return_dict=False,
        )

        tokenizer_output = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=y,
            add_special_tokens=True,
            padding='longest',
            return_tensors='pt',
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        )

        label_embeddings = self.model(
            input_ids=tokenizer_output.input_ids.to(self.device),
            attention_mask=tokenizer_output.attention_mask.to(self.device),
            output_cls_projected_embedding=True,
            return_dict=False,
        )

        # Compute the cosine similarity of sentence embeddings obtained from input text prompts.
        sim = torch.nn.functional.cosine_similarity(
            prediction_embeddings[2],
            label_embeddings[2],
        )

        mbatch_rows = []
        if self.accumulate_over_dicoms:
            for x, y, z in zip(dicom_ids, study_ids, sim.tolist()):
                mbatch_rows.append({'dicom_id': x, 'study_id': y, 'similarity': z})
        else:
            for x, y in zip(study_ids, sim.tolist()):
                mbatch_rows.append({'study_id': x, 'similarity': y})

        return mbatch_rows

