import os

import torch
from bert_score import BERTScorer
from transformers import AutoModel, AutoTokenizer


class BERTScoreReward:

    def __init__(self, device, num_workers):
        
        self.bert_scorer = BERTScorer(
            model_type='roberta-large',
            num_layers=17,
            nthreads=num_workers,
            all_layers=False,
            idf=False,
            lang='en',
            device=device,
            rescale_with_baseline=True,
        )
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def __call__(self, predictions, labels):
        return self.reward(predictions, labels)

    def reward(self, predictions, labels):

        with torch.no_grad() and torch.autocast(device_type='cuda', dtype=torch.float32):

            bert_scores = self.bert_scorer.score(predictions, labels, batch_size=len(predictions))
            f1 = bert_scores[2].to(device=self.bert_scorer.device)

        return f1


class CXRBERTReward:

    def __init__(self, device):
        self.device = device

        # Load the model and tokenizer:
        ckpt_name = 'microsoft/BiomedVLP-CXR-BERT-specialized'
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(ckpt_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def __call__(self, predictions, labels):
        return self.reward(predictions, labels)

    def reward(self, predictions, labels):
        assert isinstance(predictions, list), '"predictions" must be a list of strings.'
        assert all(isinstance(i, str) for i in predictions), 'Each element of "predictions" must be a string.'
        assert isinstance(labels, list), '"labels" must be a list of lists, where each sub-list has a multiple strings.'
        assert all(isinstance(i, list) for i in labels), 'Each element of "labels" must be a list of strings.'
        assert all(isinstance(j, str) for i in labels for j in i), 'each sub-list must have one or more strings.'

        with torch.no_grad():

            # Tokenize and compute the sentence embeddings:
            tokenizer_output = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=predictions,
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
                batch_text_or_text_pairs=[j for i in labels for j in i],
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

        return sim


class ARNReward:

    def __init__(self, n, tokenizer, device):
        self.n = n
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, predictions):
        return self.reward(predictions)

    def reward(self, predictions):
        rewards = []
        for i in predictions:
            
            tokens = self.tokenizer.tokenize(i)
            
            # If the sequence is shorter than the n-gram size, maximum reward is given:
            if len(tokens) < self.n:
                rewards.append(1.0)
                continue
            
            # Count the occurrences of n-grams:
            ngram_counts = {}
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                if ngram in ngram_counts:
                    ngram_counts[ngram] += 1
                else:
                    ngram_counts[ngram] = 1
            
            # Total number of n-grams:
            total_ngrams = len(tokens) - self.n + 1

            # Calculate the number of repeated n-grams:
            repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)

            # Calculate the reward as the absence of repeated n-grams:
            reward = 1.0 - (repeated_ngrams / total_ngrams)  # Invert the penalty to get reward:

            rewards.append(reward)

        return torch.tensor(rewards, device=self.device)
    