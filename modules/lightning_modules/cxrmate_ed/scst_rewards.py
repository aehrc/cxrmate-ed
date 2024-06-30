import nltk
import torch

from modules.lightning_modules.cxrmate2.rl import SCST
from tools.rewards.bertscore import BERTScoreReward
from tools.rewards.chexbert import CheXbertReward
from tools.rewards.cxrbert import CXRBERTReward


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


class SCSTCheXbertCXRBERTBERTScoreReward(SCST):
    
    def __init__(self, weights=[1/3]*3, **kwargs):
        super(SCSTCheXbertCXRBERTBERTScoreReward, self).__init__(**kwargs)
        self.weights = weights
        
    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward_cxrbert = CXRBERTReward(device=self.device)
        self.reward_bertscore = BERTScoreReward(device=self.device, num_workers=self.num_workers)
        self.reward_chexbert = CheXbertReward(            
            device=self.device,
            ckpt_dir=self.ckpt_zoo_dir, 
            bert_path='bert-base-uncased',
            checkpoint_path='stanford/chexbert/chexbert.pth',
        )
        
    def reward(self, predictions, labels):
        reward_cxrbert = self.reward_cxrbert(predictions, labels)
        reward_bertscore = self.reward_bertscore(predictions, labels)
        reward_chexbert = self.reward_chexbert(predictions, labels)

        reward_cxrbert = self.weights[0]*reward_cxrbert
        reward_bertscore = self.weights[1]*reward_bertscore
        reward_chexbert = self.weights[2]*reward_chexbert

        # Composite reward:
        reward = reward_cxrbert + reward_bertscore + reward_chexbert
        
        return reward 
   
    
class SCSTCXRBERTLengthPenaltyReward(SCST):
        
    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward_cxrbert = CXRBERTReward(device=self.device)

    def reward(self, predictions, labels):
        reward = self.reward_cxrbert(predictions, labels)

        # Length ratio penalty:
        length_ratio = torch.tensor([len(nltk.word_tokenize(i)) for i in predictions], dtype=torch.float32, device=self.device)
        length_ratio /= torch.tensor([len(nltk.word_tokenize(i[0])) for i in labels], dtype=torch.float32, device=self.device)
        length_ratio = 1.0 - abs(1.0 - length_ratio)
        reward = length_ratio * reward
        
        return reward
    
    
class SCSTCXRBERTBERTScoreLengthPenatlyReward(SCST):
    
    def __init__(self, weights=[0.5, 0.5], **kwargs):
        super(SCSTCXRBERTBERTScoreLengthPenatlyReward, self).__init__(**kwargs)
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
        
        # Length ratio penalty:
        length_ratio = torch.tensor([len(nltk.word_tokenize(i)) for i in predictions], dtype=torch.float32, device=self.device)
        length_ratio /= torch.tensor([len(nltk.word_tokenize(i[0])) for i in labels], dtype=torch.float32, device=self.device)
        length_ratio = 1.0 - abs(1.0 - length_ratio)
        reward = length_ratio * reward        

        return reward
    
    
class SCSTCheXbertCXRBERTBERTScoreLengthPenaltyReward(SCST):
    
    def __init__(self, weights=[1/3]*3, **kwargs):
        super(SCSTCheXbertCXRBERTBERTScoreLengthPenaltyReward, self).__init__(**kwargs)
        self.weights = weights
        
    def on_fit_start(self):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-fit-start.
        """
        self.reward_cxrbert = CXRBERTReward(device=self.device)
        self.reward_bertscore = BERTScoreReward(device=self.device, num_workers=self.num_workers)
        self.reward_chexbert = CheXbertReward(            
            device=self.device,
            ckpt_dir=self.ckpt_zoo_dir, 
            bert_path='bert-base-uncased',
            checkpoint_path='stanford/chexbert/chexbert.pth',
        )
        
    def reward(self, predictions, labels):
        reward_cxrbert = self.reward_cxrbert(predictions, labels)
        reward_bertscore = self.reward_bertscore(predictions, labels)
        reward_chexbert = self.reward_chexbert(predictions, labels)

        reward_cxrbert = self.weights[0]*reward_cxrbert
        reward_bertscore = self.weights[1]*reward_bertscore
        reward_chexbert = self.weights[2]*reward_chexbert

        # Composite reward:
        reward = reward_cxrbert + reward_bertscore + reward_chexbert
        
        # Length ratio penalty:
        length_ratio = torch.tensor([len(nltk.word_tokenize(i)) for i in predictions], dtype=torch.float32, device=self.device)
        length_ratio /= torch.tensor([len(nltk.word_tokenize(i[0])) for i in labels], dtype=torch.float32, device=self.device)
        length_ratio = 1.0 - abs(1.0 - length_ratio)
        reward = length_ratio * reward   
        
        return reward 
    