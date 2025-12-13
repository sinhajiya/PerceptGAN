'''
iqa loss
'''

import pyiqa
import torch
from PIL import Image
from torchvision import transforms
from math import exp
import torch.nn as nn
import torch.nn.functional as F
import re

class IQA(nn.Module):
    def __init__(self, gpu_ids=[], iqa_metric = 'arniqa-spaq'):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
        self.metric = pyiqa.create_metric(iqa_metric, device=self.device)   
        self.l1_loss = torch.nn.L1Loss()
        
        r = self.metric.score_range
        if isinstance(r, (list, tuple)):
            self.max_score = float(r[1])
        else:
            self.max_score = float(re.sub(r'\D', '', str(r)))

    def forward(self, generated_image: torch.Tensor, target_image: torch.Tensor, NR=False, L2=False) -> torch.Tensor:
        # Convert from [-1, 1] to [0, 1]
        if torch.isnan(generated_image).any():
            print("NaN in generated image")
            generated_image = torch.nan_to_num(generated_image, nan=0.0)

        if torch.isnan(target_image).any():
            print("NaN in target image")
            target_image = torch.nan_to_num(target_image, nan=0.0)
            
        generated_image = (generated_image + 1) / 2.0
        target_image = (target_image + 1) / 2.0
        
        generated_image = torch.clamp(generated_image, 0.0, 1.0)
        
        target_image = torch.clamp(target_image, 0.0, 1.0)

        generated_image = generated_image.to(self.device)
        target_image = target_image.to(self.device)

        if NR:
            score = self.metric(generated_image)
            diff = (self.max_score - score) / self.max_score
            if L2:
                return (diff ** 2).mean()
            return diff.mean()

        else:
            # Full-reference: L1 distance between predicted scores
            score_gen = self.metric(generated_image)
            score_ref = self.metric(target_image)
            # return self.l1_loss(score_gen, score_ref).mean()\
            return self.l1_loss(score_gen / self.max_score, score_ref / self.max_score).mean()
