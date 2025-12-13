'''
we gotta use the nima-vgg-16-ava lessgoooooooo
'''

import pyiqa
import torch
from PIL import Image
from torchvision import transforms
from math import exp
import torch.nn as nn
import torch.nn.functional as F

class NIMAVGGLoss(nn.Module):
    def __init__(self, gpu_ids=[], iqa_metric = 'nima-vgg16-ava'):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
        self.metric = pyiqa.create_metric(iqa_metric, device=self.device)   
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, generated_image: torch.Tensor, target_image: torch.Tensor, NR=False) -> torch.Tensor:
        generated_image = (generated_image + 1) / 2
        target_image = (target_image + 1) / 2
        generated_image = generated_image.to(self.device)
        target_image = target_image.to(self.device)

        score1 = self.metric(generated_image) 
        score2 = self.metric(target_image)  

        # print(score1)

        if NR:
            return (10.0 - score1).mean()
        else:
            return self.l1_loss(score1, score2).mean()