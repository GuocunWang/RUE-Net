import torch
import torch.nn as nn
from pytorch_ssim import SSIM

class ssimLoss(nn.Module):
    def __init__(self):
        super(ssimLoss, self).__init__()
        self.ssim_loss = SSIM(window_size=11)  
    
    def forward(self, image1, image2):
        ssim_value = 1 - self.ssim_loss(image1, image2) 
        return ssim_value






