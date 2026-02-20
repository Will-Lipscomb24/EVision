import torch
import torch.nn as nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class custom_loss(nn.Module):
    def __init__(self, l1_weight=1.0, lpips_weight=1.0, net='vgg', device='cuda'):
        super(custom_loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        
        # TorchMetrics Implementation
        # net_type: maps to 'vgg', 'alex', or 'squeeze'
        # normalize=False: tells it we are providing inputs in [-1, 1] range (which we do below)
        self.lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=True).to(device)
        
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight

    def forward(self, prediction, target):
        l1 = self.l1_loss(prediction, target)

        # 1. Create 3-channel versions for LPIPS
        # We assume input is (N, 1, H, W)
        pred_3c = prediction.repeat(1, 3, 1, 1)
        target_3c = target.repeat(1, 3, 1, 1)

        # 2. Pass 3-channel data to LPIPS
        # (normalize=True handles the [0,1] -> [-1,1] scaling)
        lpips_val = self.lpips_fn(pred_3c, target_3c)

        loss = self.l1_weight * l1 + self.lpips_weight * lpips_val
        return loss

