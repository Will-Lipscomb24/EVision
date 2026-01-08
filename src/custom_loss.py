# Loss Functions
import torch 
import torch.nn as nn
import lpips


class custom_loss(nn.Module):
    def __init__(self, l1_weight=1.0, lpips_weight=1.0, net = 'vgg', device='cuda'):
        super(custom_loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lpips_fn = lpips.LPIPS(net=net).to(device)
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight

    """The loss function is a combination of:
    1. MSE loss between output and target
    2. LPIPS loss which measures feature similarity between images
"""
    def forward(self, prediction, target):
        l1_loss = self.l1_loss(prediction, target)

        lpips_loss = self.lpips_fn(prediction, target).mean()

        loss = self.l1_weight*l1_loss + self.lpips_weight*lpips_loss

        return loss
