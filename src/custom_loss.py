# Loss Functions
import torch 
import torch.nn as nn
import lpips

class custom_loss(nn.Module):
    def __init__(self, l1_weight=1.0, lpips_weight=1.0, net='vgg', device='cuda'):
        super(custom_loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lpips_fn = lpips.LPIPS(net=net).to(device)
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight

    def forward(self, prediction, target):
        # 1. L1 Loss (Standard [0,1] scale)
        l1 = self.l1_loss(prediction, target)

        # 2. Normalize to [-1, 1] for LPIPS
        pred_norm = (prediction * 2.0) - 1.0
        target_norm = (target * 2.0) - 1.0

        # --- SAFETY FIX: Add non-trivial noise during training ---
        # If the model predicts the target perfectly (e.g., both are black images),
        # LPIPS distance becomes 0, and sqrt(0) gradient explodes.
        if self.training:
            # 1e-4 is small enough to not hurt accuracy, but large enough to prevent 0.0 distance
            noise = torch.randn_like(pred_norm) * 1e-4
            pred_norm = pred_norm + noise

        # 3. LPIPS Loss
        # We also add a tiny epsilon to the result just in case, 
        # though the noise above usually catches it.
        lpips_val = self.lpips_fn(pred_norm, target_norm).mean() + 1e-8

        # 4. Combine
        loss = self.l1_weight * l1 + self.lpips_weight * lpips_val
        return loss