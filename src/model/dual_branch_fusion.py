import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

class DeformableConvLayer(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super(DeformableConvLayer, self).__init__()
        
        # 1. Store attributes for use in forward()
        self.kernel_size = kernel_size
        self.padding = padding
        self.in_channels = in_channels
        
        # 2. Offset and Mask Generation Network
        # Generates (2 * k*k) offsets and (1 * k*k) mask values
        self.offset_mask_net = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 3 * kernel_size * kernel_size, kernel_size=3, padding=1)
        )
        
        # 3. Main Convolutional Weights and Bias
        # These are the actual filters applied after the sampling points are moved
        self.weight = nn.Parameter(torch.Tensor(in_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(in_channels))
        
        # 4. Initialization
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize main weights with Kaiming
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.constant_(self.bias, 0)
        
        # IMPORTANT: Initialize the last layer of offset_mask_net to zero.
        # This prevents the initial "NaN" errors by making the 
        # offsets 0 and the mask neutral at the start of training.
        nn.init.constant_(self.offset_mask_net[-1].weight, 0)
        nn.init.constant_(self.offset_mask_net[-1].bias, 0)

    def forward(self, f_F, f_E):
        # Guidance: concatenate event features and frame features
        combined = torch.cat([f_E, f_F], dim=1)
        out = self.offset_mask_net(combined)
        
        # Split output into offsets and masks
        offset_channels = 2 * self.kernel_size * self.kernel_size
        offsets = out[:, :offset_channels, :, :]
        mask_logits = out[:, offset_channels:, :, :]
        
        # Epsilon (1e-7) prevents SqrtBackward0 NaNs during interpolation
        mask = torch.sigmoid(mask_logits) 
        
        # Perform the actual Deformable Convolution
        f_Edc = deform_conv2d(
            f_E, offsets, self.weight, self.bias, 
            padding=self.padding, mask=mask
        )
        
        return F.relu(f_Edc, inplace=True)

class SpatialAttentionBlock(nn.Module):
    """
    Top Branch of the fusion module in Fig. 4.
    Uses attention to modulate event features based on frame features.
    """
    def __init__(self, channels):
        super(SpatialAttentionBlock, self).__init__()
        
        # Attention Map (M_E) Generation Branch
        # Structure: Concatenate -> Conv -> ReLU -> Conv -> Sigmoid
        self.attention_gen = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Refinement Branch to generate f_EFsa
        # Structure: Concatenate -> Conv -> ReLU
        self.refinement = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_E, f_F):
        # 1. Generate Attention Map M_E
        combined_att = torch.cat([f_E, f_F], dim=1)
        M_E = self.attention_gen(combined_att)
        
        # 2. Feature Modulation
        # Element-wise multiplication of event features with the attention map
        f_Esa = f_E * M_E
        
        # 3. Refinement to get f_EFsa
        combined_ref = torch.cat([f_Esa, f_F], dim=1)
        f_EFsa = self.refinement(combined_ref)
        
        return f_EFsa

class DualBranchFusion(nn.Module):
    """
    Main fusion module as depicted in Fig. 4.
    Combines SAB and DCB branches and performs final fusion.
    """
    def __init__(self, channels=16):
        super(DualBranchFusion, self).__init__()
        
        # --- Top Branch: Spatial Attention Block (SAB) ---
        self.sab = SpatialAttentionBlock(channels)
        
        # --- Bottom Branch: Deformable Convolutional Block (DCB) ---
        # Part 1: Deformable Convolution (generates f_Edc)
        self.dcb_layer = DeformableConvLayer(channels)
        
        # Part 2: Refinement (generates f_EFdc)
        # Structure: Concatenate -> Conv -> ReLU
        self.dcb_refinement = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # --- Final Fusion Branch ---
        # Fuses outputs from SAB and DCB.
        # Structure: Concatenate -> Conv -> ReLU
        self.final_fusion = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_E, f_F):
        """
        Args:
            f_E: Event features (B, C, H, W)
            f_F: Frame features (B, C, H, W)
        Returns:
            f_EF: Fused features (B, C, H, W)
        """
        # 1. Forward pass through SAB branch
        # Output: f_EFsa
        f_EFsa = self.sab(f_E, f_F)
        
        # 2. Forward pass through DCB branch
        # a. Apply Deformable Conv to align f_F to f_E -> f_Edc
        f_Edc = self.dcb_layer(f_F, f_E)
        
        # b. Refine f_Edc by concatenating with f_F -> f_EFdc
        combined_dcb = torch.cat([f_Edc, f_F], dim=1)
        f_EFdc = self.dcb_refinement(combined_dcb)
        
        # 3. Final Fusion
        # Concatenate outputs of both branches and refine -> f_EF
        combined_final = torch.cat([f_EFsa, f_EFdc], dim=1)
        f_EF = self.final_fusion(combined_final)
        
        return f_EF

