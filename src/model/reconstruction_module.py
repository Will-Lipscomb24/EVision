import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.image_encoder import ResidualBlock

class ReconstructionBlock(nn.Module):
    """
    Reconstruction Module as shown in Fig. 2.
    Structure: RB -> RB -> Tail (1x1 Conv + Sigmoid)
    """
    def __init__(self, in_channels=16, out_channels=1):
        """
        Args:
            in_channels (int): Number of feature channels coming from the Fusion block (f_EF).
            out_channels (int): Number of output image channels (1 for grayscale).
        """
        super(ReconstructionBlock, self).__init__()
        
        # 1. Two stacked Residual Blocks
        self.rb1 = ResidualBlock(in_channels)
        self.rb2 = ResidualBlock(in_channels)
        
        # 2. Tail Block
        # Projects features back to image space.
        # Uses 1x1 Convolution to reduce channels to 1 (grayscale).
        # Uses Sigmoid to ensure pixel values are in range [0, 1].
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, f_EF):
        """
        Args:
            f_EF: Fused features from the Dual-branch Fusion Module (B, C, H, W).
        Returns:
            enhanced_image: The final reconstructed image (B, 1, H, W).
        """
        # Pass through residual blocks
        x = self.rb1(f_EF)
        x = self.rb2(x)
        
        # Project to image space
        enhanced_image = self.tail(x)
        
        return enhanced_image

