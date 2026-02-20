import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CBAM-style, paper accurate)

    - Uses 1x1 Conv as shared MLP (W0 and W1)
    - No bias (matches formulation)
    - ReLU after W0
    - Shared weights for avg and max pooled descriptors
    """

    def __init__(self, in_planes, reduction_ratio=4):
        super(ChannelAttention, self).__init__()

        # Ensure valid hidden size
        hidden_planes = max(1, in_planes // reduction_ratio)

        # W0 and W1 (shared)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),  # W0
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_planes, in_planes, kernel_size=1, bias=False)   # W1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global descriptors
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)

        # Shared MLP
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)

        # Element-wise sum + sigmoid
        scale = self.sigmoid(avg_out + max_out)

        # Reweight
        return x * scale

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    Generates a 'Where' mask by pooling across channels.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Standard kernel size for Spatial Attention is 7
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                              padding=kernel_size//2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise pooling (dim=1) preserves spatial dimensions
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Spatial mask multiplication (Broadcasting)
        out = self.sigmoid(self.conv(x_cat))
        return x * out

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Sequence: Channel Attention -> Spatial Attention
    """
    def __init__(self, channels, reduction_ratio=4, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, reduction_ratio)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ResBlock(nn.Module):
    """
    ResNet Block with Integrated CBAM.
    Structure: Conv1 -> BN -> ReLU -> Conv2 -> BN -> CABM -> (+ Shortcut) -> ReLU
    This matches the paper's design where attention refines features before residual addition.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        
        # Standard ResNet Layers
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False) 
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # # Attention Module inserted inside the block
        # self.cbam = CABM(channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # # Apply Attention to the residual branch
        # out = self.cbam(out)
        
        # Add the original input (Identity Shortcut)
        out = out + residual
        out = self.relu(out)
        return out

# --- Main Image Encoder ---
class ImageEncoder(nn.Module):
    """
    Updated Image Encoder.
    Structure: Head -> 2x (ResBlock + CBAM)
    """
    def __init__(self, in_channels, feature_channels, cbam_reduction=4):
        """
        Args:
            in_channels (int): Input image channels (default 1).
            feature_channels (int): Internal feature width.
            cbam_reduction (int): Ratio for channel attention (recommended 4 for small models).
        """
        super(ImageEncoder, self).__init__()
        
        # 1. Head: Conv + ReLU
        # FIX: inplace=False to prevent version errors in autograd
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # 2. Stacked ResBlocks with integrated CBAM
        # We replace the old isolated blocks with the fused ResBlockCBAM
        self.rb1 = ResBlock(feature_channels)
        self.rb2 = ResBlock(feature_channels)
        self.cbam = CBAM(feature_channels)

    def forward(self, x):
        # Input x: Frame F (B, 1, H, W)
        
        # Get initial features
        f0_F = self.head(x)
        
        # Pass through the attention-enhanced blocks
        x = self.rb1(f0_F)
        x = self.rb2(x)
        f_F = self.cbam(x)
        
        # Output f_F: Refined Frame features (B, C, H, W)
        return f_F