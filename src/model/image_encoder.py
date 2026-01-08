import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Standard Basic Residual Block as used in ResNet.
    Consists of: Conv -> BN -> ReLU -> Conv -> BN -> (+ shortcut) -> ReLU
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual # Add the identity shortcut
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    """
    Channel Attention Module part of CABM.
    Uses Global Average and Max pooling, a shared MLP, and Sigmoid.
    """
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        hidden_planes = max(4, in_planes // reduction_ratio) # Ensure a minimum size
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_planes, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Adaptive pooling handles any H, W automatically
        avg_out = self.shared_mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.shared_mlp(F.adaptive_max_pool2d(x, 1))
        
        # Combine and scale input
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Standard kernel size for Spatial Attention is 7
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                              padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Spatial mask multiplication
        out = self.sigmoid(self.conv(x_cat))
        return x * out

class CABM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Applies Channel Attention followed by Spatial Attention.
    """
    def __init__(self, channels, reduction_ratio=16, spatial_kernel=7):
        super(CABM, self).__init__()
        self.ca = ChannelAttention(channels, reduction_ratio)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# --- Main Image Encoder ---

class ImageEncoder(nn.Module):
    """
    Image Encoder as described in the figure and text.
    Structure: Head (Conv+ReLU) -> 2 Residual Blocks -> CABM.
    """
    def __init__(self, in_channels, feature_channels, cbam_reduction):
        """
        Args:
            in_channels (int): Number of input image channels (default is 1 for grayscale Frame Input).
            feature_channels (int): 'C' in the description. Number of channels for internal features.
            cbam_reduction (int): Reduction ratio for the Channel Attention module.
        """
        super(ImageEncoder, self).__init__()
        
        # 1. Head: Conv + ReLU
        # Obtains initial image feature f0_F
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. Two stacked Residual Blocks
        self.rb1 = ResidualBlock(feature_channels)
        self.rb2 = ResidualBlock(feature_channels)
        
        # 3. Convolutional Block Attention Module (CABM)
        self.cabm = CABM(feature_channels, reduction_ratio=cbam_reduction)

    def forward(self, x):
        # Input x: Frame F (B, 1, H, W)
        
        # Get initial features
        f0_F = self.head(x)  # -> (B, C, H, W)
        
        # Pass through residual blocks
        x = self.rb1(f0_F)
        x = self.rb2(x)
        
        # Apply attention
        f_F = self.cabm(x)   # -> (B, C, H, W)
        
        # Output f_F: Frame feature (B, C, H, W)
        return f_F

