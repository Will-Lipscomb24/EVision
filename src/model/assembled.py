import torch
import torch.nn as nn

# Import your blocks (adjust paths as needed)
from src.model.event_encoder import EventEncoder
from src.model.image_encoder import ImageEncoder
from src.model.dual_branch_fusion import DualBranchFusion
from src.model.reconstruction_module import ReconstructionBlock

class EventImageFusionNet(nn.Module):
    def __init__(self, num_bins=5, base_channels=64, enc_channels=16, num_rfb_blocks=3, img_in_channels=1, cbam_reduction=16):
        super(EventImageFusionNet, self).__init__()
        
        # 1. Encoders
        self.event_encoder = EventEncoder(num_bins=num_bins, enc_channels=enc_channels, num_rfb_blocks=num_rfb_blocks)
        self.image_encoder = ImageEncoder(in_channels=img_in_channels, feature_channels=enc_channels, cbam_reduction=cbam_reduction)
        
        # 2. Fusion
        self.fusion = DualBranchFusion(channels=enc_channels)
        
        # 3. Reconstruction
        self.reconstruction = ReconstructionBlock(in_channels=enc_channels, out_channels=1)

    def forward(self, events, frame):
        """
        events: (B, num_bins, H, W)
        frame:  (B, 1, H, W)
        """
        # Extract featuress
        f_E = self.event_encoder(events)
        f_F = self.image_encoder(frame)
        
        # Fuse information
        f_EF = self.fusion(f_E, f_F)
        
        
        # Reconstruct image
        output = self.reconstruction(f_EF)
        
        return output