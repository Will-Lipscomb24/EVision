import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    """
    A helper block consisting of Conv2d followed by ReLU.
    The diagram shows Conv -> ReLU pairs explicitly.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class ReceptiveFieldBlock(nn.Module):
    """
    The Receptive Field Block (RFB) as detailed in the yellow box.
    It uses multiple branches with different kernel sizes and dilation rates
    to capture multi-scale features.
    """
    def __init__(self, in_channels):
        super(ReceptiveFieldBlock, self).__init__()
        
        # Internal channels for the branches. Reducing capacity internally
        # is standard practice to manage parameter count

        # With an input channel of 16, to not choke the network we will keep the sae number of intermediate channels
        inter_channels = in_channels 

        # --- Branch 1: 1x1 Conv -> 3x3 Conv ---
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, inter_channels, kernel_size=1, padding=0),
            BasicConv2d(inter_channels, inter_channels, kernel_size=3, padding=1, dilation=1)
        )

        # --- Branch 2: 3x3 Conv -> 3x3 Conv (rate=3) ---
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, inter_channels, kernel_size=3, padding=1),
            # padding = dilation * (kernel_size - 1) / 2 = 3 * (3-1) / 2 = 3
            BasicConv2d(inter_channels, inter_channels, kernel_size=3, padding=3, dilation=3)
        )

        # --- Branch 3: 5x5 Conv -> 3x3 Conv (rate=5) ---
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, inter_channels, kernel_size=5, padding=2),
            # padding = dilation * (kernel_size - 1) / 2 = 5 * (3-1) / 2 = 5
            BasicConv2d(inter_channels, inter_channels, kernel_size=3, padding=5, dilation=5)
        )

        # --- Aggregation ---
        # The outputs of the 3 branches are concatenated. 
        # The total channels will be 3 * inter_channels.
        # A 1x1 conv projects this back to the original 'in_channels' dimension.
        self.conv_cat = BasicConv2d(inter_channels * 3, in_channels, kernel_size=1, padding=0)
        
        # Final ReLU after residual addition
        self.final_relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x

        b1_out = self.branch1(x)
        b2_out = self.branch2(x)
        b3_out = self.branch3(x)

        # Concatenation (indicated by 'C' circle)
        concat = torch.cat([b1_out, b2_out, b3_out], dim=1)
        
        # Bottleneck 1x1 Conv
        out = self.conv_cat(concat)

        # Residual connection (indicated by '+' circle) followed by ReLU
        out = out + identity
        out = self.final_relu(out)

        return out

class EventEncoder(nn.Module):
    """
    The full Event Encoder architecture.
    Input: 5 x W x H
    Output: C x W x H
    """
    def __init__(self, num_bins=5, enc_channels=16, num_rfb_blocks=3):
        super(EventEncoder, self).__init__()
        
        # --- Head ---
        # Projects 5 input channels to C feature channels.
        # Using a 3x3 conv with padding 1 to maintain spatial dimensions.
        self.head = nn.Sequential(
            nn.Conv2d(num_bins, enc_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # --- Sequence of RFBs ---
        # The diagram explicitly shows 3 RFB blocks in series.
        rfb_list = []
        for _ in range(num_rfb_blocks):
            rfb_list.append(ReceptiveFieldBlock(enc_channels))
        
        self.rfb_sequence = nn.Sequential(*rfb_list)

    def forward(self, x):
        # Input shape: [Batch, 5, W, H]
        
        # Pass through Head -> f_E^0
        f_E0 = self.head(x)
        
        # Pass through sequence of RFBs -> f_E
        f_E = self.rfb_sequence(f_E0)
        
        # Output shape: [Batch, C, W, H]
        return f_E
