import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
    
# Channel Attention and Squeeze-and-Excitation Networks (SENet)
class SE(nn.Module):
    def __init__(self, channel, reduction_ratio =16):
        super(SE, self).__init__()
        # squeeze
        self.gap = nn.AdaptiveAvgPool2d(1)
        #excitation
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.mlp(y).view(b, c, 1, 1)
        return x * y.expand_as(x) 

class SpatialSELayer(nn.Module):
 # squeezing spatially and exciting channel-wise described in: *Roy et al. MICCAI 2018*
    def __init__(self, num_channels):
        # param num_channels: No of input channels
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, weights=None):
        # spatial squeeze
        batch_size, channel, a, b = x.size()
        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(x, weights)
        else:
            out = self.conv(x)
        squeeze_tensor = self.sigmoid(out)
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(x, squeeze_tensor)
        return output_tensor

class ChannelSpatialSELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = SE(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))  

    def forward(self, x):
        cse_out = self.cSE(x)
        sse_out = self.sSE(x)
        alpha = torch.clamp(self.alpha, 0, 1)
        output_tensor = alpha * cse_out + (1 - alpha) * sse_out
        return output_tensor
