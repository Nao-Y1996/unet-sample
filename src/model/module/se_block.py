import torch
import torch.nn as nn
import torch.nn.functional as F

"""
SEBlock

ChannelSELayer: 
    Channel-wise Squeeze and Excitation Layer
    paper: https://arxiv.org/abs/1709.01507
SpatialSELayer:
    Spatial-wise Squeeze and Excitation Layer
    paper: https://arxiv.org/abs/1803.02579
ChannelSpatialSELayer:
    Channel-wise and Spatial-wise Squeeze and Excitation Layer
    paper: https://arxiv.org/abs/1803.02579
"""


class ChannelSELayer(nn.Module):
    def __init__(self, in_channels: int, reduce_ratio: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.middle_channels = in_channels // reduce_ratio
        self.layers = nn.Sequential(
            nn.Linear(in_channels, self.middle_channels),
            nn.ReLU(),
            nn.Linear(self.middle_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, c, h, w = x.size()
        avg_pool = F.avg_pool2d(x, kernel_size=(h, w)).view(batch_size, c)
        channel_weights = self.layers(avg_pool)
        channel_weights = channel_weights.view(batch_size, c, 1, 1)
        x = x * channel_weights
        return x


class SpatialSELayer(nn.Module):
    def __init__(self, in_channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spatial_weight = self.layers(x)
        x = x * spatial_weight
        return x


class ChannelSpatialSELayer(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(in_channels, reduction_ratio)
        self.sSE = SpatialSELayer(in_channels)

    def forward(self, input_tensor):
        cse = self.cSE(input_tensor)
        sse = self.sSE(input_tensor)
        return cse + sse


if __name__ == '__main__':
    # Test ChannelSELayer
    x = torch.rand((3, 2, 5, 5))
    cse = ChannelSELayer(in_channels=2)
    c_out = cse(x)
    assert c_out.size() == x.size()

    # Test SpatialSELayer
    x = torch.rand((3, 2, 5, 5))
    sse = SpatialSELayer(in_channels=2)
    s_out = sse(x)
    assert s_out.size() == x.size()

    # Test ChannelSpatialSELayer
    x = torch.rand((3, 2, 5, 5))
    csse = ChannelSpatialSELayer(in_channels=2)
    cs_out = csse(x)
    assert cs_out.size() == x.size()
    
