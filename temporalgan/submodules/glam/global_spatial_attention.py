import torch
from torch import nn

class GlobalSpatialAttention(nn.Module):
    def __init__(self, in_channels, num_reduced_channels):
        super().__init__()
        
        self.conv1x1_q = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_k = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_v = nn.Conv2d(in_channels, num_reduced_channels, 1, 1)
        self.conv1x1_att = nn.Conv2d(num_reduced_channels, in_channels, 1, 1)
        self.att = None
        self.query_key = None
    def forward(self, feature_maps, global_channel_output):
        query = self.conv1x1_q(feature_maps)
        N, C, H, W = query.shape
        query = query.reshape(N, C, -1)
        key = self.conv1x1_k(feature_maps).reshape(N, C, -1)
        
        self.query_key = torch.bmm(key.permute(0, 2, 1), query)
        self.query_key = self.query_key.reshape(N, -1).softmax(-1)
        self.query_key = self.query_key.reshape(N, int(H*W), int(H*W))
        value = self.conv1x1_v(feature_maps).reshape(N, C, -1)
        self.att = torch.bmm(value, self.query_key).reshape(N, C, H, W)
        self.att = self.conv1x1_att(self.att)
        
        return (global_channel_output * self.att) + global_channel_output
