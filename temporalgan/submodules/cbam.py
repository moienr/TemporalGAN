import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Channel Attention module that learns to focus on important channels of input features.
    It contains two parallel branches, one using average pooling and another using max pooling, to capture
    both global and local information in the feature maps. The output of these branches is then passed
    through a multi-layer perceptron (MLP) with ReLU activation function to learn channel-wise importance
    scores. Finally, the importance scores are multiplied with the input features to emphasize important
    channels and suppress irrelevant channels.


    """
    def __init__(self, n_channels:int, ratio=4):
        """ Channel Attention module that learns to focus on important channels of input features.
        
        Args:
        ----
            `n_channels` (int): Number of input channels.
            `ratio` (int): Reduction ratio for the intermediate MLP layer (default: 4).

        Input:
        ---
            `x` (tensor): Input feature map of shape (batch_size, n_channels, height, width).

        Output:
        ---
            `refined_feats` (tensor): Refined feature map after applying channel attention, of shape
                                    (batch_size, n_channels, height, width).
        """
        super().__init__()

        # Global average pooling layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Global max pooling layer
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Multi-layer perceptron (MLP) to learn channel-wise importance scores
        self.mlp = nn.Sequential(
            nn.Linear(n_channels, n_channels//ratio, bias=False),
            nn.ReLU(),
            nn.Linear(n_channels//ratio, n_channels, bias=False)
        )

        # Sigmoid activation function to normalize the importance scores
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Applies channel attention to the input feature map.

        Args:
            x (tensor): Input feature map of shape (batch_size, n_channels, height, width).

        Returns:
            refined_feats (tensor): Refined feature map after applying channel attention, of shape
                                    (batch_size, n_channels, height, width).
        """
        # Apply global average pooling and pass through MLP
        x1 = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x1 = self.mlp(x1)

        # Apply global max pooling and pass through MLP
        x2 = self.max_pool(x).squeeze(-1).squeeze(-1)
        x2 = self.mlp(x2)

        # Add the outputs of the two MLP branches and apply sigmoid activation
        feats = x1 + x2
        feats = self.sigmoid(feats).unsqueeze(-1).unsqueeze(-1)

        # Multiply the importance scores with the input features to obtain refined features
        refined_feats = x * feats

        return refined_feats

class SpatialAttention(nn.Module):
    """
    This class defines a spatial attention module that enhances the informative regions of an input feature map
    by generating a spatial attention map that is multiplied with the input feature map.
    """
    def __init__(self, kernel_size=7):
        """    This class defines a spatial attention module that enhances the informative regions of an input feature map
        by generating a spatial attention map that is multiplied with the input feature map.

        Args:
            kernel_size (int, optional): Kernel size for the convolutional layer. Defaults to 7.
        """
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)

        feats = torch.cat([x1, x2], dim=1)
        feats = self.conv(feats)
        feats = self.sigmoid(feats)
        refined_feats = x * feats

        return refined_feats



class CBAM(nn.Module):
    """ This class defines a CBAM module that combines channel attention and spatial attention."""
    def __init__(self, channel):
        super().__init__()

        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x





if __name__ == '__main__':
    print('Testing Channel Attention Module...')
    x = torch.randn(1, 12, 128, 128)
    ca = ChannelAttention(12)
    y = ca(x)
    print(y.shape)
    
    print('Testing Spatial Attention Module...')
    x = torch.randn(1, 12, 128, 128)
    sa = SpatialAttention()
    y = sa(x)
    print(y.shape)
    
    print('Testing CBAM Module...')
    x = torch.randn((8, 32, 128, 128))
    module = CBAM(32)
    y = module(x)
    print(y.shape)