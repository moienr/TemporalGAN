import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    """
    A CNN block consisting of a convolutional layer, optional instance normalization,
    and a leaky ReLU activation function.

    Parameters:
    ----------
    `in_channels`: int
        The number of input channels to the convolutional layer.
    `out_channels`: int
        The number of output channels from the convolutional layer.
    `stride`: int, optional (default=2)
        The stride of the convolutional layer.
    `norm`: bool, optional (default=True)
        If True, add an instance normalization layer after the convolutional layer.

    Attributes:
    ----------
    `conv`: nn.Sequential
        A sequential module containing the convolutional layer, 
        instance normalization layer (if norm=True), and leaky ReLU activation function.

    Methods:
    -------
    `forward(x)`:
        Passes the input tensor through the convolutional block and returns the output tensor.
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size = 4, stride=2, padding = 1, bias = False, norm=True):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, padding_mode='reflect'))
        if norm: 
            layers.append(nn.InstanceNorm2d(out_channels)) 
        layers.append(nn.LeakyReLU(0.2))

        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
    
if __name__ == '__main__':
    # Test the CNNBlock class
    x = torch.randn(1, 7, 256, 256)
    block = CNNBlock(7, 64, stride=2, norm=False)
    print(block(x).shape)