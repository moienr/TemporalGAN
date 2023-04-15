import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, stride=2, padding = 1, down=True, act="relu", use_dropout=False):
        """
        A building block for the generator and discriminator models in a GAN.

        Args:
            `in_channels (int)`: The number of input channels.
            `out_channels (int)`: The number of output channels.
            `kernel_size (int)`: The size of the convolutional kernel. Defaults to 4.
            `stride (int)`: The stride of the convolution. Defaults to 2.
            `padding (int)`: The amount of padding to apply. Defaults to 1.
            `down (bool)`: If True, the block uses a downsampling convolution. Otherwise, an upsampling convolution is used.
            `act (str)`: The activation function to use. Defaults to "relu". Can also be set to "leaky_relu".
            `use_dropout (bool)`: If True, dropout is applied to the output of the block.
        """
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

if __name__ == "__main__":
    # Test the CNNBlock class
    x = torch.randn(1, 7, 256, 256)
    block = Block(7, 64, down=False, act="relu", use_dropout=False)
    print(block(x).shape)