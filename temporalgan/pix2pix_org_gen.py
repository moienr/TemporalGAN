import torch
import torch.nn as nn
from submodules.gen_cnn_block import Block

class Generator(nn.Module):
    """
    A generator model for image-to-image translation using a U-Net architecture.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to 3.
        out_channels (int, optional): Number of output channels. Defaults to 1.
        features (int, optional): Number of features in the first layer. Defaults to 64. after that the number of features will be doubled in each layer.
    """
    def __init__(self, in_channels=3,out_channels=1, features=64):
        super().__init__()
        # Initial downsampling layer
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)) #128
        
        # Downsample blocks 
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False) # 64
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False) # 32
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False) # 16
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 8
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 4
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 2
        
        # Bottleneck layer (no downsampling)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
            ) 
        # Upsample blocks
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)

        # Final upsampling layer
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


def test(summary=False):
    x = torch.randn((32, 7, 256, 256))
    model = Generator(in_channels=7, features=64)
    preds = model(x)
    print(preds.shape)

    if summary:
        from torchinfo import summary
        summary(model, input_size=[(16, 7, 256, 256)], device='cpu',col_names=["input_size", "output_size", "num_params"],
        col_width=20,
        row_settings=["var_names"])

if __name__ == "__main__":
    test(summary=True)