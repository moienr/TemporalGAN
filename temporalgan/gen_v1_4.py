"""TemporalGAN Generator Version 1
In this version I use a dual stream for encoding part, which fuses the Sentiel2_T1 with Sentinel1_T2.
We use no Attetention module.

## Version 1.2: Initial_down -> No DownSample
## Version 1.3: Channel Attentino at the bottleneck
## Version 1.4: Channel Attentino at the bottleneck + CBAM at skip connections
"""

import torch
import torch.nn as nn
from submodules.gen_cnn_block import Block
from submodules.cbam import ChannelAttention, CBAM

class Generator(nn.Module):
    """
    A generator model for image-to-image translation using a U-Net architecture.

    Args:
        s2_in_channels (int, optional): Number of input channels for the Sentinel2 image. Defaults to 3.
        s1_in_channels (int, optional): Number of input channels for the Sentinel1 image. Defaults to 1.
        out_channels (int, optional): Number of output channels. Defaults to 1.
        features (int, optional): Number of features in the first layer. Defaults to 64. after that the number of features will be doubled in each layer.
    """
    def __init__(self, s2_in_channels=3,s1_in_channels = 1,out_channels=1, features=64):
        super().__init__()
        # Initial downsampling layer for S2
        IN_OUT_KSP = (3, 1, 1) # initial layer, and finalup layer kernel size, stride, padding, 
        # I still havn't decided whether to use 3x3 or 5x5, so I will leave it as a variable.
        # In case of 3x3, the padding will be 1, and the stride will be 1. so IN_OUT_KSP = (3, 1, 1)
        # In case of 5x5, the padding will be 2, and the stride will be 1. so IN_OUT_KSP = (5, 1, 2)
        self.s2_initial_down = nn.Sequential(
            nn.Conv2d(s2_in_channels, features, IN_OUT_KSP[0], IN_OUT_KSP[1], IN_OUT_KSP[2], padding_mode="reflect"),
            nn.LeakyReLU(0.2)) #256
        
        # Initial downsampling layer for S1
        self.s1_initial_down = nn.Sequential(
            nn.Conv2d(s1_in_channels, features, IN_OUT_KSP[0], IN_OUT_KSP[1], IN_OUT_KSP[2], padding_mode="reflect"),
            nn.LeakyReLU(0.2)) #256
        
        # Downsample blocks for Sentinel-2
        self.s2_cbam_init  = CBAM(features) # input from the initial_down layer / output goes to the down1 layer
        self.s2_down1 = Block(features    , features * 2, down=True, act="leaky", use_dropout=False) # 128
        self.s2_cbam1 = CBAM(features * 2) # input from the down1 layer / output goes to the down2 layer
        self.s2_down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False) # 64
        self.s2_cbam2 = CBAM(features * 4) # input from the down2 layer / output goes to the down3 layer
        self.s2_down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False) # 32
        self.s2_cbam3 = CBAM(features * 8) # input from the down3 layer / output goes to the down4 layer
        self.s2_down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 64
        self.s2_cbam4 = CBAM(features * 8) # input from the down4 layer / output goes to the down5 layer
        self.s2_down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 8
        self.s2_cbam5 = CBAM(features * 8) # input from the down5 layer / output goes to the down6 layer
        self.s2_down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 4 * 512
        
        # Downsample blocks from Sentinel-1
        self.s1_cbam_init  = CBAM(features) # input from the initial_down layer / output goes to the down1 layer
        self.s1_down1 = Block(features    , features * 2, down=True, act="leaky", use_dropout=False) # 128
        self.s1_cbam1 = CBAM(features * 2) # input from the down1 layer / output goes to the down2 layer
        self.s1_down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False) # 64
        self.s1_cbam2 = CBAM(features * 4) # input from the down2 layer / output goes to the down3 layer
        self.s1_down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False) # 32
        self.s1_cbam3 = CBAM(features * 8) # input from the down3 layer / output goes to the down4 layer
        self.s1_down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 64
        self.s1_cbam4 = CBAM(features * 8) # input from the down4 layer / output goes to the down5 layer
        self.s1_down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 8
        self.s1_cbam5 = CBAM(features * 8) # input from the down5 layer / output goes to the down6 layer
        self.s1_down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 4 * 512
        # Channel Attention Module
        self.ca = ChannelAttention(features * 8 * 2) # The *2 is because we fuse the two streams.
        # Bottleneck layer (no downsampling)
        self.bottleneck = nn.Sequential(
        nn.Conv2d(features * 8 * 2, features * 8, 4, 2, 1), nn.ReLU()) # The *2 is because we fuse the two streams.
        
        # Upsample blocks
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 3, features * 8, down=False, act="relu", use_dropout=True) # The *3 is because we fuse the two streams + the bottleneck.
        self.up3 = Block(features * 8 * 3, features * 8, down=False, act="relu", use_dropout=True) # The *3 is because we fuse the two streams + last upsample.
        self.up4 = Block(features * 8 * 3, features * 8, down=False, act="relu", use_dropout=False)
        self.up5 = Block(features * 8 * 3, features * 4, down=False, act="relu", use_dropout=False)
        self.up6 = Block(features * 4 * 3, features * 2, down=False, act="relu", use_dropout=False)
        self.up7 = Block(features * 2 * 3, features    , down=False, act="relu", use_dropout=False)

        # Final upsampling layer
        self.final_up = nn.Sequential(
            nn.Conv2d(features * 3, out_channels, IN_OUT_KSP[0], IN_OUT_KSP[1], IN_OUT_KSP[2]),
            nn.Tanh(),
        )
    def forward(self, s2: torch.Tensor , s1:torch.Tensor) -> torch.Tensor:
        # First we do the encoding part for the Sentinel2
        d1_s2 = self.s2_initial_down(s2)
        d2_s2 = self.s2_down1(d1_s2)
        d3_s2 = self.s2_down2(d2_s2)
        d4_s2 = self.s2_down3(d3_s2)
        d5_s2 = self.s2_down4(d4_s2)
        d6_s2 = self.s2_down5(d5_s2)
        d7_s2 = self.s2_down6(d6_s2)
        # Now we do the same for the Sentinel1
        d1_s1 = self.s1_initial_down(s1)
        d2_s1 = self.s1_down1(d1_s1)
        d3_s1 = self.s1_down2(d2_s1)
        d4_s1 = self.s1_down3(d3_s1)
        d5_s1 = self.s1_down4(d4_s1)
        d6_s1 = self.s1_down5(d5_s1)
        d7_s1 = self.s1_down6(d6_s1)
        # Now we fuse the two streams
        d7 = torch.cat([d7_s2, d7_s1], 1)
        # Channel Attention
        d7 = self.ca(d7) # Channel Attention to learn the importance of each stream, before fusing them.
        # Bottleneck
        bottleneck = self.bottleneck(d7)
        # Now we do the decoding part
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, self.s2_cbam5(d6_s2), self.s1_cbam5(d6_s1)], 1))
        up4 = self.up4(torch.cat([up3, self.s2_cbam4(d5_s2), self.s1_cbam4(d5_s1)], 1))
        up5 = self.up5(torch.cat([up4, self.s2_cbam3(d4_s2), self.s1_cbam3(d4_s1)], 1))
        up6 = self.up6(torch.cat([up5, self.s2_cbam2(d3_s2), self.s2_cbam2(d3_s1)], 1))
        up7 = self.up7(torch.cat([up6, self.s2_cbam1(d2_s2), self.s1_cbam1(d2_s1)], 1))
        # Final upsampling which outputs the final image with only 1 channel.
        return self.final_up(torch.cat([up7, self.s2_cbam_init(d1_s2), self.s1_cbam_init(d1_s1)], 1))


def test(summary=False):
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    s2 = torch.rand((1, 7, 256, 256)).to(torch.float32).to(device)
    s1 = torch.rand((1, 1, 256, 256)).to(torch.float32).to(device)
    
    model = Generator(s2_in_channels=7, s1_in_channels=1, features=64).to(device)

    preds = model(s2,s1)
    print(preds.shape)
    print(torch.min(preds),torch.mean(preds),torch.max(preds),preds.dtype)
    if summary:
        from torchinfo import summary
        summary(model, input_size=[(1, 7, 256, 256),(1, 1, 256, 256)], device='cpu',col_names=["input_size", "output_size", "num_params"],
        col_width=20,
        row_settings=["var_names"])

if __name__ == "__main__":
    test(summary=False)
    