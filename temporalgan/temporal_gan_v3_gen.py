"""TemporalGAN Generator Version 2
In this version I use a dual stream for encoding part, which fuses the Sentiel2_T1 with Sentinel1_T2.
Attention
--------
For the attention we use
    * The Squeeze and Excitation from CBAM module before the fusion of the two streams(before the bottleneck).
    * The Position Attention Module(PAM) for the 128x128, 64x64, 32x32, and 16x16 layers.
    * CBAM for the 8x8, 4x4 layers, but only for the skip connection, the normal data gets donwsampled.

# Whats new since V2?
in this version, instead of first convolution downsampling layer, we use a convoluiton layer with no downsampling,
this will ouput a 256x256 tensor after the first convolution layer, the rest of the architecture is the same as V2.
removing the first downsampling layer, will result in every layer to be 2 times bigger than the previous version.
for exapmle out buttleneck input is 4x4 instead of 2x2, and the output of the bottle neck which is the smallest tesnor,
will be a 2x2 tensor instead of 1x1 tensor.

* for the skip connection, of the first layer, we use PAM but with the downsample parameter set to 2, for RAM conservation.
* since the second layer(128x128) now has more chnnels, the RAM usage was high, so I added downsaple to pam2 as well.

"""

import torch
import torch.nn as nn
from submodules.gen_cnn_block import Block
from submodules.cbam import ChannelAttention, CBAM
from submodules.scam import PAM

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
        """
        Args:
        ---
            `s2_in_channels` (int, optional): Number of input channels for the Sentinel2 image. Defaults to 3.
            `s1_in_channels` (int, optional): Number of input channels for the Sentinel1 image. Defaults to 1.
            out_channels (int, optional): Number of output channels. Defaults to 1.
            `features` (int, optional): Number of features in the first layer. Defaults to 64. after that the number of features will be doubled in each layer.
            
        Important Attributes:
        ---
            `s2_initial_down` (nn.Sequential): Initial downsampling layer for S2
            `s1_initial_down` (nn.Sequential): Initial downsampling layer for S1
            `final_up` (nn.Sequential): Final upsampling layer
            
            * These three attributes are using a 5x5 kernel, with stride 1, and padding 2,**
            I don't know whether this is the best choice, so If you saw some bluriness, use a 3x3 kernel with stride 1 and padding 1.
            My arguue is that since S1 and S2 sometimes have a little offset in geo-location, the 5x5 kernel will help to align them.
        """
        super().__init__()
        # Initial downsampling layer for S2
        IN_OUT_KSP = (5, 1, 2) # initial layer, and finalup layer kernel size, stride, padding, 
        # I still havn't decided whether to use 3x3 or 5x5, so I will leave it as a variable.
        # In case of 3x3, the padding will be 1, and the stride will be 1. so IN_OUT_KSP = (3, 1, 1)
        self.s2_initial_down = nn.Sequential(
            nn.Conv2d(s2_in_channels, features, IN_OUT_KSP[0], IN_OUT_KSP[1], IN_OUT_KSP[2], padding_mode="reflect"),
            nn.LeakyReLU(0.2)) #256
        
        # Initial downsampling layer for S1
        self.s1_initial_down = nn.Sequential(
            nn.Conv2d(s1_in_channels, features, IN_OUT_KSP[0], IN_OUT_KSP[1], IN_OUT_KSP[2], padding_mode="reflect"),
            nn.LeakyReLU(0.2)) #256
        
        # Downsample blocks for Sentinel-2
        self.s2_pam_init  = PAM(features,downsample=2) # input from the initial_down layer / output goes to the down1 layer
        self.s2_down1 = Block(features    , features * 2, down=True, act="leaky", use_dropout=False) # 128
        self.s2_pam1 = PAM(features * 2, downsample=2) # input from the down1 layer / output goes to the down2 layer
        self.s2_down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False) # 64
        self.s2_pam2 = PAM(features * 4) # input from the down2 layer / output goes to the down3 layer
        self.s2_down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False) # 32
        self.s2_pam3 = PAM(features * 8) # input from the down3 layer / output goes to the down4 layer
        self.s2_down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 64
        self.s2_cbam4 = CBAM(features * 8) # input from the down4 layer / output goes to the down5 layer
        self.s2_down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 8
        self.s2_cbam5 = CBAM(features * 8) # input from the down5 layer / output goes to the down6 layer
        self.s2_down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 4 * 512
        
        # Downsample blocks from Sentinel-1
        self.s1_pam_init  = PAM(features,downsample=2) # input from the initial_down layer / output goes to the down1 layer
        self.s1_down1 = Block(features    , features * 2, down=True, act="leaky", use_dropout=False) # 128
        self.s1_pam1 = PAM(features * 2, downsample=2) # input from the down1 layer / output goes to the down2 layer
        self.s1_down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False) # 64
        self.s1_pam2 = PAM(features * 4) # input from the down2 layer / output goes to the down3 layer
        self.s1_down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False) # 32
        self.s1_pam3 = PAM(features * 8) # input from the down3 layer / output goes to the down4 layer
        self.s1_down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 64
        self.s1_cbam4 = CBAM(features * 8) # input from the down4 layer / output goes to the down5 layer
        self.s1_down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 8
        self.s1_cbam5 = CBAM(features * 8) # input from the down5 layer / output goes to the down6 layer
        self.s1_down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) # 4 * 512
        
        # Channel Attention Module
        self.ca = ChannelAttention(features * 8 * 2) # The *2 is because we fuse the two streams.
        # Bottleneck layer (no downsampling)
        self.bottleneck = nn.Sequential(
        nn.Conv2d(features * 8 * 2, features * 8, 4, 2, 1), nn.ReLU()) # The *2 is because we fuse the two streams. output is 2x2
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
        d1_s2 = self.s2_pam_init(self.s2_initial_down(s2))
        d2_s2 = self.s2_pam1(self.s2_down1(d1_s2))
        d3_s2 = self.s2_pam2(self.s2_down2(d2_s2))
        d4_s2 = self.s2_pam3(self.s2_down3(d3_s2))
        d5_s2 = self.s2_down4(d4_s2)
        d6_s2 = self.s2_down5(d5_s2)
        d7_s2 = self.s2_down6(d6_s2)
        # Now we do the same for the Sentinel1
        d1_s1 = self.s1_pam_init(self.s1_initial_down(s1))
        d2_s1 = self.s1_pam1(self.s1_down1(d1_s1))
        d3_s1 = self.s1_pam2(self.s1_down2(d2_s1))
        d4_s1 = self.s1_pam3(self.s1_down3(d3_s1))
        # No Spatial Attention Module for the last two layers, when downsampling, but we send their CBAM to be concatenated with the upsampled layers.
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
        # Concatenate the upsampled layer, with the CBAM of the corresponding downsampled layer, for up3 and up4.
        up3 = self.up3(torch.cat([up2, self.s2_cbam5(d6_s2), self.s1_cbam5(d6_s1)], 1))
        up4 = self.up4(torch.cat([up3, self.s2_cbam4(d5_s2), self.s1_cbam4(d5_s1)], 1))
        up5 = self.up5(torch.cat([up4, d4_s2, d4_s1], 1))
        up6 = self.up6(torch.cat([up5, d3_s2, d3_s1], 1))
        up7 = self.up7(torch.cat([up6, d2_s2, d2_s1], 1))
        # Final upsampling which outputs the final image with only 1 channel.
        return self.final_up(torch.cat([up7, d1_s2, d1_s1], 1))


def test(summary=False, gpu=False):
    if gpu:
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: 
        device = torch.device('cpu')
        
    
    s2 = torch.randn((1, 7, 256, 256)).to(device)
    s1 = torch.randn((1, 1, 256, 256)).to(device)
    
    model = Generator(s2_in_channels=7, s1_in_channels=1, features=64)
    model.to(device)
    preds = model(s2,s1)
    print(preds.shape)
    if summary:
        from torchinfo import summary
        summary(model, input_size=[(1, 7, 256, 256),(1, 1, 256, 256)], device=device,col_names=["input_size", "output_size", "num_params"],
        col_width=20,
        row_settings=["var_names"])

if __name__ == "__main__":
    test(summary=True, gpu=True)
    