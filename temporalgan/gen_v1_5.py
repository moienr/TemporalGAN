"""TemporalGAN Generator Version 1
In this version I use a dual stream for encoding part, which fuses the Sentiel2_T1 with Sentinel1_T2.
We use no Attetention module.

## Version 1.2: Initial_down -> No DownSample
## Version 1.3: Channel Attentino at the bottleneck
## Version 1.5: GLAM at 8x8 downstreams
"""

import torch
import torch.nn as nn
from submodules.gen_cnn_block import Block
from submodules.cbam import ChannelAttention
from submodules.glam.glam import GLAM
from gen_v1_3 import Generator as Generator_v1_3

class Generator(Generator_v1_3):
    def __init__(self, s2_in_channels=3, s1_in_channels=1, out_channels=1, features=64, num_reduced_channels=32):
        super().__init__(s2_in_channels, s1_in_channels, out_channels, features)
        # self.glam5_s2 = GLAM(in_channels=features * 8, num_reduced_channels=num_reduced_channels, feature_map_size=8,kernel_size=5) 
        
        self.glam5_s2 = GLAM(in_channels=features * 8, num_reduced_channels=num_reduced_channels, feature_map_size=8,kernel_size=3) 
        self.glam5_s1 = GLAM(in_channels=features * 8, num_reduced_channels=num_reduced_channels, feature_map_size=8,kernel_size=3)
        self.down5_s2 = nn.Sequential(self.down5_s2, self.glam5_s2)
        self.down5_s1 = nn.Sequential(self.down5_s1, self.glam5_s1)
        



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
    