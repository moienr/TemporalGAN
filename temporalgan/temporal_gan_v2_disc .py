import torch
import torch.nn as nn
from submodules.disc_cnn_block import CNNBlock
from submodules.cbam import CBAM, ChannelAttention, SpatialAttention
from submodules.scam import PAM

class Discriminator(nn.Module):
    """
    The Class that creates the Discriminator Model Object.

    Attributes
    ----------
    `in_channels`: int
        number of input channels (e.g. if RGB in_channels = 3)
    `features` : list
        number of layers in each hiddel layer as a list
    

    """
    def __init__(self, s2_in_channels=7, s1_in_channels = 1, s1_out_channels=1, features=[64, 128, 256, 512]):
        """
        Args:
        -----
            `s2_in_channels`: int (default: 7): number of input channels for Sentinel 2 + number of the lyars of ChangeMap (if used)
            `s1_in_channels`: int (default: 1): number of input channels for Sentinel 1 + number of the lyars of ChangeMap (if used)
            `s1_out_channels`: int (default: 1): the stacked Sentinel 1 output of the generator
            `features` : list (default: [64, 128, 256, 512]): number of layers in each hiddel layer as a list

        """
        super().__init__()
    
        # Adding the number of channels of the input to the number of channels of the output of the generator
        s2_in_channels = s2_in_channels + s1_out_channels 
        s1_in_channels = s1_in_channels + s1_out_channels 
        
        self.s2_init_cov = CNNBlock(s2_in_channels, features[0], stride=2, norm=False) # we don't want batch/instance norm in the first layer.
        self.s1_init_cov = CNNBlock(s1_in_channels, features[0], stride=2, norm=False) # we don't want batch/instance norm in the first layer.
        # Spatial Attention for input
        self.s2_pam_init = PAM(features[0])
        self.s1_pam_init = PAM(features[0])
        
        
        # encoder of S2
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            
            layers.append(
                CNNBlock(in_channels,
                        feature,
                        stride = 1 if feature==features[-1] else 2, # Stride in the last layer must be 1. this will turn a 32x32 into a 31x31
                        norm = True 
                        ))
            in_channels = feature # setting the in_channels to the last layer created
            
        self.middle_s2 = nn.Sequential(*layers)
        
        # encoder of S1
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            
            layers.append(
                CNNBlock(in_channels,
                        feature,
                        stride = 1 if feature==features[-1] else 2, # Stride in the last layer must be 1. this will turn a 32x32 into a 31x31
                        norm = True 
                        ))
            in_channels = feature # setting the in_channels to the last layer created
            
        self.middle_s1 = nn.Sequential(*layers)
        
        # CBAM before fusion
        self.fusion_cbam = CBAM(in_channels*2)
        
        # A the last layer to turn 512 channels into 1 and the 31x31 into 30x30 | the *2 is because we are concatenating the S2 and S1
        self.fuse_conv = nn.Conv2d(in_channels*2, 1, 4, stride=1, padding=1, padding_mode='reflect') 

        

    def forward(self, s2_in , s1_in, s1_out):
        """
        The Discriminator model takes, the Fake(generated) image and the Real Image and outputs a `30x30` patch (if input is 256x256) 
        of vlues which 1 corresponds to real and 0 to fake.
        This is called a PatchGAN.

        Parameters
        ----------
        `s2_in`: torch.Tensor : the input Sentinel 2 image
        `s1_in`: torch.Tensor : the input Sentinel 1 image
        `s1_out`: torch.Tensor : the output Sentinel 1 image of the generator (fake image)
        """
        s2_in = torch.cat([s2_in, s1_out], dim=1) # Concatenating the output of the generator with the real s2 image
        s1_in = torch.cat([s1_in, s1_out], dim=1) # Concatenating the output of the generator with the real s1 image
        
        s2_in = self.s2_pam_init(self.s2_init_cov(s2_in))
        s1_in = self.s1_pam_init(self.s1_init_cov(s1_in))
        
        s2_in = self.middle_s2(s2_in)
        s1_in = self.middle_s1(s1_in)
        
        concat = torch.cat([s2_in, s1_in], dim=1) # Concatenating the output of the two streams.
        concat = self.fusion_cbam(concat) # Applying CBAM before the last layer
        
        return self.fuse_conv(concat)


def test(summary=False, gpu=False):
    if gpu:
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: 
        device = torch.device('cpu')
        
    
    s2_in = torch.randn((1,8,256,256)).to(device)
    s1_in = torch.randn((1,2,256,256)).to(device)
    s1_out = torch.randn((1,1,256,256)).to(device)
    model = Discriminator(s2_in_channels=8, s1_in_channels=2, s1_out_channels=1).to(device)
    preds = model(s2_in, s1_in, s1_out)
    print(preds.shape)
    
    if summary:
        from torchinfo import summary
        summary(model, input_size=[(1, 8, 256, 256),(1, 2, 256, 256),(1,1,256,256)], device=device,col_names=["input_size", "output_size", "num_params"],
        col_width=20,
        row_settings=["var_names"])


if __name__ == "__main__": # testing the model
    test(summary=True,gpu=True)
    

