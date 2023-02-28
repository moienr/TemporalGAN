import torch
import torch.nn as nn
from submodules.disc_cnn_block import CNNBlock

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
    def __init__(self, in_channels=3,out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        
        layers = []
        in_channels = in_channels + out_channels # Input is for example 7 bands of S2 and output is 1 bands S1 so the total in of Disc is 7 + 1
        for feature in features:
            layers.append(
                CNNBlock(in_channels,
                        feature,
                        stride = 1 if feature==features[-1] else 2, # Stride in the last layer must be 1.
                        norm = False if feature == features[0] else True # we don't want batch/instance norm in the first layer.
                        ))
            in_channels = feature # setting the in_channels to the last layer created

        layers.append(
            nn.Conv2d(in_channels,1,4,stride=1, padding=1, padding_mode='reflect') # A the last layer to turn 512 channels into 1
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        The Discriminator model takes, the Fake(generated) image and the Real Image and outputs a `30x30` patch (if input is 256x256) 
        of vlues which 1 corresponds to real and 0 to fake.

        Parameters
        ----------
        `x` : Generated(fake) Image
        `y` : Real Image
        """
        x = torch.cat([x,y], dim=1) # Concatenating real and fake along the channels -> (batch_num, num_cahnnels*2 , 256 , 256) -> e.g: (1,6,256,256)
        return self.model(x)


if __name__ == "__main__": # testing the model
    x = torch.randn((1,3,256,256))
    y = torch.randn((1,1,256,256))
    model = Discriminator()
    preds = model(x,y)
    print(preds.shape)