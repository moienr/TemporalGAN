import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    """
    BAM's channel-wise attention .
    Args:
    gate_channel (int): the number of input channels to the module
    reduction_ratio (int, optional): the reduction ratio used to reduce the number of channels in the module
    num_layers (int, optional): the number of fully-connected layers used in the module

    Attributes:
        gate_c (nn.Sequential): a sequential module that contains the fully-connected layers of the channel gate

    Methods:
        forward(in_tensor): computes the forward pass of the module

    Example:
        gate = ChannelGate(gate_channel=64, reduction_ratio=16, num_layers=2)
        x = torch.randn(1, 64, 32, 32)
        out = gate(x)
    """
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        """
        Initializes a new instance of the ChannelGate module.

        Args:
            gate_channel (int): the number of input channels to the module
            reduction_ratio (int, optional): the reduction ratio used to reduce the number of channels in the module
            num_layers (int, optional): the number of fully-connected layers used in the module 
        """
        super(ChannelGate, self).__init__()
        #self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        """
        Computes the forward pass of the ChannelGate module.

        Args:
            in_tensor (torch.Tensor): the input tensor to the module

        Returns:
            A tensor with the same shape as the input tensor, where each channel has been scaled by a channel-wise attention weight.
        """
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatialGate(nn.Module):
    """
    BAM's spatial attention.
    
    Args:
    gate_channel (int): the number of input channels to the module
    reduction_ratio (int, optional): the reduction ratio used to reduce the number of channels in the module
    dilation_conv_num (int, optional): the number of dilated convolution layers used in the module
    dilation_val (int, optional): the dilation value used in the dilated convolution layers

    Attributes:
        gate_s (nn.Sequential): a sequential module that contains the convolutional layers of the spatial gate

    Methods:
        forward(in_tensor): computes the forward pass of the module

    Example:
        gate = SpatialGate(gate_channel=64, reduction_ratio=16, dilation_conv_num=2, dilation_val=4)
        x = torch.randn(1, 64, 32, 32)
        out = gate(x)

    """
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        """
        Initializes a new instance of the SpatialGate module.

        Args:
            gate_channel (int): the number of input channels to the module
            reduction_ratio (int, optional): the reduction ratio used to reduce the number of channels in the module
            dilation_conv_num (int, optional): the number of dilated convolution layers used in the module
            dilation_val (int, optional): the dilation value used in the dilated convolution layers
        """
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        """
        Computes the forward pass of the SpatialGate module.

        Args:
            in_tensor (torch.Tensor): the input tensor to the module

        Returns:
            A tensor with the same shape as the input tensor, where each spatial location has been scaled by a spatial attention weight.
        """
        return self.gate_s( in_tensor ).expand_as(in_tensor)
class BAM(nn.Module):
    """
    Args:
    - gate_channel (int): The number of channels used in the gating mechanism.
    - use_c (bool): If True, the BAM module includes a channel attention mechanism.
    - use_s (bool): If True, the BAM module includes a spatial attention mechanism.

    Forward arguments:
    - in_tensor (tensor): A tensor of shape (batch_size, channels, height, width) representing the input tensor.

    Returns:
    - out_tensor (tensor): A tensor of shape (batch_size, channels, height, width) representing the output tensor after
    applying the BAM module.

    Raises:
    - ValueError: If use_c and use_s are both False.
    """
    def __init__(self, gate_channel, use_c=False, use_s=True):
        """_summary_

        Args:
            gate_channel (int): The number of channels used in the gating mechanism.
            use_c (bool): If True, the BAM module includes a channel attention mechanism.
            use_s (bool): If True, the BAM module includes a spatial attention mechanism.
        """
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel) if use_c else None
        self.spatial_att = SpatialGate(gate_channel) if use_s else None
        
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.sigmoid3 = nn.Sigmoid()
        
    def forward(self,in_tensor):
        """
        Args:
            in_tensor (tensor): A tensor of shape (batch_size, channels, height, width) representing the input tensor.
            * batch_size should be greater than 1, since we are using batch statistics for batch normalization.
        """

        if self.channel_att and self.spatial_att:
            att = 1 + self.sigmoid1( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        elif self.channel_att:
            att = 1 + self.sigmoid2( self.channel_att(in_tensor) )
        elif self.spatial_att:
            att = 1 + self.sigmoid3( self.spatial_att(in_tensor) )
        else:
            raise ValueError( 'use_c and use_s should not be both False.')
        return att * in_tensor


if __name__ == "__main__":
    divice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("BAM")
    bam = BAM(64).to(divice)
    rand_tensor = torch.randn(2,64,256,256).to(divice)
    out = bam(rand_tensor)
    print("out.shape: ", out.shape)