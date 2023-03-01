import torch
from torch.nn import Module, Conv2d, Parameter, Softmax

""" Adapted from DANet: Dual Attention Network for Scene Segmentation 
Ref: https://arxiv.org/abs/1809.02983
by Zilong Huang, Ping Luo, Chen Change Loy, Xiaoou Tang
"""


class PAM_Module(Module):
    """Position attention module.
    """
    def __init__(self, in_dim):
        """    
    Args:
    -----
        `in_dim` (int): Number of input channels. it must be greater than or equal to 8, which is the ratio that determines the output channels of
        query_conv, key_conv

    Returns:
    --------
        torch.Tensor: Output tensor with the same shape as the input tensor."""
        
        super().__init__()
        self.chanel_in = in_dim
        self.OUT_RATIO = 8
        
        if in_dim < self.OUT_RATIO:
            raise ValueError("Input channels must be greater than or equal to out_ratio.")

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//self.OUT_RATIO, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//self.OUT_RATIO, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps, number of channels must be equal to in_dim in the constructor which is more than or equal to 8.
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps(B, C, H, W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
    
    
if __name__ == '__main__':
    print('Testing Position Attention Module...')
    n_channels = 98
    x = torch.randn(1, n_channels, 128, 128)
    pam = PAM_Module(n_channels)
    y = pam(x)
    print(y.shape)
    print('Testing Channel Attention Module...')
    cam = CAM_Module(n_channels)
    z = cam(x)
    print(z.shape)