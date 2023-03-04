import torch
from torch.nn import Module, Conv2d, Parameter, Softmax, MaxPool2d, Upsample, AvgPool2d

""" Modified the code from DANet: Dual Attention Network for Scene Segmentation 
Ref: https://arxiv.org/abs/1809.02983
by Zilong Huang, Ping Luo, Chen Change Loy, Xiaoou Tang
"""


class PAM(Module):
    """ # Position attention module.
    * We introduced a new input: `downsample`, this is very useful when the input tensor is very large, and you want to reduce the memory usage.
    
    ### How it works:
    
    when `downsample` is set, the input tensor will be downsampled by this factor, to create the query and key tensors, 
    this will reduce the Matrix Multiplication (MM) computation, and saves RAM.
    The attention map will still be upsampled to the original size, so the output tensor will have the same size as the input tensor.
    
    """
    def __init__(self, in_dim, downsample: int = None):
        """    
    Args:
    -----
        `in_dim` (int): Number of input channels. it must be greater than or equal to 8, which is the ratio that determines the output channels of
        query_conv, key_conv
        
        `downsample` (bool): If set, the input tensor will be downsampled by this factor, to save RAM. Default: None.
        The output tensor will still be upsampled to the original size, so the output tensor will have the same size as the input tensor.
        This is very useful when the input tensor is very large, and you want to reduce the memory usage.
        * suggested for input tensors with size (b, c, h, w) where h and w are greater than 128. or the batch size is too large.
        * the input height and width should be devisible by downsample, otherwise, an error will be raised.
        * good value for downsample would be a number where: hight/downsample and width/downsample are equal or less than 128.
    
    Attributes:
    ----------
        `pool`: MaxPool2d operation to downsample the input tensor, which creates the query and key tensors.
        `upsample`: Upsample operation to upsample the attention map to the original size, so it could be applied to the input tensor.

    Returns:
    --------
        torch.Tensor: Output tensor with the same shape as the input tensor."""
        
        super().__init__()
        self.chanel_in = in_dim
        self.OUT_RATIO = 8
        self.downsample = downsample
        if in_dim < self.OUT_RATIO:
            raise ValueError("Input channels must be greater than or equal to out_ratio.")
        
        if downsample:
            self.pool = AvgPool2d(kernel_size=downsample, stride=downsample)
            self.upsample = Upsample(scale_factor=downsample, mode='bilinear', align_corners=True)

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
        # getting the height and width of the input tensor
        b, c, h, w = x.size()
        # value error if the input tesnor's height and width are not devisible by downsampling factor
        if h % self.downsample != 0 or w % self.downsample != 0:
            raise ValueError("Height and width must be divisible by downsample factor.")
        
        if self.downsample:
            # downsample the input tensor to the size of (b, c, h/2, w/2)
            x_down = self.pool(x)
        
        m_batchsize, C, height, width = x_down.size()
        proj_query = self.query_conv(x_down).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x_down).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x_down).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        
        if self.upsample:
            out = self.upsample(out)
            
        out = self.gamma*out + x
        return out


class CAM(Module):
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
    
def test(summary=False,gpu=False):
    if gpu:
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: 
        device = torch.device('cpu')
        
    
    print('Testing Position Attention Module...')
    n_channels = 16
    x = torch.randn(1, n_channels, 512, 512).to(device)
    pam = PAM(n_channels, downsample=4)
    pam = pam.to(device)
    y = pam(x)
    print(y.shape)
    
    print('Testing Channel Attention Module...')
    cam = CAM(n_channels)
    cam = cam.to(device)
    z = cam(x)
    print(z.shape)
    
    if summary:
        from torchinfo import summary
        summary(pam, input_size=(1,n_channels, 128, 128), device=device,col_names=["input_size", "output_size", "num_params"],
            col_width=20,
            row_settings=["var_names"])
if __name__ == '__main__':
    test(summary=False,gpu=True)
    