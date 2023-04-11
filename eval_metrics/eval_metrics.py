import torch
import math

def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.) -> float:
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1 (torch.Tensor): The first image. Must be a 3D tensor with shape (C, H, W).
        img2 (torch.Tensor): The second image. Must have the same shape as `img1`.
        max_val (float, optional): The maximum value of the pixels. Defaults to 1.

    Returns:
        float: The PSNR value between the two images.

    Raises:
        ValueError: If `img1` and `img2` have different shapes.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape. Got {img1.shape} and {img2.shape}")
    
    mse = torch.mean((img1 - img2)**2)
    if mse == 0:
        # If the images are identical, PSNR is infinity.
        return math.inf
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


import torch.nn.functional as F
from typing import Callable, Sequence, Union
def _gaussian(kernel_size: int, sigma: float) -> torch.Tensor:
    ksize_half = (kernel_size - 1) * 0.5
    kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
    return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

def _gaussian_or_uniform_kernel(kernel_size: Sequence[int], sigma: Sequence[float]) -> torch.Tensor:
    kernel_x = _gaussian(kernel_size[0], sigma[0])
    kernel_y = _gaussian(kernel_size[1], sigma[1])
    return torch.matmul(kernel_x.t(), kernel_y)  # (kernel_size, 1) * (1, kernel_size)

def ssim(tensor1, tensor2, window_size=11, size_average=True, full=False, max_val=1.):
    # Values below which we clamp the numerator and denominator of SSIM
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    window = _gaussian_or_uniform_kernel((window_size,window_size), (1.5,1.5))
    window = window.unsqueeze(0).unsqueeze(0)
    padding = window_size // 2
    mu1 = F.conv2d(tensor1, window, padding=padding, groups=tensor1.shape[1])
    mu2 = F.conv2d(tensor2, window, padding=padding, groups=tensor2.shape[1])

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(tensor1 * tensor1, window, padding=padding, groups=tensor1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(tensor2 * tensor2, window, padding=padding, groups=tensor2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(tensor1 * tensor2, window, padding=padding, groups=tensor1.shape[1]) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    if size_average:
        ssim_val = ssim_map.mean()
    else:
        ssim_val = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ssim_val, ssim_map
    else:
        return ssim_val
