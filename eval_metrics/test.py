import torch
from torchvision.transforms import functional as F
from ssim import WSSIM
from ignite.metrics import SSIM as SSIM_ignite
# Create two dummy tensors
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


tensor1 = torch.rand(7, 3, 256, 256).to(device)
tensor2 = tensor1.clone() + torch.rand(7, 3, 256, 256).to(device) * 0.2
weight_map = torch.rand(7, 3, 256, 256).to(device)

wssim = WSSIM(data_range=1.0)
print(wssim((tensor1, tensor2), weight_map))
print(wssim((tensor2, tensor1), weight_map))
print(wssim((tensor1, tensor2), weight_map))




from psnr import wpsnr

print(wpsnr((tensor1, tensor2), weight_map))