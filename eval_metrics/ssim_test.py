import torch
from torchvision.transforms import functional as F
from ignite.metrics import SSIM

# Create two dummy tensors
tensor1 = torch.rand(1, 1, 256, 256)
tensor2 = tensor1.clone() + torch.rand(1, 1, 256, 256) * 0.2
# Apply SSIM on the tensors
ssim = SSIM(data_range=1.0)
ssim.update((tensor1, tensor2))
print(ssim.compute())


from eval_metrics import ssim as my_ssim
print(my_ssim(tensor1, tensor2))
