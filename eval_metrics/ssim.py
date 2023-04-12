from typing import Callable, Sequence, Union

import torch
import torch.nn.functional as F

__all__ = ["WSSIM"]

class WSSIM():
    """
    Computes the Weighted Structural Similarity Index (W-SSIM) between two images. W-SSIM measures the similarity between
    two images while taking into account the presence of different regions with different perceptual importance.
    The weight map can be used to emphasize or de-emphasize certain regions in the image. 

    Args
    ---
        `data_range` (Union[int, float]): The data range of the image. For instance, if the image is of type torch.uint8,
            data_range should be set to 255. If the image is of type torch.float32, data_range should be set to 1.0.
        `kernel_size` (Union[int, Sequence[int]]): The size of the Gaussian or uniform filter kernel. If an integer is
            passed, a square kernel of that size will be used. If a sequence is passed, it should contain two integers
            representing the height and width of the kernel.
        `sigma` (Union[float, Sequence[float]]): The standard deviation of the Gaussian filter kernel. If a float is
            passed, the same value will be used for both the horizontal and vertical directions. If a sequence is passed,
            it should contain two floats representing the standard deviation in the horizontal and vertical directions.
        `k1` (float): A constant used to stabilize the SSIM calculation. Default is 0.01.
        `k2` (float): A constant used to stabilize the SSIM calculation. Default is 0.03.
        `gaussian` (bool): Whether to use a Gaussian or uniform filter kernel. Default is True.
        `output_transform` (Callable): A callable that takes in the output tensor and applies a transformation to it.
            Default is the identity function.

    Raises
    ---
        ValueError: If kernel_size is not an integer or a sequence of integers, or if sigma is not a float or a
            sequence of floats.
        ValueError: If kernel_size or sigma contain zero or negative numbers, or if kernel_size contains an even number.
        TypeError: If y_pred and y do not have the same data type.
        ValueError: If y_pred and y do not have the same shape.
        ValueError: If y_pred and y do not have a BxCxHxW shape.
        TypeError: If weight_map and y do not have the same data type.
        ValueError: If weight_map does not have the same B, H, and W as y, or if weight_map has more than one channel and
            has a different number of channels than y.

    Attributes
    ---
        `c1` (float): The value of k1 * data_range squared.
        `c2` (float): The value of k2 * data_range squared.
        `pad_h` (int): The number of padding pixels to add to the top and bottom of the image.
        `pad_w` (int): The number of padding pixels to add to the left and right of the image.
        `kernel_size` (Sequence[int]): The size of the Gaussian or uniform filter kernel.
        `sigma` (Sequence[float]): The standard deviation of the Gaussian filter kernel.
        `_device` (Union[str, torch.device]): The device where the computation will take place, Atuomatically set to the `y` device
        `_kernel` (torch.Tensor): The Gaussian or uniform filter kernel.
        
    Example
    ---
    ```
    tensor1 = torch.rand(7, 3, 256, 256).to(device)
    tensor2 = tensor1.clone() + torch.rand(7, 3, 256, 256).to(device) * 0.7
    weight_map = torch.rand(7, 3, 256, 256).to(device)

    wssim = WSSIM(data_range=1.0)
    print(wssim((tensor1, tensor2), weight_map))
    ```

    """
    def __init__(
        self,
        data_range: Union[int, float],
        kernel_size: Union[int, Sequence[int]] = (11, 11),
        sigma: Union[float, Sequence[float]] = (1.5, 1.5),
        k1: float = 0.01,
        k2: float = 0.03,
        gaussian: bool = True,
        output_transform: Callable = lambda x: x,
    ):
        """
        Args
        ---
            `data_range` (Union[int, float]): The data range of the image. For instance, if the image is of type torch.uint8,
                data_range should be set to 255. If the image is of type torch.float32, data_range should be set to 1.0.
            `kernel_size` (Union[int, Sequence[int]]): The size of the Gaussian or uniform filter kernel. If an integer is
                passed, a square kernel of that size will be used. If a sequence is passed, it should contain two integers
                representing the height and width of the kernel.
            `sigma` (Union[float, Sequence[float]]): The standard deviation of the Gaussian filter kernel. If a float is
                passed, the same value will be used for both the horizontal and vertical directions. If a sequence is passed,
                it should contain two floats representing the standard deviation in the horizontal and vertical directions.
            `k1` (float): A constant used to stabilize the SSIM calculation. Default is 0.01.
            `k2` (float): A constant used to stabilize the SSIM calculation. Default is 0.03.
            `gaussian` (bool): Whether to use a Gaussian or uniform filter kernel. Default is True.
            `output_transform` (Callable): A callable that takes in the output tensor and applies a transformation to it.
                Default is the identity function.
        """
        
        if isinstance(kernel_size, int):
            self.kernel_size: Sequence[int] = [kernel_size, kernel_size]
        elif isinstance(kernel_size, Sequence):
            self.kernel_size = kernel_size
        else:
            raise ValueError("Argument kernel_size should be either int or a sequence of int.")

        if isinstance(sigma, float):
            self.sigma: Sequence[float] = [sigma, sigma]
        elif isinstance(sigma, Sequence):
            self.sigma = sigma
        else:
            raise ValueError("Argument sigma should be either float or a sequence of float.")

        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(f"Expected kernel_size to have odd positive number. Got {kernel_size}.")

        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected sigma to have positive number. Got {sigma}.")

        super().__init__()
        self.gaussian = gaussian
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        
        

    def _uniform(self, kernel_size: int) -> torch.Tensor:
        max, min = 2.5, -2.5
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=self._device)
        for i, j in enumerate(kernel):
            if min <= j <= max:
                kernel[i] = 1 / (max - min)
            else:
                kernel[i] = 0

        return kernel.unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian(self, kernel_size: int, sigma: float) -> torch.Tensor:
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=self._device)
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian_or_uniform_kernel(self, kernel_size: Sequence[int], sigma: Sequence[float]) -> torch.Tensor:
        if self.gaussian:
            kernel_x = self._gaussian(kernel_size[0], sigma[0])
            kernel_y = self._gaussian(kernel_size[1], sigma[1])
        else:
            kernel_x = self._uniform(kernel_size[0])
            kernel_y = self._uniform(kernel_size[1])

        return torch.matmul(kernel_x.t(), kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    def __call__(self, compare_tesnors: Sequence[torch.Tensor], weight_map: torch.Tensor = None) -> None:
        """_summary_

        Args:
            compare_tesnors (Sequence[torch.Tensor]): The two q tensors to compare. They must have the same shape. (y and y_pred)
            weight_map (torch.Tensor, optional): The weight map to use. Defaults to None. if None, use ones, which is the same as Normal SSIM.

        Raises:
            TypeError: The data types of the two tensors must be the same and the same.
            ValueError: y and y_pred must have the same shape.
            ValueError: arrays must have 4 dimensions. (B, C, H, W)
            TypeError: Weight map must have the same data type as y and y_pred.
            ValueError: Weight map shoud have 4 dimensions. (B, C, H, W)
            ValueError: Weight map shoud have the same B, H, W as y and y_pred. and C = 1 or C = C of y and y_pred.
        Returns:
            torch.Tensor: The Weighted SSIM score between y_pred and y, weighted by weight_map.
        """
  
        self._device = compare_tesnors[0].device
        self._kernel = self._gaussian_or_uniform_kernel(kernel_size=self.kernel_size, sigma=self.sigma)
        if weight_map is None: # if no weight map is provided, use ones | this will be the same is normal SSIM which uses torch.mean()
            weight_map = torch.ones_like(compare_tesnors[0]).to(compare_tesnors[0].device)
                
        y_pred, y = compare_tesnors[0].detach(), compare_tesnors[1].detach()
        weight_map = weight_map.detach()
        
        if y_pred.dtype != y.dtype:
            raise TypeError(
                f"Expected y_pred and y to have the same data type. Got y_pred: {y_pred.dtype} and y: {y.dtype}."
            )

        if y_pred.shape != y.shape:
            raise ValueError(
                f"Expected y_pred and y to have the same shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

        if len(y_pred.shape) != 4 or len(y.shape) != 4:
            raise ValueError(
                f"Expected y_pred and y to have BxCxHxW shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )


        if weight_map.dtype != y.dtype:
            raise TypeError(
                f"Expected weight_map and y to have the same data type. Got weight_map: {weight_map.dtype} and y: {y.dtype}."
            )
        if len(weight_map.shape) != 4:
            raise ValueError(
                f"Expected weight_map to have BxCxHxW shape. Got weight_map: {weight_map.shape}."
            )
        if weight_map.shape[0] != y.shape[0] or (weight_map.shape[1] != y.shape[1] and weight_map.shape[1] != 1) or weight_map.shape[2] != y.shape[2] or weight_map.shape[3] != y.shape[3]:
            raise ValueError(
                f"Expected weight_map to have the same B, H, and W as y | and must have eaither the same n_channels as y or have only 1 channel . Got weight_map: {weight_map.shape} and y: {y.shape}."
            )

        if weight_map.shape[1] == 1:
            weight_map = weight_map.expand(-1, y.shape[1], -1, -1)
        
        channel = y_pred.size(1) # as defined in pytorch docs: for Functional Conv2d kernel should be (out_channels, in_channels/groups, kH, kW)
        if len(self._kernel.shape) < 4:
            self._kernel = self._kernel.expand(channel, 1, -1, -1).to(device=y_pred.device) # out_channels is the same as in_channels | in_channels/groups = 1 sice we will set groups = channels

        
        y_pred = F.pad(y_pred, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        y = F.pad(y, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        
        weight_map = F.pad(weight_map, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        weight_map = F.conv2d(weight_map, self._kernel, groups=channel)

        input_tensor = torch.cat([y_pred, y, y_pred * y_pred, y * y, y_pred * y]) # SHAPE -> (B*5, C, H, W)
        outputs = F.conv2d(input_tensor, self._kernel , groups=channel) # running the kernel on each channel seperatly (groups devieds cheannls by chennel which is 1 | it has nothing to do with batches)

        output_list = [outputs[x * y_pred.size(0) : (x + 1) * y_pred.size(0)] for x in range(int(len(outputs)/y_pred.size(0)))] # len(outputs) is B*5 so we need to devidei t by B so its only 5 -> [y_pred, y, y_pred * y_pred, y * y, y_pred * y]
        
        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2) # SHAPE -> (B, C, H, W)
        # Multiplying SSIM map by the Weight map, to get the weighted sum
        ssim_idx_weighted = ssim_idx * weight_map 
        ssim_idx_weighted_sum = torch.sum(ssim_idx_weighted, (1, 2, 3), dtype=torch.float32)# sum ssim_idx over (C, H, W) -> out shape:(B)
        weight_map_sum = torch.sum(weight_map, (1, 2, 3), dtype=torch.float32) # sum of weighted_map over (C, H, W) -> out shape:(B)
        ssim_idx_weighted_mean = ssim_idx_weighted_sum / weight_map_sum # Weighted mean 

        
        _sum_of_ssim = ssim_idx_weighted_mean.sum().to(self._device) # sum of all batches

        _num_examples = y.shape[0] # number of batche
        return (_sum_of_ssim / _num_examples).item()

        
