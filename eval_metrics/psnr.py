import torch
from typing import Sequence

def wpsnr(compare_tesnors: Sequence[torch.Tensor], weight_map: torch.Tensor = None, max_val: float = 1.) -> torch.Tensor:
    """Compute the Peak Weighted Signal-to-Noise Ratio (PSNR) between two batches of images based on a weight map.

    Args
    ---
        compare_tensors (Sequence[torch.Tensor]): A sequence of two tensors (y and y_pred) representing the batches of images to compare. 
            Both tensors must have shape (B, C, H, W).
        weight_map (torch.Tensor, optional): The weight map. Must have the same Batch size, H and W as `y` tesnor but channel could be either 1 or the same as `y` 
        max_val (float, optional): The maximum value of the pixels. Defaults to 1. Example: for `uint8` images, 
            it should be 255.

    Returns
    ---
        float: The mean PSNR value between each pair of images in the batches based on the weight map.

    Raises
    ---
        ValueError: If `compare_tensors` and `weight_map` have different shapes, or if they are not 4D tensors with
            the expected shapes.
    """
    y_pred, y = compare_tesnors[0].detach(), compare_tesnors[1].detach()
    weight_map = weight_map.detach()
    
    if y.shape != y_pred.shape:
        raise ValueError(f"Images and weight map must have the same shape. Got {y.shape}, {y_pred.shape}, and {weight_map.shape}")
    
    if len(weight_map.shape) != 4 or len(y.shape) != 4:
        raise ValueError(f"Expected weight_mapm y and y_pred to be 4D tensors. Got weight_map: {weight_map.shape} and y: {y.shape}, and y_pred: {y_pred.shape}.")

    if weight_map.shape[0] != y.shape[0] or (weight_map.shape[1] != y.shape[1] and weight_map.shape[1] != 1) or weight_map.shape[2] != y.shape[2] or weight_map.shape[3] != y.shape[3]:
        raise ValueError(
            f"Expected weight_map to have the same B, H, and W as y | and must have eaither the same n_channels as y or have only 1 channel . Got weight_map: {weight_map.shape} and y: {y.shape}."
        )
    if weight_map.shape[1] == 1:
        weight_map = weight_map.expand(-1, y.shape[1], -1, -1)
    
    mse = torch.sum(weight_map * (y - y_pred)**2, dim=[1, 2, 3]) / torch.sum(weight_map, dim=[1, 2, 3])
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return torch.mean(psnr).item()
