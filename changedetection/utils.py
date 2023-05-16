from skimage.filters import threshold_otsu
import skimage
import torch
import numpy as np


def get_ones_ratio(arr):
    """
    Calculates the ratio of ones in a 2D array.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    if len(arr.shape) == 3:
        arr = arr.squeeze(0)
    ones_count = np.count_nonzero(arr == 1)
    total_pixels = arr.shape[0] * arr.shape[1]
    ratio = ones_count / total_pixels
    return ratio

def get_binary_change_map(diff: torch.Tensor, threshold: float = 0.09) -> torch.Tensor:
    """
    Returns binary thresholded change map.

    Parameters
    ----------
    diff (torch.Tensor):  The input tensor.
    threshold (float, optional):  The threshold value. If None, threshold_otsu is used to calculate the threshold value.

    Returns
    -------
    torch.Tensor
        The binary thresholded change map tensor.
    """
    diff = diff.numpy()
    if threshold is None:
        threshold = threshold_otsu(diff)
    diff = (diff > threshold).astype(int)
    diff =diff[0,:,:]
    diff = skimage.morphology.binary_closing(diff)
    diff = skimage.morphology.remove_small_objects(diff, min_size=40)
    # numpy to torch tensor
    diff = torch.from_numpy(diff)
    diff = diff.unsqueeze(0)
    return diff # binary thresholded change map