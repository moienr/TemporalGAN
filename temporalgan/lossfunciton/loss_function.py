import torch

import torch
import torch.nn as nn

class WeightedL1Loss(nn.Module):
    def __init__(self, change_weight = 5):
        """
        Args:
        - change_weight: A scalar value representing the weight of L1 loss for changed pixels.
                        the weight of L1 loss for unchanged pixels is 1.

        Returns:
        - None
        """
        super().__init__()
        self.change_weight = change_weight

    def forward(self, input, target, change_map, reversed_change_map):
        """
        Calculates the L1 loss between the input and target images using a change map.

        Args:
        - input: A PyTorch tensor of size (batch_size, channels, height, width) representing the input image.
        - target: A PyTorch tensor of size (batch_size, channels, height, width) representing the target image.
        - change_map: A PyTorch tensor of size (batch_size, 1, height, width) representing the change weight map.
        - reversed_change_map: A PyTorch tensor of size (batch_size, 1, height, width) representing the reversed change weight map.

        Returns:
        - loss: A PyTorch scalar representing the weighted L1 loss.
        """

        # Calculate the absolute difference between the input and target images
        abs_diff = torch.abs(input - target)
        
        # Calculate the mean of the change map along the channels dimension so the weights are a 2D tensor (batch, 1, height, width)
        change_map = torch.mean(change_map, dim=1, keepdim=True)
        reversed_change_map = torch.mean(reversed_change_map, dim=1, keepdim=True)
        
        # Multiply the absolute difference by the cahnge map
        change_weighted_diff = abs_diff * change_map
        # Sum the weighted differences along the height and width dimensions
        sum_change_weighted_diff = torch.sum(change_weighted_diff, dim=[2, 3])
        # Sum the weights along the height and width dimensions
        sum_weights = torch.sum(change_map, dim=[2, 3])
        # Divide the sum of the weighted differences by the sum of the weights
        changed_loss = torch.mean(sum_change_weighted_diff / sum_weights)
        
        
        # Multiply the absolute difference by the unhanged map
        unchange_weighted_diff = abs_diff * reversed_change_map
        # Sum the weighted differences along the height and width dimensions
        sum_unchange_weighted_diff = torch.sum(unchange_weighted_diff, dim=[2, 3])
        # Sum the weights along the height and width dimensions
        sum_weights = torch.sum(reversed_change_map, dim=[2, 3])
        # Divide the sum of the weighted differences by the sum of the weights
        unchanged_loss = torch.mean(sum_unchange_weighted_diff / sum_weights)
        
        # Calculate the final loss as a weighted sum of the changed and unchanged losses
        loss = unchanged_loss + self.change_weight * changed_loss

        return loss


if __name__ == "__main__":
    # Create a dummy input and target image
    input = torch.randn(1, 3, 256, 256)
    target = torch.randn(1, 3, 256, 256)

    # Create a dummy weight map
    change_map = torch.ones(1, 3, 256, 256)
    unchanged_map = (torch.max(change_map) - change_map) + torch.min(change_map)

    # Calculate the weighted L1 loss
    loss = WeightedL1Loss(change_weight=1)(input, target, change_map, unchanged_map)

    print(loss)