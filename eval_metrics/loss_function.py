import torch

import torch
import torch.nn as nn

def reverse_map(change_map):
    """
    Reverses the change map so that the changed pixels have a weight of Min and the unchanged pixels have a weight of Max.
    
    Args:
    - change_map: A PyTorch tensor of size (batch_size, c, height, width) representing the change weight map.
    
    Returns:
    - reversed_change_map: A PyTorch tensor of size (batch_size, c, height, width) representing the reversed change weight map.
    """
    # Find the maximum and minimum for each channel in the change map | although S1 is one channel but this can come in handy if we want to VH polarization in the future.
    max_values, _ = torch.max(change_map, dim=3, keepdim=True)
    max_values, _ = torch.max(max_values, dim=2, keepdim=True)
    min_values, _ = torch.max(change_map, dim=3, keepdim=True)
    min_values, _ = torch.max(min_values, dim=2, keepdim=True)
    reversed_change_map = max_values - change_map + min_values
    return reversed_change_map

class WeightedL1Loss(nn.Module):
    def __init__(self, change_weight = 5, convert_to_float32: bool = True, legacy_chage_map: bool = False):
        """
        Args
        ----
        change_weight: A scalar value representing the weight of L1 loss for changed pixels.
                        the weight of L1 loss for unchanged pixels is 1.
        convert_to_float32: A boolean value representing whether to convert the input and target images to float32.
                            This is useful for when the input and target images are float16 which can cause the loss to be NaN.
        legacy_chage_map: A boolean value representing whether to use the legacy change map or not.
                            The legacy change map is calculated as (1-change_map) instead of (max(change_map) - change_map + min(change_map))
            
                        

        Returns:
        - None
        """
        super().__init__()
        self.change_weight = change_weight
        self.convert_to_float32 = convert_to_float32
        self.legacy_chage_map = legacy_chage_map
    
        
    def forward(self, input, target, change_map):
        """
        Calculates the L1 loss between the input and target images using a change map.

        Args:
        - input: A PyTorch tensor of size (batch_size, channels, height, width) representing the input image.
        - target: A PyTorch tensor of size (batch_size, channels, height, width) representing the target image.
        - change_map: A PyTorch tensor of size (batch_size, 1, height, width) representing the change weight map.
        
        Attributes:
        - reversed_change_map: A PyTorch tensor of size (batch_size, C, height, width) representing the reversed change weight map.


        Returns:
        - loss: A PyTorch scalar representing the weighted L1 loss.
        """
        if self.legacy_chage_map:
            reversed_change_map = (1-change_map.clone())
        else:
            cm_copy = change_map.clone()
            # Find the maximum and minimum for each channel in the change map | although S1 is one channel but this can come in handy if we want to VH polarization in the future.
            reversed_change_map = reverse_map(cm_copy)
            
        if self.convert_to_float32:
            input = input.to(torch.float64)
            target = target.to(torch.float64)
            change_map = change_map.to(torch.float64)
            reversed_change_map = reversed_change_map.to(torch.float64)
            
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
        sum_weights.masked_fill_(sum_weights == 0, 0.0001)
        # Divide the sum of the weighted differences by the sum of the weights
        changed_loss = torch.mean(sum_change_weighted_diff / sum_weights)
        
        
        # Multiply the absolute difference by the unhanged map
        unchange_weighted_diff = abs_diff * reversed_change_map
        # Sum the weighted differences along the height and width dimensions
        sum_unchange_weighted_diff = torch.sum(unchange_weighted_diff, dim=[2, 3])
        # Sum the weights along the height and width dimensions
        sum_weights = None
        sum_weights = torch.sum(reversed_change_map, dim=[2, 3])
        sum_weights.masked_fill_(sum_weights == 0, 0.0001)
        # Divide the sum of the weighted differences by the sum of the weights
        unchanged_loss = torch.mean(sum_unchange_weighted_diff / sum_weights)
    
        # Calculate the final loss as a weighted sum of the changed and unchanged losses
        loss = (unchanged_loss + self.change_weight * changed_loss) / (1 + self.change_weight)
        
        if torch.isnan(loss) or torch.isnan(loss).any():
            raise ValueError(f"Loss is NaN \n \
                            changed_loss: {torch.mean(changed_loss)} | unchanged_loss: {torch.mean(unchanged_loss)} \n \
                            sum_unchange_weighted_diff: {torch.mean(sum_unchange_weighted_diff)} | sum_weights: {torch.mean(sum_weights)} \n \
                            abs_diff: {torch.mean(abs_diff)} | change_map: {torch.mean(change_map)} | reversed_change_map: {torch.mean(reversed_change_map)} \n \
                            input: {torch.mean(input)} | target: {torch.mean(target)}")
        
        return loss.to(torch.float64)


if __name__ == "__main__":
    # Create a dummy input and target image
    input = torch.rand((1, 3, 256, 256)).to(torch.float16)
    target = torch.rand((1, 3, 256, 256)).to(torch.float16)

    # Create a dummy weight map
    change_map = torch.rand((1, 3, 256, 256)).to(torch.float16)
    print("mean->", torch.min(change_map), change_map.shape, change_map.dtype)
    # Calculate the weighted L1 loss
    loss_dytpe16 = WeightedL1Loss(change_weight=1,convert_to_float32=False)(input, target, change_map)
    loss_dytpe32 = WeightedL1Loss(change_weight=1,convert_to_float32=True)(input, target, change_map)

    print("Loss with float16: ", loss_dytpe16)
    print("Loss with float32: ", loss_dytpe32)