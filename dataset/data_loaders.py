import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from skimage import io
import os
from torchvision import datasets, transforms
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms

class Sen12Dataset(Dataset):
    """Dataset class for the Sen12MS dataset."""
    def __init__(self,
               s1_t2_dir,
               s2_t2_dir,
               s1_t1_dir,
               s2_t1_dir,
               s2_bands: list = None ,
               transform = None,
               hist_match=False,
               two_way = True,
               verbose=False):
        """
        Args:
            s1_t2_dir (str): Path to the directory containing the S1 time-2 images.
            s2_t2_dir (str): Path to the directory containing the S2 time-2 images.
            s1_t1_dir (str): Path to the directory containing the S1 time-1 images.
            s2_t1_dir (str): Path to the directory containing the S2 time-1 images.
            s2_bands (list, optional): List of indices indicating which bands to use from the S2 images.
                                       If not specified, all bands are used.
            transform (callable, optional): Optional transform to be applied to the S2 images.
            hist_match (bool, optional): Whether to perform histogram matching between the S2 time-2 
                                         and S2 time-1 images.
        """
        self.verbose = verbose
        # Set the directories for the four sets of images
        self.s1_t2_dir = s1_t2_dir
        self.s2_t2_dir = s2_t2_dir
        self.s1_t1_dir = s1_t1_dir
        self.s2_t1_dir = s2_t1_dir
        
        # Get the names of the S2 and S1 time-2 images and sort them
        self.s2_t2_names= os.listdir(s2_t2_dir)
        self.s1_t2_names= os.listdir(s1_t2_dir)
        self.s2_t2_names.sort()
        self.s1_t2_names.sort()
        # Get the names of the S2 and S1 time-1 images and sort them
        self.s2_t1_names= os.listdir(s2_t1_dir)
        self.s1_t1_names= os.listdir(s1_t1_dir)
        self.s2_t1_names.sort()
        self.s1_t1_names.sort()
        # Verify that the four sets of images have the same names
        if self.s1_t2_names != self.s2_t2_names or self.s1_t2_names != self.s2_t1_names or self.s1_t2_names != self.s2_t1_names:
            raise ValueError("The four directories do not contain the same image pairs.")
        
        self.s2_bands = s2_bands if s2_bands else None 

        self.transform = transform
        self.hist_match = hist_match
        
        self.two_way = two_way
        self.used_reversed_way = False

    def __len__(self):
        """Return the number of images in the dataset."""
        return 2 * len(self.s2_t2_names) if self.two_way else len(self.s2_t2_names)
  
    def __getitem__(self, index):
        """Get the S2 time-2 image, S1 time-2 image, S2 time-1 image, S1 time-1 image, 
           difference map and reversed difference map for the specified index.
           
        Args:
            index (int): Index of the image to get.
            
        Returns:
            tuple: A tuple containing the S2 time-2 image, S1 time-2 image, S2 time-1 image, S1 time-1 image, Difference map and Reversed difference map.
            * Difference map: `np.abs(s2_t2_img - s2_t1_img)`
            * Reversed difference map: `np.max(diff_map) - diff_map + np.min(diff_map) `
        """
        if self.two_way:
            if index < len(self.s2_t2_names):
                img_name = self.s2_t2_names[index] 
                self.used_reversed_way = False
            else:
                img_name = self.s2_t2_names[index - len(self.s2_t2_names)]
                self.used_reversed_way = True
        else:
            img_name = self.s2_t2_names[index] 
            self.used_reversed_way = False # just to be sure

        s2_t2_img_path = os.path.join(self.s2_t2_dir,img_name)
        s1_t2_img_path = os.path.join(self.s1_t2_dir,img_name)
        
        s2_t2_img = io.imread(s2_t2_img_path)
        if self.s2_bands: s2_t2_img = s2_t2_img[self.s2_bands,:,:]
        s1_t2_img = io.imread(s1_t2_img_path)
        
        s2_t1_img_path = os.path.join(self.s2_t1_dir,img_name)
        s1_t1_img_path = os.path.join(self.s1_t1_dir,img_name)
        
        s2_t1_img = io.imread(s2_t1_img_path)
        print(f's2 shape apon reading: {s2_t1_img.shape}')
        if self.s2_bands: s2_t1_img = s2_t1_img[self.s2_bands,:,:]
        s1_t1_img = io.imread(s1_t1_img_path)
    

        if self.hist_match:
            if self.used_reversed_way:
                s2_t1_img = match_histograms(s2_t1_img, s2_t2_img, channel_axis=0) # match the histograms of the two images (image, reference)
            else:
                s2_t2_img = match_histograms(s2_t2_img, s2_t1_img, channel_axis=0) # match the histograms of the two images (image, reference)

        if self.transform:
            sample = s2_t2_img, s1_t2_img
            s2_t2_img, s1_t2_img  = self.transform(sample)
            sample = s2_t1_img, s1_t1_img
            s2_t1_img, s1_t1_img  = self.transform(sample)

        if self.verbose:
            print(f"s2_t2_img shape: {s2_t2_img.shape}")
            print(f"s1_t2_img shape: {s1_t2_img.shape}")
            print(f"s2_t1_img shape: {s2_t1_img.shape}")
            print(f"s1_t1_img shape: {s1_t1_img.shape}")
        
        diff_map = np.abs(s2_t2_img - s2_t1_img) # to focus on the changes in the s2 image
        reversed_diff_map = np.max(diff_map) - diff_map + np.min(diff_map) # to focus the unchanged areas in the s2 image
        
        
        if self.used_reversed_way:
            return  s2_t1_img, s1_t1_img, s2_t2_img, s1_t2_img, diff_map, reversed_diff_map
        else:
            return s2_t2_img, s1_t2_img, s2_t1_img, s1_t1_img, diff_map, reversed_diff_map



class myToTensor:
    """Transform a pair of numpy arrays to PyTorch tensors"""
    def __init__(self,dtype=torch.float16):
        """Transform a pair of numpy arrays to PyTorch tensors
            Args:
                dtype (torch.dtype): Data type for the output tensor (default: torch.float16)
        """
        self.dtype = dtype
        
    def reshape_tensor(self,tensor):
        """Reshape a 2D or 3D tensor to the expected shape of pytorch models which is (channels, height, width)
        
        Args:
            tensor (numpy.ndarray): Input tensor to be reshaped
        
        Returns:
            torch.Tensor: Reshaped tensor
        """
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3 and tensor.shape[2] < tensor.shape[0]:
            tensor = tensor.permute((2,0,1))
        elif tensor.dim() == 3 and tensor.shape[2] > tensor.shape[0]:
            pass
        else:
            raise ValueError(f"Input tensor shape is unvalid: {tensor.shape}")
        return tensor

    def __call__(self,sample):
        img1, img2 = sample
        return self.reshape_tensor(torch.from_numpy(img1)).to(dtype=self.dtype)  , self.reshape_tensor(torch.from_numpy(img2)).to(dtype=self.dtype)  

class S2S1Normalize:
    """
    Class for normalizing Sentinel-2 and Sentinel-1 images for use with a pix2pix model.
    """

    def __init__(self, s1_min = -25, s1_max = 10 , s2_min = 0 , s2_max = 1):
        """
        Args:
            s1_min (float): Minimum value for Sentinel-1 data. Default is -25.
            s1_max (float): Maximum value for Sentinel-1 data. Default is 10.
            s2_min (float): Minimum value for Sentinel-2 data. Default is 0.
            s2_max (float): Maximum value for Sentinel-2 data. Default is 1.
        """
        self.s1_min = s1_min
        self.s1_max = s1_max
        self.s2_min = s2_min
        self.s2_max = s2_max
    def __call__(self,sample):
        """
        Normalize Sentinel-2 and Sentinel-1 images for use with a pix2pix model.

        Args:
            sample (tuple): Tuple containing Sentinel-2 and Sentinel-1 images as numpy arrays.

        Returns:
            tuple: Tuple containing normalized Sentinel-2 and Sentinel-1 images.
        """
        s2_img, s1_img = sample
        # Sentinel 2 image  is between 0 and 1 it is surface reflectance so it can't be more than 1 or less than 0
        s2_img[s2_img>self.s2_max] = self.s2_max
        s2_img[s2_img<self.s2_min] = self.s2_min
        # Normalizing the Senitnel 2 data between -1 and 1 so it could be used on pix2pix model
        s2_img = (s2_img * 2) - 1
        
        # Sentinel 1 VV image  is between -25 and 10 dB (we insured that in the data preparation step)
        # print(np.min(target),np.max(target))
        s1_img[s1_img>self.s1_max] = self.s1_max
        s1_img[s1_img<self.s1_min] = self.s1_min

        # Normalizing the Senitnel 1 data between -1 and 1  so it could be used on pix2pix model
        s1_img = (s1_img - np.min(s1_img)) / (np.max(s1_img) - np.min(s1_img))
        s1_img = (s1_img * 2) - 1
        
        return s2_img, s1_img
    
    

if __name__ == "__main__":
    s1s2_dataset = Sen12Dataset(s1_t1_dir="")