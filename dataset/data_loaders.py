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

  def __init__(self,
               s1_t2_dir,
               s2_t2_dir,
               s1_t1_dir,
               s2_t1_dir,
               s2_bands: list = None ,
               transform = None,
               hist_match=True):
    
    self.s1_t2_dir = s1_t2_dir
    self.s2_t2_dir = s2_t2_dir
    
    self.s2_t2_names= os.listdir(s2_t2_dir)
    self.s1_t2_names= os.listdir(s1_t2_dir)
    self.s2_t2_names.sort()
    self.s1_t2_names.sort()
    
    self.s2_t1_names= os.listdir(s2_t1_dir)
    self.s1_t1_names= os.listdir(s1_t1_dir)
    self.s2_t1_names.sort()
    self.s1_t1_names.sort()
    
    if self.s1_t2_names != self.s2_t2_names or self.s1_t2_names != self.s2_t1_names or self.s1_t2_names != self.s2_t1_names:
        raise ValueError("The four directories do not contain the same image pairs.")

    self.s2_bands = s2_bands if s2_bands else None 

    self.transform = transform
    self.hist_match = hist_match
    
    self.s1_t1_dir = s1_t1_dir
    self.s2_t1_dir = s2_t1_dir

  def __len__(self):
        return len(self.s2_t2_names)
  
  def __getitem__(self, index):
        img_name = self.s2_t2_names[index] 

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
            s2_t2_img = match_histograms(s2_t2_img, s2_t1_img, channel_axis=0) # match the histograms of the two images (image, reference)

        if self.transform:
            sample = s2_t2_img, s1_t2_img
            s2_t2_img, s1_t2_img  = self.transform(sample)
            sample = s2_t1_img, s1_t1_img
            s2_t1_img, s1_t1_img  = self.transform(sample)

        
        print(f"s2_t2_img shape: {s2_t2_img.shape}")
        print(f"s1_t2_img shape: {s1_t2_img.shape}")
        
        print(f"s2_t1_img shape: {s2_t1_img.shape}")
        print(f"s1_t1_img shape: {s1_t1_img.shape}")
        
        diff_map = np.abs(s2_t2_img - s2_t1_img) # to focus on the changes in the s2 image
        reversed_diff_map = np.max(diff_map) - diff_map + np.min(diff_map) # to focus the unchanged areas in the s2 image
        
        
        return s2_t2_img, s1_t2_img, s2_t1_img, s1_t1_img, diff_map, reversed_diff_map



class myToTensor:
    def __init__(self,dtype=torch.float16):
        self.dtype = torch.float16
    def reshape_tensor(self,tensor):
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
        input, target = sample
        return self.reshape_tensor(torch.from_numpy(input)).to(dtype=self.dtype)  , self.reshape_tensor(torch.from_numpy(target)).to(dtype=self.dtype)  

class myNormalize:
    def __init__(self, s1_min = -25, s1_max = 10 , s2_min = 0 , s2_max = 1):
        self.s1_min = s1_min
        self.s1_max = s1_max
        self.s2_min = s2_min
        self.s2_max = s2_max
    def __call__(self,sample):
        input, target = sample
        # input image is the Sentinel 2 image which is between 0 and 1
        input[input>1] = 1
        input[input<0] = 0
        input = input * 2
        input = input - 1
        # Target is Sentinel 1 VV image which is between -25 and 10
        # print(np.min(target),np.max(target))
        target[target>self.s1_max] = self.s1_max
        target[target<self.s1_min] = self.s1_min

        # Normalizing the Senitnel 1 data between -1 and 1 
        target += np.abs(self.s1_min)
        target = target/(np.abs(self.s1_max) + np.abs(self.s1_min))
        target = target * 2
        target = target - 1


        return input, target