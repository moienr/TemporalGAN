import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2 as cv

def plot_s1s2_tensors(tensors, names, n_rows, n_cols):
    tensors = [tensor.to(torch.float32) for tensor in tensors]
    fig, axs = plt.subplots(n_rows, n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx >= len(tensors):
                break
            tensor = tensors[idx].to(torch.float32)
            if torch.min(tensor) < 0:
                tensor = (tensor + 1)/2
            name = names[idx] if names is not None else None
            if tensor.ndim > 2 and tensor.shape[0] > 1:
                axs[i][j].imshow(tensor[[3,2,1],:,:].permute(1,2,0).cpu().numpy())
                axs[i][j].set_title(name)
            else:
                axs[i][j].imshow(tensor[0].cpu().numpy())
                axs[i][j].set_title(name)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
    plt.show()
    
    
def save_s1s2_tensors_plot(tensors, names, n_rows, n_cols, filename, fig_size,change_map_name = 'change map'):
    tensors = [tensor.to(torch.float32) for tensor in tensors]
    fig, axs = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx >= len(tensors):
                break
            tensor = tensors[idx].to(torch.float32)
            if torch.min(tensor) < 0:
                tensor = (tensor + 1)/2
            name = names[idx] if names is not None else None
            if tensor.ndim > 2 and tensor.shape[0] > 1:
                if name == change_map_name:
                    tensor = combine_cm_bands(tensor)
                    array = tensor[[0,1,2],:,:].permute(1,2,0).cpu().numpy()
                else:
                    array = tensor[[2,1,0],:,:].permute(1,2,0).cpu().numpy()
                
                array = stretch_img(array)
                axs[i][j].imshow(array)
                axs[i][j].set_title(name)
            else:
                axs[i][j].imshow(tensor[0].cpu().numpy())
                axs[i][j].set_title(name)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
    plt.savefig(filename)
      
import torch

def combine_cm_bands(input_tensor):
    """
    Takes a PyTorch tensor of shape [6, 256, 256] and returns a tensor of shape
    [3, 256, 256] where the first band is the max of bands 0, 1, and 2, the
    second band is band 3, and the third band is the max of bands 4 and 5.
    
    Usage:
        The Change map is RGB NIR SWIR1 SWIR2 after this combination
        the first band is the max of RGB, the second band is NIR and the third band is the max of SWIR1 and SWIR2
        * Red is the change in RGB values
        * seond band (green) corresponds to vegetaiton change
        * Blue is the cahnge in SWIR values
        

    Parameters:
        input_tensor (torch.Tensor): Input tensor of shape [6, 256, 256]

    Returns:
        torch.Tensor: Output tensor of shape [3, 256, 256]
    """
    max_band_0_1_2 = torch.max(input_tensor[:3], dim=0, keepdim=True)[0]
    band_3 = input_tensor[3:4]
    max_band_4_5 = torch.max(input_tensor[4:], dim=0, keepdim=True)[0]
    output_tensor = torch.cat([max_band_0_1_2, band_3, max_band_4_5], dim=0)
    return output_tensor
  
      
      

def stretch_img(img, clipLimit = 0.1 ,  tileGridSize=(32,32) ):
    """
    Enhance the contrast of an RGB image using Contrast Limited Adaptive Histogram Equalization (CLAHE) 
    and convert it to a stretched RGB image using the HSV color space.

    Parameters:
    -----------
    img : numpy.ndarray
        A 3-dimensional numpy array representing the input BGR image. - Band Blue should be index 0 and band Red should be index 2

    clipLimit : int, optional (default=20)
        The threshold value for contrast limiting.

    tileGridSize : tuple, optional (default=(16,16))
        The size of the grid used to divide the image into small tiles for local histogram equalization.

    Returns:
    --------
    numpy.ndarray
        A 3-dimensional numpy array representing the stretched RGB image with enhanced contrast.
    """
    img = img[:,:,[2,1,0]] # convert to bgr
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]


    
    clahe = cv.createCLAHE(clipLimit, tileGridSize)
    v = clahe.apply(v) #stretched histogram for showing the image with better contrast - its not ok to use it for scientific calculations

    hsv_img = np.dstack((h,s,v))

    # NOTE: HSV2RGB returns BGR instead of RGB
    bgr_stretched = cv.cvtColor(hsv_img, cv.COLOR_HSV2RGB)


    # if the valuse are float, plt will have problem showing them
    bgr_stretched = bgr_stretched.astype('uint8')

    return bgr_stretched





if __name__ == "__main__":
    from skimage import io 
    img = io.imread("E:\\s1s2\\s1s2_patched_light\\s1s2_patched_extra_light\\2021\\s2_imgs\\test\\014_brasilia_r00_c01.tif")
    print(img.shape)
    img = img[[2,1,0],:,:]
    img = img.swapaxes(0,2)
    img = img.swapaxes(0,1)
    img = stretch_img(img, clipLimit = 0.1 ,  tileGridSize=(32,23))
    plt.imshow(img)
    plt.show()