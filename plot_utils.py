import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
import cv2 as cv
from PIL import Image
import os

"""Functions Needed in train_utils.py"""
    



def normalize(x):
    return (x - x.min()) / (x.max() - x.min())
def convert2uint8(x):
    return (x * 255).astype(np.uint8)

def save_numpy_array(image_array, filename="image.jpg"):
    """Save a NumPy array with shape (H x W x C) as an image."""
    # if shape is (H,,W) convert it to (H,W,1)
    if image_array.ndim == 2:
        image_array = image_array[:,:,np.newaxis]
    # if its a binary image, convert it to 3 channels
    if image_array.shape[-1] == 1:
        image_array = np.concatenate([image_array]*3, axis = -1)    
    
    image_array = convert2uint8(normalize(image_array))
        
    #print(f"Saving image of shape {image_array.shape} to {filename}")
    # Convert the NumPy array to a PIL Image object
    image = Image.fromarray(image_array)
    # Save the image as a JPEG file
    image.save(filename)
    
    
def save_s1s2_tensors_plot(tensors, names, n_rows, n_cols, filename,
                           fig_size, bands_to_plot = [2,1,0], title = None, just_show = False,img_indx=None, save_raw_images_folder = None):
    """
    Saves a grid of PyTorch tensors as an image file.

    Parameters
    ---
        tensors (List[torch.Tensor]): List of PyTorch tensors to be plotted, if number of channels of a tensor is more than 3, only the bands specified in `bands_to_plot` will be ploted as rgb 
        names (Optional[List[str]]): List of names for each tensor. If None,
                                      no names will be displayed.
        n_rows (int): Number of rows in the output grid.
        n_cols (int): Number of columns in the output grid.
        filename (str): Name of the output image file.
        fig_size (Tuple[int, int]): Size of the output image in inches.
        change_map_name (str): Name of the tensor containing the change map.
                               Default is 'change map'.
        bands_to_plot (list): the index of 3 bands to be ploted in case of tensor having more that 3 bands.                        
    
    Chnage Map
    ---
    converts the change map into 3 bands where the Red band is the change in RGB values, Green band is 
    the change in NIR values and Blue band is the change in SWIR valuess.

    Returns
    ---
        None
    """
    # checking ig save_raw_images_folder exists else create it
    if save_raw_images_folder:
        if not os.path.exists(save_raw_images_folder):
            os.makedirs(save_raw_images_folder)
    
    tensors = [tensor.to(torch.float32) for tensor in tensors]
    fig, axs = plt.subplots(n_rows, n_cols, figsize=fig_size)
    fig.suptitle(title) if title is not None else None
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
                if "change" in name:
                    tensor = combine_cm_bands(tensor)
                    array = tensor[[0,1,2],:,:].permute(1,2,0).cpu().numpy()
                else:
                    array = tensor[bands_to_plot,:,:].permute(1,2,0).cpu().numpy()
                    array = stretch_img(array) # we don't want to stretch the change map, since it will be misleading.
     
                axs[i][j].imshow(array)
                axs[i][j].set_title(name)
            else:
                array = tensor[0].cpu().numpy()
                axs[i][j].imshow(array) if "change" in name else axs[i][j].imshow(array,cmap='gray')
                axs[i][j].set_title(name)
            if save_raw_images_folder:
                if array.ndim == 2:
                    array = array[:,:,np.newaxis]
                if array.ndim > 2 and array.shape[2] ==1:
                    array = convert2uint8(normalize(array))
                    array = cv.applyColorMap(array, cv.COLORMAP_VIRIDIS) if "change" in name else array
                    array = cv.cvtColor(array, cv.COLOR_BGR2RGB)
                if img_indx:
                    save_numpy_array(array, filename=f"{save_raw_images_folder}/img{img_indx}_{name}.jpg")
                else:
                    save_numpy_array(array, filename=f"{save_raw_images_folder}/{name}.jpg")
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
    plt.tight_layout()
    if just_show:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
        matplotlib.pyplot.close()
      
import torch

def combine_cm_bands(input_tensor):
    """
    Takes a PyTorch tensor of shape [6, 256, 256] and returns a tensor of shape
    [3, 256, 256] where the first band is the max of bands 0, 1, and 2, the
    second band is band 3, and the third band is the max of bands 4 and 5.
    
    Usage
    ---
        The Change map is RGB NIR SWIR1 SWIR2 after this combination
        the first band is the max of RGB, the second band is NIR and the third band is the max of SWIR1 and SWIR2
        * Red is the change in RGB values
        * seond band (green) corresponds to vegetaiton change
        * Blue is the cahnge in SWIR values
        

    Parameters
    ---
        input_tensor (torch.Tensor): Input tensor of shape [6, 256, 256]

    Returns
    ---
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

    Parameters
    -----------
    img : numpy.ndarray
        A 3-dimensional numpy array representing the input BGR image. - Band Blue should be index 0 and band Red should be index 2

    clipLimit : int, optional (default=20)
        The threshold value for contrast limiting.

    tileGridSize : tuple, optional (default=(16,16))
        The size of the grid used to divide the image into small tiles for local histogram equalization.

    Returns
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


def plot_np_images(images, names, plot_name,folder, subplot_shape, fig_size= (10,10), subplot_spacing=(0.2, 0.2), save_path=None, img_format=".jpg"):
    """
    Plots a list of numpy images with shape (h,w,3) or (h,w,1) and a list of names in a subplot.

    Parameters:
    images (list): a list of numpy images with shape (h,w,3) or (h,w,1)
    names (list): a list of names for each image in the images list
    plot_name (str): a name for the overall plot
    subplot_shape (tuple): a tuple specifying the shape of the subplot (e.g. (2,3) for a 2x3 grid)
    subplot_spacing (tuple): a tuple specifying the horizontal and vertical spacing between subplots
    save_path (str): a file path to save the plot. If None, the plot will be displayed using plt.show()
    img_format (str): the format to save the images in (e.g. ".jpg", ".png", etc.)
    """
    # Create a figure object and subplots
    fig, axs = plt.subplots(subplot_shape[0], subplot_shape[1], figsize=fig_size)

    # Adjust the spacing between subplots
    fig.subplots_adjust(wspace=subplot_spacing[0], hspace=subplot_spacing[1])

    # Loop through each image and its corresponding name
    for name, image, ax in zip(names, images, axs.flatten()):
        ax.imshow(image)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        save_numpy_array(image, f"{folder}/{name}{img_format}")

    # Add a title to the plot
    fig.suptitle(plot_name)

    # Save the plot if a file path is provided
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show()








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