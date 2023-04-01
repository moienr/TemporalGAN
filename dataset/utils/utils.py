""" Utilities for the dataset module, that don't use GEE
"""
import numpy as np

def correct_image_shape(image):
    """
    Transposes an image with size (C, H, W) to an image with size (H, W, C).

    Args:
        image (numpy array): An input image with size (C, H, W).

    Returns:
        numpy array: The transposed image with size (H, W, C).
    """
    # Swap the axes of the input image
    transposed_image = np.swapaxes(image, 0, 2)
    transposed_image = np.swapaxes(transposed_image, 0, 1)
    
    return transposed_image

import pandas as pd
def read_csv(csv_path):
    # read csv file into a pandas dataframe
    df = pd.read_csv(csv_path)
    # first column is the index, so we drop it
    df = df.iloc[:, 1:4]
    df.columns = ['Piont_id', 'long', 'lat']
    return df



from datetime import datetime
def milsec2date(millsec_list: list, no_duplicate = False)->list:
  '''
  Input
  ---
  this function takes `imgcollection.aggregate_array('system:time_start')` which is a list of milliseconds dates as input

  Reutrns
  ---
  * Defult: a list of dates in GEE date string format
  * No_duplicate: returns the list of dates but removes the duplicates
    '''
  if no_duplicate:
    date = [datetime.fromtimestamp(t/1000.0).strftime('%Y-%m-%d') for t in millsec_list]
    date_no_duplicate = list(dict.fromkeys(date))
    return  date_no_duplicate
  else:
    date = [datetime.fromtimestamp(t/1000.0).strftime('%Y-%m-%d') for t in millsec_list] 
    return date



def test_function(function,shape=False, *args, **kwargs):
    try:
        output = function(*args, **kwargs)
        print('Test passed!')
        if shape:
            print(output.shape)
        else:
            print(output)
    except Exception as e:
        print('Test failed!')
        print(e)
        

def mean_date(dates: list) -> str:
    '''the input is list in millisecound format and returns a date in gee sting format '''
    mil_mean = round(sum(dates) / len(dates))
    dates_mean = milsec2date([mil_mean])
    return dates_mean[0]

def date_diffrence(date1:str, date2:str) -> int:
    ''' date1 and date2 should be in gee string format `"YYYY-MM-DD"`
        reutns the number of days between the two dates in int format
    '''
    date1 = datetime.strptime(date1, '%Y-%m-%d')
    date2 = datetime.strptime(date2, '%Y-%m-%d')
    diff = date2 - date1
    return diff.days


class TextColors:
    """
    A class containing ANSI escape codes for printing colored text to the terminal.
    
    Usage:
    ------
    ```
    print(TextColors.HEADER + 'This is a header' + TextColors.ENDC)
    print(TextColors.OKBLUE + 'This is OK' + TextColors.ENDC)
    ```
    
    Attributes:
    -----------
    `HEADER` : str
        The ANSI escape code for a bold magenta font color.
    `OKBLUE` : str
        The ANSI escape code for a bold blue font color.
    `OKCYAN` : str
        The ANSI escape code for a bold cyan font color.
    `OKGREEN` : str
        The ANSI escape code for a bold green font color.
    `WARNING` : str
        The ANSI escape code for a bold yellow font color.
    `FAIL` : str
        The ANSI escape code for a bold red font color.
    `ENDC` : str
        The ANSI escape code for resetting the font color to the default.
    `BOLD` : str
        The ANSI escape code for enabling bold font style.
    `UNDERLINE` : str
        The ANSI escape code for enabling underlined font style.
        
    Subclasses:
    `BOLDs`
    `UNDERLINEs`
    `BACKGROUNDs`
    `HIGHLIGHTs`
    `HIGH_INTENSITYs`
    `BOLD_HIGH_INTENSITYs`
    `HIGH_INTENSITY_BACKGROUNDs`
    `BOLD_BACKGROUNDs`
    
    
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    SLIME = '\033[38;2;165;165;0m'
    
    
    class BOLDs:
        BLACK = '\033[1;30m'
        RED = '\033[1;31m'
        GREEN = '\033[1;32m'
        YELLOW = '\033[1;33m'
        BLUE = '\033[1;34m'
        PURPLE = '\033[1;35m'
        CYAN = '\033[1;36m'
        WHITE = '\033[1;37m'
        ORANGE ='\033[38;2;255;165;0m'


    class UNDERLINEs:
        BLACK = '\033[4;30m'
        RED = '\033[4;31m'
        GREEN = '\033[4;32m'
        YELLOW = '\033[4;33m'
        BLUE = '\033[4;34m'
        PURPLE = '\033[4;35m'
        CYAN = '\033[4;36m'
        WHITE = '\033[4;37m'
    
    class BACKGROUNDs:
        BLACK = '\033[40m'
        RED = '\033[41m'
        GREEN = '\033[42m'
        YELLOW = '\033[43m'
        BLUE = '\033[44m'
        PURPLE = '\033[45m'
        CYAN = '\033[46m'
        WHITE = '\033[47m'
        DEFAULT = '\033[49m'
    
    class HIGH_INTENSITYs:
        BLACK = '\033[0;90m'
        RED = '\033[0;91m'
        GREEN = '\033[0;92m'
        YELLOW = '\033[0;93m'
        BLUE = '\033[0;94m'
        PURPLE = '\033[0;95m'
        CYAN = '\033[0;96m'
        WHITE = '\033[0;97m'
    
    class BOLD_HIGH_INTENSITYs:
        BLACK = '\033[1;90m'
        RED = '\033[1;91m'
        GREEN = '\033[1;92m'
        YELLOW = '\033[1;93m'
        BLUE = '\033[1;94m'
        PURPLE = '\033[1;95m'
        CYAN = '\033[1;96m'
        WHITE = '\033[1;97m'
        
    class HIGH_INTENSITY_BACKGROUNDs:
        BLACK = '\033[0;100m'
        RED = '\033[0;101m'
        GREEN = '\033[0;102m'
        YELLOW = '\033[0;103m'
        BLUE = '\033[0;104m'
        PURPLE = '\033[0;105m'
        CYAN = '\033[0;106m'
        WHITE = '\033[0;107m'

    class BOLD_BAKGROUNDs:
        BLACK = '\033[1;40m'
        RED = '\033[1;41m'
        GREEN = '\033[1;42m'
        YELLOW = '\033[1;43m'
        BLUE = '\033[1;44m'
        PURPLE = '\033[1;45m'
        CYAN = '\033[1;46m'
        WHITE = '\033[1;47m'
        ORANGE = '\033[48;2;255;165;0m\033[1m'
        S1 ='\033[48;2;100;50;50m'
        S2 = '\033[48;2;50;50;100m'
    
    class BLACK_TEXT_WIHT_BACKGROUNDs:
        BLACK = '\033[40m'
        RED = '\033[41m'
        GREEN = '\033[42m'
        YELLOW = '\033[43m'
        BLUE = '\033[44m'
        PURPLE = '\033[45m'
        CYAN = '\033[46m'
        WHITE = '\033[47m'
        
        
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

def month_add(date:str,months_to_add = 1) -> str:
    ''' date should be string in `'2020-02-01' format`
    
    Usage
        `day_add('2020-12-30',days_to_add = 2)`
    
    '''
    date_time_obj = datetime.strptime(date, '%Y-%m-%d')
    new_date_time_obj= date_time_obj + relativedelta(months=+months_to_add)
    new_date_str = new_date_time_obj.strftime('%Y-%m-%d')
    return new_date_str


def day_add(date:str,days_to_add = 2) -> str:
    ''' date should be string in `'2020-02-01' format`
    
    Usage:
        `month_add('2020-12-01',months_to_add = -1)`
    
    '''
    date_time_obj = datetime.strptime(date, '%Y-%m-%d')
    new_date_time_obj= date_time_obj + relativedelta(days=+days_to_add)
    new_date_str = new_date_time_obj.strftime('%Y-%m-%d')
    return new_date_str


def day_buffer(days_list: list,no_duplicate=True)->list:
    """
    Returns a list of dates that includes the input dates plus a buffer of
    dates up to 2 days before and 2 days after each input date.

    Args:
        days_list (list): A list of dates in the format "YYYY-MM-DD".
        no_duplicate (bool, optional): If True, removes any duplicate dates
            from the output list. Defaults to True.

    Returns:
        list: A list of dates in the format "YYYY-MM-DD".
        
    
    Usage:
    ```
    dummy_dates = ['2020-06-06','2020-06-02','2020-06-12']
    x =day_buffer(dummy_dates)
    x.sort()
    ---
    output: ['2020-06-04', '2020-06-05', '2020-06-06', '2020-06-07', '2020-06-08', '2020-05-31', '2020-06-01', '2020-06-02', '2020-06-03', '2020-06-10', '2020-06-11', '2020-06-12', '2020-06-13', '2020-06-14']
    
    ```
    """
    
    f0 = lambda x: day_add(x,days_to_add = 0)
    fp1 = lambda x: day_add(x,days_to_add = 1)
    fp2 = lambda x: day_add(x,days_to_add = 2)
    fm1 = lambda x: day_add(x,days_to_add = -1)
    fm2 = lambda x: day_add(x,days_to_add = -2)

    bufferd = [f(x) for x in days_list for f in (fm2,fm1,f0,fp1,fp2)]
    if no_duplicate:
        bufferd_no_duplicate = list(dict.fromkeys(bufferd))
        return bufferd_no_duplicate
    else:
        return bufferd
    
    

def list_intersection(in_list,ref_list):
    '''checks if the Items in the `in_list` are  in `ref_list`
    
    Returns
    ---
    index of items 
    
    Usage
    -----
    ```
    list_1 = ['2020-06-02','2020-06-07','2020-06-06','2020-06-07','2020-06-12'] # for example this is sen1 images
    list_2 = ['2020-06-07','2020-06-12'] # this could be snowy images
    list_intersection(list_1,list_2)
    ----
    ouput: [1, 3, 4]
    ```
    '''
    intersect_list = [i for i,x in enumerate(in_list) if x in ref_list] # https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
    return intersect_list

def dict_to_int(dic):
    pix_num = list(dic.items())[0][1] 
    return pix_num




import numpy as np
import math
def best_step_size(img_size, patch_size = 256, ov_first_range = 32, acceptable_r = 50, mute=False):
    """
    inputs
    ---
    * `img_size`: the length or width of the image
    * `patch_size`: size of the patch in the same directoin of chosen image size
    * `ov_first_range`: first finds the remainders of  overlaps less than this chosen range, if the remainder is less 
        than the `acceptable_r` we choose the corresponding `ov` as the optimum, if not the `extreme mode` gets activated
        which looks for best overlap in range of `ov_first_range` and `patch_size/2`
    * `acceptable_r`: the threshold of remainder, where the extreme mode activates.
    
    outputs
    ---
    * `step_size`: `patch_size - optimum overlap` how much the moving_window moves to capture the next patch. 
    * `number_of_patches`
    * `opt_ov` : optimum overlap which gives the least remainder value.
    """
    l = img_size
    best_r = 255 # the maximum remainder can be 255
    opt_ov = 0
    for ov in range(0,ov_first_range+1): # ov from 0 to 32
        r = (l-patch_size)%(patch_size-ov)
        if r<best_r:
            best_r = r
            opt_ov = ov

    if best_r > acceptable_r:
        print('extreme mode activated!')
        for ov in range(ov_first_range+1,int(patch_size/2)): # we accept maximum overlap of 128
            r = (l-patch_size)%(patch_size-ov) 
            if r<best_r:
                best_r = r
                opt_ov = ov
    number_of_patches = math.floor(((l-patch_size)/(patch_size-opt_ov))+1)
    if not mute:
        print('remainder        : ', best_r)
        print('optimum overlap  : ', opt_ov)
        print('optimum stepsize : ', patch_size-opt_ov)
        print('number of patches: ', number_of_patches)
    step_size = patch_size-opt_ov #step_size = patch_size - overlaps
    return step_size , number_of_patches , opt_ov

def perfect_patchify(img, patch_size=(256,256) , ov_first_range=32, acceptable_r=50,mute=True):
    """
    Inputs
    ---
    * `img`: a 3D numpy array of `(rows,columns,channels)`
    * `patch_size`: size of each patch `(row_size,column_size)`
    * `ov_first_range` , `acceptable_r` , `mute`: arguments of funcion `best_step_size` to caclutate optimum `step_size`
    
    Workflow
    ---
    note that image `width` corresponds to number of columns, and `hight` refers to the number of rows.
    
    Output:
    * `stacked_rows`: a 5D numpy array, containing patches
                        of the image `(number_patches_in_each_row, number_patches_in_each_column, patch_hight, patch_width, patch_channels)`
    """
    patch_width = patch_size[1]
    patch_hight = patch_size[0]
    clmn_sz, clmn_n_ptchs,clmn_opt_ovl = best_step_size(img.shape[1],patch_size=patch_width, ov_first_range=ov_first_range, acceptable_r=acceptable_r,mute=mute)
    row_sz, row_n_ptchs ,row_opt_ovl  = best_step_size(img.shape[0],patch_size=patch_hight, ov_first_range=ov_first_range, acceptable_r=acceptable_r ,mute=mute)
    stacked_rows = [] # each row consists of columns.
    for i in range(row_n_ptchs): # this loop chooses a row
        row_patches = [] # each row consists of column patches.
        rp_start= i * row_sz # row_patch_start: we start from 0 then 0+step_size and so on
        rp_end = rp_start + patch_hight # end of each patch is strt +patch size for example 0-256 adn 200-456
        for j in range(clmn_n_ptchs): # this loop captures the clumnwise patches from the loop i
            cp_start= j * clmn_sz # column patch starting pixel
            cp_end = cp_start + patch_width  # column patch last pixel
            patch = img[rp_start:rp_end,cp_start:cp_end,:] # a slice of image  based on the rows and columns dfeined by loop i and j
            row_patches.append(patch)  # A list contaiing patchs of row i
        #print('length: ', len(row_patches))
        row_patches = np.stack(row_patches)  # converting the list into a stacked numpy array (rows_stack,hight,wdith,channels)
        #print(row_patches.shape)
        stacked_rows.append(row_patches)

    #print('length: ', len(stacked_rows))
    stacked_rows = np.stack(stacked_rows)  # stacking all the rows into a numpy array which gives us the final image patches.
    if not mute: print('final stacked shape: ',stacked_rows.shape)    

    return stacked_rows 


def reshape_tensor(tensor):
    """Takes in a pytorch tensor and reshapes it to (C,H,W) if it is not already in that shape.
    
    This Algorithm won't work if C is larger than H or W
    We assume that the smallest dimension is the channel dimension.
    """
    if tensor.dim() == 2: # If it is a 2D image we need to add a channel dimension
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3 and tensor.shape[2] < tensor.shape[0]: # if it is a 3D image and 3rd dim is smallest, it means it it the channel so we permute
        tensor = tensor.permute((2,0,1))
    elif tensor.dim() == 3 and tensor.shape[2] > tensor.shape[0]: # if it is a 3D image and and the first dimension is smaller than others it means its the C and we don't need to permute.
        pass
    else:
        raise ValueError(f"Input tensor shape is unvalid: {tensor.shape}")
    return tensor

def reshape_array(array: np.ndarray, channel_first=True) -> np.ndarray:
    """Takes in an array and reshapes it to (C,H,W) or (H,W,C) based on user input.
    
    Args
    ----
    `array`: a numpy array of shape (H,W,C) or (C,H,W) or (H,W)
    `channel_first`: if True, the output array will be of shape (C,H,W) otherwise it will be (H,W,C)
    
    This Algorithm won't work if C is larger than H or W
    We assume that the smallest dimension is the channel dimension.
    """
    if channel_first:
        if array.ndim == 2: # If it is a 2D image we need to add a channel dimension
            array = np.expand_dims(array, axis=0)
        elif array.ndim == 3 and array.shape[2] < array.shape[0]: # if it is a 3D image and 3rd dim is smallest, it means it it the channel so we permute
            array = np.swapaxes(array, 0, 2)
            array = np.swapaxes(array, 1, 2)
        elif array.ndim == 3 and array.shape[2] > array.shape[0]: # if it is a 3D image and and the first dimension is smaller than others it means its the C and we don't need to permute.
            pass
        else:
            raise ValueError(f"Input array shape is invalid: {array.shape}")
    
    else:
        if array.ndim == 2:
            array = np.expand_dims(array, axis=-1)
        elif array.ndim == 3 and array.shape[0] < array.shape[2]:
            array = np.swapaxes(array, 0, 2)
            array = np.swapaxes(array, 0, 1)
        elif array.ndim == 3 and array.shape[0] > array.shape[2]:
            pass
        else:
            raise ValueError(f"Input array shape is invalid: {array.shape}")
    
    return array




def nan_remover(image,nan_threshhold = 1, replace_with = 0.01):
    """
    Removes the nans from the image and replaces them with `replace_with` value
    
    Inputs
    ---
    `image`: a nd numpy array
    `nan_threshhold`: the precentaage of nans that is acceptable
    `replace_with`(defualt= 0.01): the value to replace the nans with 
    """
    nan_ratio = (np.count_nonzero(np.isnan(image))/image.size) * 100
    print(f'NaN Ratio: {nan_ratio} Percent')
    if nan_ratio > nan_threshhold:
            print(TextColors.WARNING,f'⚠️ High NaN ratio! ⚠️',TextColors.ENDC)

    image[np.isnan(image)] = replace_with
    return image



import os
import fnmatch
def count_files(folder, formart = '*.tif'):
    """finds the number of tif files in a folder

    Args:
        folder (str): the folder to search in
        formart (str, optional): the file foramt to look for . Defaults to '*.tif'.

    Returns:
        int: the number of files
    """
    count = 0
    for root, dirs, files in os.walk(folder):
        for filename in fnmatch.filter(files, formart):
            count += 1
    return count    



from skimage import io
def patch_folder(in_path, out_path, input_sat = 'S2', remove_year = True):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    files = os.listdir(in_path)
    files.sort()
    # only keep the tif files
    files = [f for f in files if f.endswith(".tif") or f.endswith(".tiff")]
    
    for file in files:
        print(TextColors.HIGH_INTENSITY_BACKGROUNDs.PURPLE, file ,TextColors.ENDC)
        x=io.imread(in_path+file)
        img = reshape_array(x,channel_first=False)
        print("shape after fix:", img.shape)

        print("range before norm: ",np.min(img),np.mean(img),np.std(img),np.max(img))

        if input_sat == 'S2':
            img[img>0.99] = 0.99
            img[img<0] = 0
        elif input_sat == 'S1':
            img[img>15] = 15
            img[img<-25] = -25
            
        img = nan_remover(img)

        print("range after norm and NaN removal: ",TextColors.HIGH_INTENSITYs.CYAN,np.min(img),np.mean(img),np.std(img),np.max(img),TextColors.ENDC)

        patches = perfect_patchify(img,mute=True)
        
        print('✅',TextColors.OKGREEN,'Final Shape->',patches.shape,TextColors.ENDC)

        # SAVING PATHCES
        img_name = file.split('.')[0]
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i,j,:,:,:]
                if input_sat == 'S2': # reshaping to (H,W,C) so io.imsave can save it as a tif
                    patch = np.swapaxes(patch, 2,0)
                    patch= np.swapaxes(patch, 1,2)
                #print(f'patch shape: {patch.shape}')
                if remove_year:
                    io.imsave(out_path + img_name[:-2] + '_r'+ str(i).zfill(2) + '_c' + str(j).zfill(2) + '.tif', patch) # the [:-2] removes the year from the names
                else:
                    io.imsave(out_path + img_name + '_r'+ str(i).zfill(2) + '_c' + str(j).zfill(2) + '.tif', patch)

if __name__ == "__main__":
    patch_folder(in_path = 'E:\s1s2\s1s2\content\drive\MyDrive\TemporalGAN-main\dataset\s1s2\\2021\s1_imgs\\test\\', out_path = 'E:\s1s2\s1s2\content\drive\MyDrive\TemporalGAN-main\dataset\s1s2_patched\\2021\s1_imgs\\test\\', input_sat = 'S1')