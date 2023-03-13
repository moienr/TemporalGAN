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




if __name__ == '__main__':
    #test_function(get_square_roi, 40.02, -105.25, roi_size=1920)
    #test_function(correct_image_shape,True,  np.random.rand(3, 256, 256))
    df = read_csv('D:\\python\\SoilNet\\dataset\\utils\\test.csv')
    print(df)
    
    
    
