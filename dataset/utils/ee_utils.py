""" A collection of utility functions for working with Earth Engine (EE) in Python.

    Functions
    ---------
    ### `get_square_roi` :
        returns a square region of interest (ROI) centered at the given latitude and longitude coordinates with the specified size.
    ### `get_cloud_mask` : 
        Takes an ee.Image and returns the cloud, cloud shadow and  cloud_or_cloudShadow mask
    ### `get_snow_mask` : 
        Takes an ee.Image and returns the snow mask
    ### `get_mean_ndvi` :
        Takes an ee.Image and returns the mean NDVI value of the image
    ### `get_mask_ones_ratio` : 
        Takes a  01 mask as an ee.Image and returns the ratio of ones in the mask
    ### `get_not_nulls_ratio` : 
        Takes an ee.Image and returns the ratio of pixels that are not null in the image.
    ### `add_mineral_indices` : 
        Takes an ee.Image and adds the following mineral indices to it as it bands: clayIndex, ferrousIndex, carbonateIndex, rockOutcropIndex
    ### `get_closest_image` : 
        Takes an ee.ImageCollection and a date and returns the image in the collection that is closest to the given date.
    ### `radiometric_correction`: 
        Takes an ee.Image and returns the radiometrically corrected image. (only the Reflectance bands will change)
"""

import ee
import geemap
from utils.utils import *
# if __name__ != '__main__':
#     try:
#         ee.Initialize()
#     except Exception as e:
#         print("Failed to initialize Earth Engine: ", e)
#         print("Maybe try ee.Authenticate() and ee.Initialize() again?")


def get_square_roi(lat, lon, roi_size = 1920, return_gee_object = False):
    """
    Returns a square region of interest (ROI) centered at the given latitude and longitude
    coordinates with the specified size. By default, the ROI is returned as a list of
    coordinate pairs (longitude, latitude) that define the corners of the square. If
    `return_gee_object` is True, the ROI is returned as an Earth Engine geometry object.

    Args
    ----
        `lat` (float): Latitude coordinate of the center of the ROI.
        `lon` (float): Longitude coordinate of the center of the ROI.
        `roi_size` (int, optional): Size of the square ROI in meters. Default is 1920 meters. (about `64` pixels of `30m` resolution)
        `return_gee_object` (bool, optional): Whether to return the ROI as an Earth Engine geometry
            object instead of a list of coordinates. Default is False.

    Returns
    -------
        list or ee.Geometry.Polygon: If `return_gee_object` is False (default), a list of coordinate
            pairs (longitude, latitude) that define the corners of the square ROI. If `return_gee_object`
            is True, an Earth Engine geometry object representing the square ROI.

    Usage
    -----
        # Get a square ROI centered at lat=37.75, lon=-122.42 with a size of 1000 meters
        roi = get_square_roi(37.75, -122.42, roi_size=1000)
        print(roi)  # Output: [[-122.431, 37.758], [-122.408, 37.758], [-122.408, 37.741], [-122.431, 37.741], [-122.431, 37.758]]

    """

    # Convert the lat-long point to an EE geometry object
    point = ee.Geometry.Point(lon, lat)

    # Create a square buffer around the point with the given size
    roi = point.buffer(roi_size/2).bounds().getInfo()['coordinates']
    
    if return_gee_object:
        return ee.Geometry.Polygon(roi, None, False)
    else:
        # Return the square ROI as a list of coordinates
        return roi
    
    

# Modifed for Sentinel 2
def get_cloud_mask(img: ee.Image, pixel_quality_band='QA60',
                    cloud_bit = 10,
                    cirrus_bit = 11,):
    """Takes an ee.Image and returns the cloud, cloud shadow and  cloud_or_cloudShadow mask

    Args:
        `img` (ee.Image): An ee.Image object containing a pixel quality band. (e.g. 'QA60' of Sentine 2 )
        `pixel_quality_band` (str, optional): Name of the pixel quality band. Default is 'QA60'. (e.g. 'QA60' of Sentinel 2)
        `cloud_bit` (int, optional): Bit position of the cloud bit. Default is 3.
        `ciruss_bit` (int, optional): Bit position of the cloud shadow bit. Default is 4.

    Returns:
        tuple: A tuple containing the cloud mask, cloud shadow mask, and the combined mask. (ee.Image, ee.Image, ee.Image)
    """
    qa = img.select(pixel_quality_band)
    # Get the pixel values for the cloud, cloud shadow, and snow/ice bits
    cloud = qa.bitwiseAnd(1 << cloud_bit)
    cirrus = qa.bitwiseAnd(1 << cirrus_bit)
    return cloud, cirrus, cloud.Or(cirrus)


def get_cloud_mask_form_scl(image: ee.Image) -> ee.Image:
    """
    This function takes a Sentinel-2 Level 2A Earth Engine image as input and returns the mask for clouds and cloud shadows.

    Args:
        image: Sentinel-2 Level 2A Earth Engine image to be processed.

    Returns:
        A binary mask indicating the presence of clouds and cloud shadows in the input image. The mask is of the same dimensions as the input image, with a value of 1 indicating the presence of clouds or cloud shadows and 0 indicating their absence.

    Note:
        The Sentinel-2 Cloud Mask is generated from the Scene Classification Layer (SCL), which is included in the Level 2A product. The function uses the SCL band to identify the pixels classified as clouds or cloud shadows based on their SCL values. In particular, a pixel is classified as a cloud if its SCL value is 3, and as a cloud shadow if its SCL value is 9.
    """
    scl = image.select('SCL')
    mask = scl.eq(3).Or(scl.eq(9))
    return mask


# Snow/Ice mask
def get_snow_mask(img: ee.Image, pixel_quality_band='QA_PIXEL',
                    snow_bit = 5,
                    snow_confidence_bit = 12):
    """Takes an ee.Image and returns the Snow mask

    Args:
        `img` (ee.Image): An ee.Image object containing a pixel quality band. (e.g. 'QA_PIXEL' of Landsat8 SR)
        `pixel_quality_band` (str, optional): Name of the pixel quality band. Default is 'QA_PIXEL'. (e.g. 'QA_PIXEL' of Landsat8 SR)
        `snow_bit` (int, optional): Bit position of the snow bit. Default is 3.
        `snow_confidence_bit` (int, optional): Bit position of the snow confidence bit. Default is 8.

        
        * Refrence for Defualt Values: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#bands

    Returns:
        tuple: A tuple containing the snow mask, snow shadow mask, and the combined mask. (ee.Image, ee.Image, ee.Image)
    """
    qa = img.select(pixel_quality_band)
    snow = qa.bitwiseAnd(1 << snow_bit).And(qa.bitwiseAnd(3<<snow_confidence_bit))
    return snow


from typing import List
def get_mean_ndvi(image, bands: List[str] = ['SR_B5', 'SR_B4']):
    """
    Returns the mean NDVI of the given image.
    """
    # Compute NDVI
    ndvi = image.normalizedDifference(bands)
    
    # Compute mean of NDVI
    mean_ndvi = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=image.geometry(),
        scale=image.projection().nominalScale(),
        maxPixels=1e13
    ).get('nd')
    
    return mean_ndvi



# Function to get the Ratio of ones to total pixels
def get_mask_ones_ratio(mask:ee.Image, scale = 100, in_percentage = True):
    """
    Function to get the percentage or the ratio of ones to total pixels in an Earth Engine image mask.

    Args:
    -----
        `mask` (ee.Image): An Earth Engine image mask.
        `scale` (int, optional): The scale to use for reducing the image. Defaults to 30.
        `in_percentage` (bool, optional): Whether to return the ratio or the percentage. Defaults to True.
        `rescale_attempts` (int, optional): Number of times to rescale the mask if reducer raised a too many pixel error.

    Returns:
    --------
        float: The ratio of ones to total pixels in the mask.
    """
    # Compute the number of ones and total number of pixels in the mask
    #band_name = mask.bandNames().getInfo()[0]
    


    stats = mask.reduceRegion(
        reducer=ee.Reducer.sum().combine(
            reducer2=ee.Reducer.count(),
            sharedInputs=True
        ),
        geometry=mask.geometry(),
        scale=scale,
        maxPixels=1e9,
        bestEffort=True
        )


    # Extract the number of ones and total number of pixels from the result
    ones = stats.get(stats.keys().get(1))
    total = stats.get(stats.keys().get(0))
    

    # Compute the ratio of ones to total pixels
    ratio = ee.Number(ones).divide(total)
    

    # Return the ratio
    return ratio.multiply(100) if in_percentage else ratio


# Function to get the Ratio of Nulls to total pixels that an roi could have
def get_not_nulls_ratio(image:ee.Image, roi:ee.Geometry ,scale = 100, in_percentage = True) -> ee.Number:
    """
        Calculates the ratio of not null null values to total pixels that an ROI (Region of Interest) could have for a given image.
        
        Args:
        -----
        - image (ee.Image): The image for which the nulls ratio needs to be calculated.
        - roi (ee.Geometry): The region of interest for which the nulls ratio needs to be calculated.
        - scale (int, optional): The scale at which to perform the reduction. Defaults to 30.
        
        Returns:
        --------
        - ratio (ee.Number): The ratio of not null null values to total pixels for the given ROI and image.
    """

    # Creates a 1 & 0 mask of the image, 0 on null areas, and 1 for pixels with values
    # th clip is really important since, mask() method goes over boundries.
    mask = image.mask().select(0).clip(roi)
    # Return the ratio
    return get_mask_ones_ratio(mask, scale = scale, in_percentage = in_percentage)



def add_mineral_indices(inImage):
    """
    Adds four new bands (clayIndex, ferrousIndex, carbonateIndex, and rockOutcropIndex) to an input image.
    
    Parameters:
        inImage (ee.Image): The input image to add the new bands to.
        
    Returns:
        ee.Image: The output image with the added bands.
    """
    # Clay Minerals = swir1 / swir2
    clayIndex = inImage.select('SR_B6').divide(inImage.select('SR_B7')).rename('clayIndex')

    # Ferrous Minerals = swir / nir
    ferrousIndex = inImage.select('SR_B6').divide(inImage.select('SR_B5')).rename('ferrousIndex')

    # Carbonate Index = (red - green) / (red + green)
    carbonateIndex = inImage.normalizedDifference(['SR_B4','SR_B3']).rename('carbonateIndex')

    # Rock Outcrop Index = (swir1 - green) / (swir1 + green)
    rockOutcropIndex = inImage.normalizedDifference(['SR_B6','SR_B3']).rename('rockOutcropIndex')

    # Add bands
    outStack = inImage.addBands([clayIndex, ferrousIndex, carbonateIndex, rockOutcropIndex])

    return outStack


def get_closest_image(image_collection:ee.ImageCollection, date:str, clip_dates: int = None) -> ee.Image:
    """
    Returns the closest image in the given image collection to the given date.
    Parameters:
    -----------
    `image_collection` : ee.ImageCollection
        The image collection from which to find the closest image.
    `date` : str or datetime
        The target date as a string in "YYYY-MM-DD" format or a datetime object.
    `clip_dates` : int, optional
        The number of days to clip the image collection to. Only images within this range
        of the target date will be considered. If not specified, all images in the collection
        will be considered.

    Returns:
    --------
    closest_image : ee.Image
        The closest image in the image collection to the target date.

    """
    # Convert the date to milliseconds since the Unix epoch
    date_millis = ee.Date(date).millis()
    
    if clip_dates:
        # Filter the collection to images within 7 days of the target date
        filtered_collection = image_collection.filterDate(
            ee.Date(date).advance(-1*clip_dates, 'day'),
            ee.Date(date).advance(clip_dates, 'day')
        )
    else:
        filtered_collection = image_collection
    
    # Compute the time difference between each image and the target date
    filtered_collection = ee.ImageCollection(
        ee.List(filtered_collection.toList(filtered_collection.size()))
        .map(lambda image: image.set('timeDiff', abs(ee.Number(image.date().millis()).subtract(date_millis))))
    )
    
    # Get the image with the minimum time difference
    closest_image = filtered_collection.sort('timeDiff').first()
    
    return closest_image


# applying the Mult and Add function to the image bands but the QABand
def radiometric_correction(image: ee.Image , sr_bands_list = ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']):
    """
    Applies radiometric correction to the surface reflectance (SR) bands of an input image, and leaves other bands unchanged.

    Args:
        image: An ee.Image object representing the input image.
        sr_bands_list: A list of strings representing the names of the surface reflectance bands to be corrected.

    Returns:
        An ee.Image object with the radiometrically corrected SR bands added as new bands to the input image.
    """
    sr_bands = image.select(sr_bands_list).multiply(2.75e-05).add(-0.2)
    image = image.addBands(sr_bands, None, True)
    return image


def millis_to_date_string(millis: ee.Number) -> ee.String:
    """
    Converts a millisecond timestamp to a formatted date string.

    Args:
        millis: A millisecond timestamp.

    Returns:
        A formatted date string in the format 'YYYY-MM-dd'.
    """
    date = ee.Date(millis)
    return date.format('YYYY-MM-dd')

# In Prevoius version we were using .getInfo() every line, which is not efficient, now we are doing everything on server-side
# then in after the list is ready, we call getInfo() and we print it
def ee_property_printer(s1_collection, propertie_name_list = ['system:time_start','orbitProperties_pass',
                                                     'resolution','resolution_meters','platform_number',
                                                     'productType','orbitNumber_start','orbitNumber_stop',
                                                     'transmitterReceiverPolarisation','system:band_names','instrumentMode',
                                                     'relativeOrbitNumber_stop','relativeOrbitNumber_start','cycleNumber'],first_is_t_in_millis= True):
    """
    A function that prints the properties EarthEngine Image.

    Parameters:
    -----------
    s1_collection : ee.ImageCollection
        The Sentinel-1 image collection whose properties are to be printed.
    propertie_name_list : list of str, optional (default=['system:time_start', 'orbitProperties_pass', 'resolution', 'resolution_meters', 'platform_number', 'productType', 'orbitNumber_start', 'orbitNumber_stop', 'transmitterReceiverPolarisation', 'system:band_names', 'instrumentMode', 'relativeOrbitNumber_stop', 'relativeOrbitNumber_start', 'cycleNumber'])
        A list of property names to be printed for each image in the collection.
    first_is_t_in_millis : bool, optional (default=True)
        A flag indicating whether the first property in `property_name_list` represents a timestamp in milliseconds.
        If True, the timestamp will be converted to a human-readable date string before being printed.

    Returns:
    --------
    None

    Example:
    --------
    >>> collection = ee.ImageCollection('COPERNICUS/S1_GRD').filterDate('2019-01-01', '2019-01-31').filterBounds(geometry)
    >>> sen1_print(collection, propertie_name_list=['system:time_start', 'relativeOrbitNumber_start', 'transmitterReceiverPolarisation'], first_is_t_in_millis=True)
    system:time_start -> [datetime.datetime(2019, 1, 1, 5, 42, 36, 196000), datetime.datetime(2019, 1, 1, 5, 42, 51, 427000), ...]
    relativeOrbitNumber_start -> [17, 35, ...]
    transmitterReceiverPolarisation -> ['VV', 'VV', ...]
    """

    if first_is_t_in_millis:
            agg_list = [s1_collection.aggregate_array(propertie_name_list[0]).map(millis_to_date_string)]
    else:
        agg_list = [s1_collection.aggregate_array(propertie_name_list[0])]
        
    for propertie_name in propertie_name_list[1:]:
        agg_list.append(s1_collection.aggregate_array(propertie_name))
        
    ee_list =ee.List(agg_list)
    
    # this is only to make the output look nice
    max_len = max(len(s) for s in propertie_name_list)  # Find the length of the longest string in the list
    formatted_lst = [f"{s:<{max_len}}".replace(" ", "-") for s in propertie_name_list]  # Add spaces to each string to make them the same length
    
    properties_list = ee_list.getInfo()
    for name, element in zip(formatted_lst,properties_list):
        print(name, "-> ", element,sep="")
        
# Wrapping ee_property_printer() to print the properties of Sentinel-1 and Sentinel-2 image collections
sen1_print = lambda s1_collection: ee_property_printer(s1_collection)
sen2_print = lambda s2_collection: sen1_print(s2_collection, propertie_name_list=['system:time_start', 'CLOUDY_PIXEL_PERCENTAGE',
                                                                                  'CLOUD_SHADOW_PERCENTAGE', 'VEGETATION_PERCENTAGE',
                                                                                  'NOT_VEGETATED_PERCENTAGE', 'CLOUD_COVERAGE_ASSESSMENT',
                                                                                  'GENERATION_TIME', 'SENSING_ORBIT_NUMBER', '',
                                                                                  'NODATA_PIXEL_PERCENTAGE','DATATAKE_TYPE',
                                                                                  'SENSING_ORBIT_NUMBER','SNOW_ICE_PERCENTAGE',
                                                                                  'THIN_CIRRUS_PERCENTAGE','WATER_PERCENTAGE',
                                                                                  'system:band_names'], first_is_t_in_millis=True)
  

def is_col_empty(im_collection):
  '''if collection is empty returns `True`, if has vlues returns `False`'''
  dates = im_collection.size().getInfo()
  if dates:
    return False
  else:
    return True


def mosaic_covers_roi(imgecollection, roi, ref_band_name = 'B2',acceptance_rate = 85,scale = 100, verbose = True):
    '''
    the input is an image collection that has beed filterd by date and boundry

    Returns
    ---
    *  `True`  if the ratio of image to whole area is bigger that acceptance rate
    *  `False` the collection is empy or the ratio of image to whole area is smaller that acceptance rate
    '''
    if is_col_empty(imgecollection): # first we check if collection is not empty
        print('Collection was empty!')
        return False


    img = imgecollection.mosaic().clip(roi).select(ref_band_name) # convertin image to mosaic and clip it by roi

    ratio = get_not_nulls_ratio(img,roi,scale=scale).getInfo()

    if verbose:
        print(f'Mosiac Covers {ratio} percent of the roi.')
    #print(f'Image Pixels = {img_pix_int} / All Pixels = {msk_pix_int}')
    if ratio >= acceptance_rate: 
        print('Mosaic Coverege Accepted')
        return True
    else:
        print('Mosaic Coverege Not Accepted')
        return False





def gee_list_item_remover(img_collection,img_indcies_list:list):
    '''
    the inputs are an image collection, and a list of indices of items to be remove
    and the output is a the image collection whitouth those items
    '''

    print('collection size before removing snowy dates: ',img_collection.size().getInfo())

    img_col_list = img_collection.toList(img_collection.size()) # converting imgcollection to gee list
    img_indcies_list.sort(reverse=True) # we sort the list dscening, bevuse our only option is to remove them one by one 
    #and if we remove for example index 0, then idex 1 becomes 0 and index 2 becomes 1 and so on,
    #to prevent this from happening we start removing form the largest index.
    for indx in img_indcies_list:
        print('image with index:',indx,' removed')
        img_col_list =img_col_list.splice(indx,1) # https://developers.google.com/earth-engine/apidocs/ee-list-splice

    removed_col =ee.ImageCollection(img_col_list) # convert the gee list back to imgcollection
    print('collection size after removing snowy dates: ',removed_col.size().getInfo())
    return removed_col



