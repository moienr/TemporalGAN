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
                                                     'relativeOrbitNumber_stop','relativeOrbitNumber_start','cycleNumber'],
                                                    first_is_t_in_millis= True, df_instead_of_print = True):
    """
    A function that prints the properties EarthEngine Image.

    Parameters:
    -----------
    `ImageCollection` : ee.ImageCollection
        The Sentinel-1 image collection whose properties are to be printed.
    `propertie_name_list` : list of str, optional (default=['system:time_start', 'orbitProperties_pass', 'resolution', 'resolution_meters', 'platform_number', 'productType', 'orbitNumber_start', 'orbitNumber_stop', 'transmitterReceiverPolarisation', 'system:band_names', 'instrumentMode', 'relativeOrbitNumber_stop', 'relativeOrbitNumber_start', 'cycleNumber'])
        A list of property names to be printed for each image in the collection.
    `first_is_t_in_millis` : bool, optional (default=True)
        A flag indicating whether the first property in `property_name_list` represents a timestamp in milliseconds.
        If True, the timestamp will be converted to a human-readable date string before being printed.
    `df_instead_of_print` : bool, optional (default=True)
        A flag indicating whether the properties should be printed or returned as a pandas DataFrame. use df when it is the only function in the cell that
        prints something. Otherwise, the order of the prints will be messed up.
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
    if not df_instead_of_print:
        for name, element in zip(formatted_lst,properties_list):
            print(name, "-> ", element,sep="")
    if df_instead_of_print:    
        df = pd.DataFrame(properties_list)
        df.insert(0, 'Property', propertie_name_list)
        styled_df = df.style.set_properties(**{'text-align': 'left'})
        styled_df = styled_df.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
        return styled_df

# Wrapping ee_property_printer() to print the properties of Sentinel-1 and Sentinel-2 image collections
sen1_print = lambda s1_collection: ee_property_printer(s1_collection,df)
sen2_print = lambda s2_collection: ee_property_printer(s2_collection, propertie_name_list=['system:time_start','roi_cloud_cover', 'CLOUDY_PIXEL_PERCENTAGE',
                                                                                  'CLOUD_SHADOW_PERCENTAGE', 'VEGETATION_PERCENTAGE',
                                                                                  'NOT_VEGETATED_PERCENTAGE', 'CLOUD_COVERAGE_ASSESSMENT',
                                                                                  'GENERATION_TIME', 'SENSING_ORBIT_NUMBER',
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




## To Linear and back again!
#as was said in the `Woodhouse pg. 324` we cannot averasge over `db` values.
#so we convert the data into `linear` scale, then we use `.mean()` method

#1 #https://gis.stackexchange.com/questions/424225/convert-sentinel-1-images-data-from-db-to-linear   #some how functions don't work but gives a good idea #used code in 'data_extractro_foliom' to translate to py
#2 #https://gis.stackexchange.com/questions/419849/google-earth-engine-python-expression-syntax       #correct way of writing the functions in python
#3 #https://developers.google.com/earth-engine/apidocs/ee-image-addbands
#4 #https://developers.google.com/earth-engine/apidocs/ee-image-expression

# adds band `VV_lin` to the images, code in link #1 doesn't work beucase it uses name `VV` and sets the replace in #3 to True which then overwrites the band we created with original `VV` !
def toLinear(db:ee.Image):
    """
    Adds a band named 'VV_lin' to the input image collection where the values of the band 'VV' are converted to linear scale using the formula pow(10, db / 10).
    
    Args:
        `db`: An ee.ImageCollection object with a 'VV' band in dB scale.
    
    Returns:
        An ee.Image object with an additional band named 'VV_lin'.
    """
    lin = db.expression('pow(10, db / 10)', {'db': db.select('VV')}).rename('VV_lin')
    return db.addBands(lin)

# reads the added band `VV_linear` and converts it to db scale, we use this after we averaged over linear values.
def toDb(linear:ee.Image, input_band_name:str = 'VV_lin'):
    """
    Converts the linear `VV_lin` band of the input image collection `linear` to decibels using the formula 10 * log10(linear).
    The resulting band is named `VV_db`.
    
    Args:
        `linear`: An Earth Engine image collection with a band named 'VV_lin' in linear scale.
        `input_band_name`: The name of the linear band to convert to decibels. Default is 'VV_lin'.
    
    Returns:
        An Earth Engine image with an added band named 'VV_db' in decibels.
    """
    lin = linear.expression('10 * log10(linear)', {'linear': linear.select(input_band_name)}).rename('VV_db')
    return linear.addBands(lin)



def get_s2(date_range: tuple,roi,max_cloud = 10,max_snow = 5, scl = False, check_snow = False):
    ''' 
    Inputs
    ---
    `date_range` : the date range two string element tupple in gee format  like `('2020-02-01','2020-03-01')`
    `roi` : the region of interest, can be a `ee.Geometry` or `ee.Feature` or `ee.FeatureCollection`
    `max_cloud` : the maximum cloud cover percentage, default is 10
    `max_snow` : the maximum snow cover percentage, default is 5
    `scl` : if True, the function will find the Cloud Cover based on SCL band, if False, it will use the QA60 band, default is False
    `check_snow` : if True, the function will filter the collection by snow cover, if False, it will not, default is False
        don't use it for cloudy summer images - it has a high false positive rate.
    

    Algorithm
    ---
    the function first tries to find an image or images where the S2 scene fully covers the 
    `roi`.
    if the first attemp was not sucsessful then it will look for all the secnes that has some 
    overlap with the `roi`, with the user specified criteria, then it checks that the new collection
    can cover the whole `roi`, if not, it will expand the `date_range` defined by user by one motth 
    and start rcursing untill the valid result 

    Return
    ---
    the function reutrns an GEE `image collection`
    
    '''
    print('◍◍Finding S2')
    #first atempt
    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
                        .filterDate(date_range[0], date_range[1]) \
                        .filterBounds(roi) \
                        .filter(ee.Filter.contains('.geo', roi)) #this line checks if the scene completly covers the roi, which mean roi is in the scene
    
    if check_snow:
        s2 = s2.filter(ee.Filter.lt('SNOW_ICE_PERCENTAGE',max_snow)) 
    
    if scl:
        s2 = s2.map(lambda img: img.set('roi_cloud_cover', get_mask_ones_ratio(get_cloud_mask_form_scl(img))))               
    else:                                    
        s2 = s2.map(lambda img: img.set('roi_cloud_cover', get_mask_ones_ratio(get_cloud_mask(img)[2])))
    s2 = s2.filter(ee.Filter.lt('roi_cloud_cover',5)) 

    if  is_col_empty(s2): # if the collection is empty we go and check if therse a mosaic that covers the whole area
        print('◍No single scene coverge was found!')

        s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
                        .filterDate(date_range[0], date_range[1]) \
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',max_cloud)) \
                        .filter(ee.Filter.lt('SNOW_ICE_PERCENTAGE',max_snow)) \
                        .filterBounds(roi)
                        
        if scl:
            s2 = s2.map(lambda img: img.set('roi_cloud_cover', get_mask_ones_ratio(get_cloud_mask_form_scl(img))))               
        else:                                    
            s2 = s2.map(lambda img: img.set('roi_cloud_cover', get_mask_ones_ratio(get_cloud_mask(img)[2])))
        s2 = s2.filter(ee.Filter.lt('roi_cloud_cover',max_cloud)) 
        
        
        if mosaic_covers_roi(s2,roi,ref_band_name = 'B2'):
            print(f'◍Image Mosaic found in date range of {date_range[0]} to {date_range[1]}')
            return s2
        else:
            new_date = month_add(date_range[1])
            print(' - Month Range Expaned ', f'new range: {date_range[0]} -to- {new_date}')
            return get_s2((date_range[0],new_date),roi,max_cloud)
    else:
        print('◍Single scene coverege was fount!')
        return s2
    
    

def s1_col_func(start_date,end_date,roi,path,single_scene=False):
    if single_scene:
        s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
          .filterDate(start_date, end_date) \
          .filterBounds(roi) \
          .filter(ee.Filter.eq('resolution','H')) \
          .filter(ee.Filter.eq('instrumentMode','IW'))\
          .filter(ee.Filter.contains('.geo', roi)) \
          .filterMetadata('orbitProperties_pass', 'equals', path) 
    else:
        s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
          .filterDate(start_date, end_date) \
          .filterBounds(roi) \
          .filter(ee.Filter.eq('resolution','H')) \
          .filter(ee.Filter.eq('instrumentMode','IW'))\
          .filterMetadata('orbitProperties_pass', 'equals', path) 

    return s1_collection



def get_s1(s2_collection,roi,max_snow = 10,priority_path = 'ASCENDING',
           check_second_priority_path = True,month_span = 1,retry_days = 0,
           snow_removal=False, best_orbit = True):
    '''
    Inputs
    ---
    * s2_collection
    * roi
    * max_snow: images with more snow thatn `max_snow` are considered as snowy 
    * priority_path: whether to first check for Ascending or Dscending Data
    * check_second_priority_path: whether to check the other path as well
    * retry_days: Increasing the date span by how many days in case of empty collection
    
    '''
    print('◍◍Finding S1')
    # if ASC is prioriy then DESC is the second prioriy and vice versa
    if priority_path == 'ASCENDING':
        second_priority = 'DESCENDING'
    else:
        second_priority = 'ASCENDING'


    mean_s2_date = mean_date(s2_collection.aggregate_array('system:time_start').getInfo()) #find the average date of S2 collection which will be the center of our S1 collection
    print('mean date: ', mean_s2_date)

    start_date = month_add(mean_s2_date,months_to_add =-1 * month_span) # 1 month before center date
    end_date   = month_add(mean_s2_date,months_to_add = month_span) # 1 month after  center date
  
    # in case of failure in the first atemmpet the reucrse will activate and increase the date span
    if retry_days !=0:
        start_date = day_add(start_date,days_to_add = -retry_days)
        end_date   = day_add(end_date,days_to_add =  retry_days)
    print('final date range: ',start_date,end_date)

    if snow_removal:
        # S2 collection in the S1 Collection range to find the snowy days
        s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.greaterThan('SNOW_ICE_PERCENTAGE',max_snow)) \
                    .filterBounds(roi) #finding all the images in the two month range that are snowy

        # if an image is snowy we consider 2 days before and after it as snowy becuase S2 temporal resouliton is 5 days
        # the collection should be s2 and not the s2_collection, in the last version I made this mistake which resualted in wrong snowy dates.
        snowy_days = milsec2date(s2.aggregate_array('system:time_start').getInfo(),no_duplicate=True)
        print('Snowy days       : ',snowy_days)
        snowy_days_buffered = day_buffer(snowy_days)
        print('Snowy days Buffed: ',snowy_days_buffered)
  
    #====================================================================================================

    ## First we check if theres and Priority_path data that covers the whole area with a singe scene
    print(f'◍checking for {priority_path} single scene')
    s1_collection = s1_col_func(start_date,end_date,roi,path = priority_path,single_scene=True)


    ## if not,  we check if theres and Descending data that covers the whole area
    if is_col_empty(s1_collection) and check_second_priority_path:
        print(f'◍{priority_path} singe scene was not fount, checking {second_priority} single scene ...')
        s1_collection = s1_col_func(start_date,end_date,roi,path = second_priority,single_scene=True)

    if is_col_empty(s1_collection):
        print(f'◍No single scene was fount, checking {priority_path} mosaic ...')
        s1_collection = s1_col_func(start_date,end_date,roi,path = priority_path,single_scene=False)


    if (is_col_empty(s1_collection) or not mosaic_covers_roi(s1_collection,roi,ref_band_name = 'VV')) and check_second_priority_path:
        print(f'◍{priority_path} and {second_priority} singe scene was not fount, {priority_path} mosaic was not found,  checking {second_priority}  mosaic ...')
        s1_collection = s1_collection = s1_col_func(start_date,end_date,roi,path = second_priority,single_scene=False)
      
    if is_col_empty(s1_collection) or not mosaic_covers_roi(s1_collection,roi,ref_band_name = 'VV', verbose=False):
        print('◍No S1 dataset was found!')
        if not check_second_priority_path: print('◍check_second_priority_path was set to True and check range was buffed by 5 days')

        return get_s1(s2_collection,roi,max_snow = max_snow+2,priority_path=priority_path,check_second_priority_path=True,retry_days = retry_days+5 ,month_span = month_span) # if the check only primiray didn't work for the first time,
        
    else: #probabily won't work, so we set check_second_priority_path to True
        if snow_removal:
            s1_date_list = milsec2date(s1_collection.aggregate_array('system:time_start').getInfo(),no_duplicate=False)
            s1_snowy_dates = list_intersection(s1_date_list,snowy_days_buffered)

            s1_snow_removed_col = gee_list_item_remover(s1_collection,s1_snowy_dates)
            print('◍Collection Found!')
            
            if best_orbit:
                s1_snow_removed_col = s1_snow_removed_col.filter(ee.Filter.eq('relativeOrbitNumber_start', get_best_sen1_orbit(s1_snow_removed_col,roi)))
            return s1_snow_removed_col
        else:
            if best_orbit:
                s1_collection = s1_collection.filter(ee.Filter.eq('relativeOrbitNumber_start', get_best_sen1_orbit(s1_collection,roi)))
            return s1_collection



def s1s2(roi, date = ('yyyy-mm-dd', 'yyyy-mm-dd'),priority_path = 'ASCENDING',
         check_second_priority_path = True,month_span = 1,max_cloud = 5,max_snow = 5,
         retry_days = 0, best_orbit = True, snow_removal = False):
    """
    Returns Sentinel-2 and Sentinel-1 image collections filtered by specified parameters.

    Parameters:
    -----------
    `roi` : ee.Geometry
        The region of interest to filter the image collections by.

    `date` : tuple, optional
        A tuple containing two strings representing the start and end dates in 'yyyy-mm-dd' format.
        Default is ('yyyy-mm-dd', 'yyyy-mm-dd').

    `priority_path` : str, optional
        The priority path to filter the Sentinel-1 image collection by.
        Default is 'ASCENDING'.

    `check_second_priority_path` : bool, optional
        A flag to determine whether to check the second priority path for Sentinel-1 images.
        Default is True. if True, when First priority path is not found, the second priority path will be checked.

    `month_span` : int, optional
        The number of months from the mean date of S2 Collection to search for Sentinel-1 images.
        Default is 1.

    `max_cloud` : int, optional
        The maximum cloud cover percentage to filter Sentinel-2 images by.
        Default is 5.

    `max_snow` : int, optional
        The maximum snow cover percentage to filter Sentinel-1 and Sentinel-2 images by.
        Default is 5.

    `retry_days` : int, optional
        The number of days to retry fetching Sentinel-1 images in case of failure.
        Default is 0.
    
    `best_orbit` : bool, optional
        A flag to determine whether to reutrn all the orbits or only the best orbit.
        Default is True. set it false on rois that have a low coverage.
    `snow_removal` : bool, optional
        A flag to determine whether to remove snowy images from the Sentinel-1 image collection.
        - don't use it on sumemer images, it has a lot of false positives.

    Returns:
    --------
    tuple
        A tuple containing the filtered Sentinel-2 and Sentinel-1 image collections.
    
    """
    s2_col = get_s2(date,roi,max_cloud,max_snow)
    s1_col = get_s1(s2_col, roi, max_snow, priority_path, check_second_priority_path, month_span=month_span, retry_days=retry_days,
                    best_orbit = best_orbit, snow_removal = snow_removal)
    return s2_col,s1_col



def find_most_repeated_element(ee_list: ee.List) -> ee.List:
    """
    Given an Earth Engine List, this function finds the element(s) that is/are the most repeated.

    Args
    ----
    `ee_list`: An Earth Engine List containing elements.

    Returns
    -------
    ee.List: An Earth Engine List containing the element(s) with the highest count of occurrences.
    
    Usage
    -----
    ```
    rel_list = s1_col.aggregate_array('relativeOrbitNumber_start')
    x = find_most_repeated_element(rel_list)
    ----
    x: ee.List([110,8]) # The most repeated elements are 110 and 8
    ```
    
    """
    # Get the distinct elements in the list
    distinct_elements = ee_list.distinct()

    # Map over the distinct elements and count their occurrences in the list
    def count_occurrences(element):
        count = ee_list.filter(ee.Filter.eq('item', element)).size()
        return ee.Feature(None, {'element': element, 'count': count})

    occurrences = distinct_elements.map(count_occurrences)
    occurrences = ee.FeatureCollection(occurrences)
    # Sort the occurrences by count in descending order
    sorted_occurrences = occurrences.sort('count', False)

    # Get the element(s) with the highest count
    max_count = sorted_occurrences.first().get('count')
    most_repeated_elements = sorted_occurrences.filter(ee.Filter.eq('count', max_count)).aggregate_array('element')

    return most_repeated_elements



def get_band_average(image:ee.Image, band_name:str, roi=None)-> ee.Number:
    """
    Computes the average value of a given band in an Earth Engine image within the image's bounding box.

    Args
    ----
        image: An Earth Engine image.
        band_name: A string specifying the name of the band of interest.
        roi: An Earth Engine geometry specifying the region of interest. If not specified, the image's bounding box will be used.
    Returns
    -------
        The average value of the specified band within the image's bounding box as a float.
        
    Usage
    -----
    ```
    sen1_img = sen1_col.mean().clip(roi)
    get_band_average(sen1_img,'angle')
    ----
    ee.Number(31.777401400973883)
    ```
    Or:
    
    ```
    sen1_col_with_angle = sen1_col.map(lambda img: img.set('avg_angle',get_band_average(img,'angle',roi)))
    ```
    """
    # Get the band of interest
    band = image.select(band_name)
    if roi:
        band = band.clip(roi)

    # Get the image's bounding box
    bbox = image.geometry().bounds()

    # Compute the mean of the band within the bounding box
    band_mean = band.reduceRegion(reducer=ee.Reducer.mean(),scale = 100, geometry=bbox, maxPixels=1e9)

    # Return the mean value as a float
    return band_mean.get(band_name)


def collection_splitter(collection, property_name:str, property_list:ee.List):
    """
    Split an image collection into multiple image collections based on a list of property values.
    
    Note:
    -----
        you have to convert the resulting ComputedObject to an ImageCollection on each get call, to use aggregate_array, etc.
        
    ```
    col_list = collection_splitter(collection, property_name, property_list)
    col = ImageCollection(col_list.get(0)) # This is necessary to use aggregate_array, etc.
    col.aggregate_array('property_name')
    ```

    Args:
    -----
        `collection`: The image collection to split.
        `property_name`: The name of the property to filter by.
        `property_list`: The list of property values to filter by.

    Returns:
    --------
        A list of image collections, where each image collection contains images with the specified property value.
            
        
    """
    
    
    # Map the filter function over each property value in the list, and convert the resulting ComputedObject to an ImageCollection.
    col_list = property_list.map(lambda x: collection.filter(ee.Filter.eq(property_name, x)))
    return col_list
    
    
def get_best_sen1_orbit(s1_col,clip_roi:ee.Geometry, choose_smallest = False) -> ee.Number:
    """
    Get the relative orbit number of the best Sentinel-1 orbit based on the number of times it appears in the input collection 
    and the average incidence angle of the images in that orbit.

    Args:
    -----
        `s1_col`: The input Sentinel-1 image collection.
        `clip_roi`: The region of interest to clip the images to. Defaults to None.
        `choose_smallest`: A boolean indicating whether to choose the smallest (True) or largest (False) average incidence angle.
                            in SAR images, the larger the incidence angle, the better the range resolution. (although it is all resampled to 10m)
                          Defaults to False.
                          
        * it is important to use roi, when calculating local average, otherwise the average will be calculated over the whole image.
            for expample when calculating `angle` band, the average will be calculated over the whole image, and not over the roi, and will
            result for the same value around `38` for all the `S1` images. 


    Returns:
    --------
        The relative orbit number of the best Sentinel-1 orbit based on the number of times it appears in the input collection 
        and the average incidence angle of the images in that orbit.
    """
    rel_list = s1_col.aggregate_array('relativeOrbitNumber_start')
    most_list = find_most_repeated_element(rel_list)
    s1_col_filtered = s1_col.filter(ee.Filter.inList('relativeOrbitNumber_start',most_list))
    s1_col_with_avg_angle = s1_col_filtered.map(lambda img: img.set('avg_angle',get_band_average(img,'angle',clip_roi)))
    return s1_col_with_avg_angle.sort('avg_angle',choose_smallest).first().get('relativeOrbitNumber_start')
    