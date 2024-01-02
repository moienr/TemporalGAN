# How to donwload, and add your own data

## 1. Download the dataset

To download the dataset you can simply run [this notebook](./Dataset_creator.ipynb) in on your local machine or on Google Colab. The notebook will download the dataset and preprocess it.

This fellowchart shows the steps of the dataset creation process.

![Dataset creation](../readme_assests/dataset_felowchart_v2.jpg)




To add your own data to the dataset you can adjust the `.xlsx` files in the [dataset](.\dataset) folder, by addin a new row for each ROI. The `.xlsx` files are in the following format:




Each row in the `.xlsx` files represents an ROI. With the following columns:

- **name**: The name of the ROI. This name will be used to create the folder structure.

- **year**: The year of the ROI. This will be used to create the folder structure. (should be two digits e.g. 21)

- **roi**: the roi in format of a list of coordinates. e.g. `[[[10,10],[10,20],[20,20],[20,10]]]`

- **date**: start and end date of S2 search. (should be in format of `YYYY-MM-DD`)

- **priority_path**: First check Ascending or Descending orbits. 

- **check_second_priority_path**: If the first priority path is not available, check the second priority path.

- **max_cloud**: Maximum cloud coverage of S2 images.

- **max_snow**: Maximum snow coverage of S2 images. If you are using high max cloud percentages, sentinel algorithm can falsely detect snow as cloud. 

- **month_span**: The inital expantion of month around the S2 mean date. For S1 search.

- **retry_days**: The number of days to add to the span search if no S1 image was found.

- **train_test**: Whether to use the ROI for training or testing. (should be `train` or `test`)

- **type**: Addional information about the ROI.

Make sure you have two `.xlsx` files. Each represneting a year.

**Update**: I am working on making a python module instead of the need of all above steps. But you're gonna need a google cloud account to use it. I will update this section as soon as I finish it.