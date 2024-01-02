# TSGAN: An Optical-to-SAR Dual Conditional GAN for Optical based SAR Temporal Shifting

This is the code implementation of the paper [TSGAN: An Optical-to-SAR Dual Conditional GAN for Optical based SAR Temporal Shifting](https://arxiv.org/abs/2401.00440) by Moien Rangzan, Sara Attarchi, Richard Gloaguen and Seyed Kazem Alavipanah. The paper is submitted to ISPRS Journal of Photogrammetry and Remote Sensing.

# Abstract
In contrast to the well-investigated field of SAR-to-Optical translation, this study explores the lesser-investigated domain of Optical-to-SAR translation, a challenging field due to the ill-posed nature of this translation. The complexity arises as a single optical data can have multiple SAR representations based on the SAR viewing geometry. We propose a novel approach, termed SAR Temporal Shifting, which inputs an optical data from the desired timestamp along with a SAR data from a different temporal point but with a consistent viewing geometry as the expected SAR data, both complemented with a change map of optical data during the intervening period. This model modifies the SAR data based on the changes observed in optical data to generate the SAR data for the desired timestamp. Our model, a dual conditional Generative Adversarial Network (GAN), named Temporal Shifting GAN (TSGAN), incorporates a siamese encoder in both the Generator and the Discriminator. To prevent the model from overfitting on the input SAR data, we employed a change weighted loss function. Our approach surpasses traditional translation methods by eliminating the GAN's fiction phenomenon, particularly in unchanged regions, resulting in higher SSIM and PSNR in these areas. Additionally, modifications to the Pix2Pix architecture and the inclusion of attention mechanisms have enhanced the model's performance on all regions of the data. This research paves the way for leveraging legacy optical datasets, the most abundant and longstanding source of Earth datary data, extending their use to SAR domains and temporal analyses.


# Proposed Model
The architecutre of the proposed model is shown below. For detailed information please refer to the paper.

| Generator | Discriminator | 
| :---: | :---: |
| ![](readme_assests/generator.jpg) | ![](readme_assests/Discriminator.jpg) |


# How to use

## 1. Download the dataset
To download the dataset you can simply run [this notebook](./dataset/Dataset_creator.ipynb) in on your local machine or on Google Colab. The notebook will download the dataset and preprocess it.

A detailed description of the dataset and how to add your own data can be found in [dataset](./dataset/) folder.

## 1.1. Hard Evaluation
In the paper, we mention that in order to evaluate the model, we have used a hard evaluation method. In this method, selected around 150 images from the test dataset, these images where showing dicernable urban changes. These images are listed in [this file](.\changedetection\changed_pairs.csv). 

if you went on and build your own dataset, you can use the same method to evaluate your model.Follow [this notebook](./changedetection/tensor_cd.ipynb) to create a new `.csv` file containing the images you want to evaluate your model on.



## 2. Train the model
To train the model run the [train.py](./train.py) file. 

```bash
> conda activate pytorch_env
> python train.py --num_epoch <number of epochs> --changed_l1_weight <weight of the changed L1 loss> 
```

Example:
```bash 
> python train.py --num_epoch 10 --changed_l1_weight 5 --gen_version 1.3 --no_input_change_map 
```

Help:
```bash
> python train.py --help
```




## Results
You can find the results of the model in detain in the paper. 

<!-- ![Results](readme_assests/example1.jpg) -->

![More Results](readme_assests/more_examples.jpg)


![Removal](readme_assests/removal%2000_00_00-00_00_30.gif)

![Creation](readme_assests/creation%2000_00_00-00_00_30.gif)

![Attention](readme_assests/att%2000_00_00-00_00_30.gif)


# There is more!
The models you saw in the paper are only a handful of the models that have been tested. 

In the [temporalgan](./temporalgan/) folder you can find the code for the most of the models that have been tested. I tested mupltiple attention mechanisms, at different levels of the model. CBAM, and PAM are the two attention mechanisms that didn't make it to the paper.

Also GLAM has been used in both 8x8 and 16x16 levels of the model.

Be carefull in deleting unwanted models, as subversion of each model inherits from the previous model. For example, if you delete the `v1.2` folder, you will corrupt the `v1.3`, `v1.5`, and `v1.6` models.

Here is a buitifull sketch of my initial ideas for the model architecture.

![Model archi](readme_assests/model%20arch.png)


# Credits
If you find this work useful, please consider citing:

```bibtex
    @misc{rangzan2023tsgan,
      title={TSGAN: An Optical-to-SAR Dual Conditional GAN for Optical based SAR Temporal Shifting}, 
      author={Moien Rangzan and Sara Attarchi and Richard Gloaguen and Seyed Kazem Alavipanah},
      year={2023},
      eprint={2401.00440},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2401.00440}}

```
