# CNN Architectures for Medical Image Segmentation

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)
---


### Author
Arpit Aggarwal


### Introduction to the Project
In this project, different CNN Architectures like FCN-8, FCN-16, FCN-32, U-Net and SegNet were used for the task of medical image segmentation. The input to the CNN networks was a (224x224x3) image and the number of classes were equal to 1. The CNN architectures were implemented in PyTorch and the loss function was Binary Cross Entropy(BCE) Loss. The hyperparameters to be tuned were: Number of epochs(e), Learning Rate(lr) and batch size(bs). 


### Data
The custom dataset used was ISIC 2018 for melanoma detection. The dataset can be downloaded from here: https://challenge2018.isic-archive.com/task1/training/
<p align="center">
  <img src="scripts/1.jpg" width="256" title="Melanoma Image">
</p> 
<p align="center">
  <img src="scripts/2.png" width="256" title="Segmented Mask">
</p>


### Architectures Used
Different CNN architectures used for the task of medical image segmentation are given below:<br>
FCN-8, FCN-16, FCN-32, U-Net and SegNet


### Results
The results after using different CNN architectures are given below:

1. <b>U-Net</b><br>

Training Accuracy = 96.46% and Validation Accuracy = 94.19% (e=100, lr=0.001, bs=8)<br>


2. <b>FCN-8</b><br>

Training Accuracy = 95.76% and Validation Accuracy = 93.37% (e=100, lr=0.001, bs=8)<br>


3. <b>SegNet</b><br>

Training Accuracy = 92.43% and Validation Accuracy = 90.82% (e=100, lr=0.001, bs=8)<br>


### Software Required
To run the jupyter notebooks, use Python 3. Standard libraries like Numpy and PyTorch are used.


### Credits
The following links were helpful for this project:
1. https://github.com/pytorch/tutorials
2. https://github.com/bodokaiser/piwise/
3. https://github.com/meetshah1995/pytorch-semseg/
