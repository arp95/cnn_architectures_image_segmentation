# CNN Architectures for Image Segmentation

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)
---


### Author
Arpit Aggarwal


### Introduction to the Project
In this project, different CNN Architectures like FCN-8, FCN-16, FCN-32, Pretrained-FCN, Pretrained-DeepLabv3 were used for the task of image segmentation on a custom dataset (CrackForest). The input to the CNN networks was a (320 x 480 x 3) image and the number of classes were 1 (Crack or not). The CNN architectures were implemented in PyTorch and the loss function was Mean Square Error(MSE). The hyperparameters to be tuned were: Number of epochs(e), Learning Rate(lr), momentum(m), weight decay(wd) and batch size(bs). 


### Data
The custom dataset used was CrackForest. The dataset can be downloaded from here: https://github.com/msminhas93/DeepLabv3FineTuning


### Architectures Used
Different CNN architectures used for the task of image segmentation are given below:

1. FCN-8: Encoder/Backbone used is VGG-16.
2. FCN-16: Encoder/Backbone used is VGG-16.
3. FCN-32: Encoder/Backbone used is VGG-16.
4. Pretrained-FCN: Encoder/Backbone used is ResNet-101
5. Pretrained-DeepLabv3: Encoder/Backbone used is ResNet-101


### Software Required
To run the jupyter notebooks, use Python 3. Standard libraries like Numpy and PyTorch are used.


### Credits
The following links were helpful for this project:
1. https://github.com/msminhas93/DeepLabv3FineTuning
2. https://github.com/pytorch/tutorials
3. https://github.com/bodokaiser/piwise/
4. https://github.com/meetshah1995/pytorch-semseg/
