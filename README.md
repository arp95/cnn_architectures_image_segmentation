# CNN Architectures for Image Segmentation

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)
---


### Author
Arpit Aggarwal


### Introduction to the Project
In this project, different CNN Architectures like FCN-8, FCN-16, FCN-32, UNet, SegNet were used for the task of image segmentation on the Pascal VOC 2012 dataset. The input to the CNN networks was a (256 x 256 x 3) image and the number of classes were 21. The CNN architectures were implemented in PyTorch and the loss function was Cross Entropy Loss(2d). The hyperparameters to be tuned were: Number of epochs(e), Learning Rate(lr), momentum(m), weight decay(wd) and batch size(bs). 


### Data
The dataset used was Pascal VOC 2012. It is available in torchvision and no additional download was needed.


### Architectures Used
Different CNN architectures used for the task of image segmentation are given below:

1. FCN-8: Encoder/Backbone used is VGG-16.
2. FCN-16: Encoder/Backbone used is VGG-16.
3. FCN-32: Encoder/Backbone used is VGG-16.
4. UNet: Encoder/Backbone used is VGG-16.
5. SegNet: Encoder/Backbone used is VGG-16.


### Results
The results after using different CNN architectures are given below:

1. <b>FCN-32</b><br>

Training mIoU = 20.2% and Validation mIoU = 7.3% (epochs = 150, lr = 0.0001, momentum = 0.95, bs = 8, wd = 5e-4)


2. <b>FCN-16</b><br>

Training mIoU = 51.4% and Validation mIoU = 18.8% (epochs = 150, lr = 0.001, momentum = 0.95, bs = 8, wd = 1e-3)


3. <b>FCN-8</b><br>

Training mIoU = 50.8% and Validation mIoU = 18.2% (epochs = 150, lr = 0.001, momentum = 0.95, bs = 8, wd = 5e-4)


### Software Required
To run the jupyter notebooks, use Python 3. Standard libraries like Numpy and PyTorch are used.


### Credits
The following links were helpful for this project:
1. https://github.com/pytorch/tutorials
2. https://github.com/bodokaiser/piwise/
3. https://github.com/meetshah1995/pytorch-semseg/
