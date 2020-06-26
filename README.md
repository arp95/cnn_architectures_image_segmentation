# CNN Architectures for Image Segmentation

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)
---


### Author
Arpit Aggarwal


### Introduction to the Project
In this project, different CNN Architectures like FCN-8, FCN-16, FCN-32, UNET were used for the task of image segmentation on the Pascal VOC 2012 dataset. The input to the CNN networks was a (256 x 256 x 3) image and the number of classes were 22. The CNN architectures were implemented in PyTorch and the loss function was Cross Entropy Loss(2d). The hyperparameters to be tuned were: Number of epochs(e), Learning Rate(lr), momentum(m), weight decay(wd) and batch size(bs). 


### Data
The dataset used was Pascal VOC 2012. It is available in torchvision and no additional download was needed.


### Results
The results after using different CNN architectures are given below:


### Software Required
To run the jupyter notebooks, use Python 3. Standard libraries like Numpy and PyTorch are used.


### Credits
The following links were helpful for this project:
1. https://github.com/pytorch/tutorials
2. https://github.com/bodokaiser/piwise/
