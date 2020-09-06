# header files needed
import numpy as np
import glob
import cv2
import torch
import torch.nn as nn
import torchvision


# transforms (as the dataset class is a dictionary)
class Resize(object):

  # init method
  def __init__(self, imageresize, maskresize):
    self.imageresize = imageresize
    self.maskresize = maskresize

  # call method
  def __call__(self, sample):
    image, mask = sample["image"], sample["mask"]
    image = image.transpose(1, 2, 0)
    image = cv2.resize(image, self.imageresize, cv2.INTER_AREA)
    mask = cv2.resize(mask, self.maskresize, cv2.INTER_AREA)
    image = image.transpose(2, 0, 1)
    return {"image": image, "mask": mask}


class ToTensor(object):

  # call method
  def __call__(self, sample):
    image, mask = sample["image"], sample["mask"]
    return {"image": torch.from_numpy(image), "mask": torch.from_numpy(mask)}


class Normalize(object):

  # call method
  def __call__(object):

    # call method
    def __call__(self, sample):
      image, mask = sample["image"], sample["mask"]
      return {"image": image.type(torch.FloatTensor) / 255., "mask": mask.type(torch.FloatTensor) / 255.}
