# header files needed
import numpy as np
import glob
import cv2
import torch
import torch.nn as nn
import torchvision


# custom dataset
class SegDataset(torch.utils.data.Dataset):

  # init method
  def __init__(self, rootDir, imageFolder, maskFolder, transform=None, seed=None, fraction=None, subset=None):
    self.imageColorFlag = 1
    self.maskColorFlag = 0
    self.rootDir = rootDir
    self.transform = transform

    if not fraction:
      self.imageNames = sorted(glob.glob(self.rootDir + imageFolder + "/*"))
      self.maskNames = sorted(glob.glob(self.rootDir + maskFolder + "/*"))
    else:
      self.fraction = fraction
      if seed:
        np.random.seed(seed)
      self.imageList = glob.glob(self.rootDir + imageFolder + "/*")
      self.maskList = glob.glob(self.rootDir + maskFolder + "/*")
      indices = np.arange(len(self.imageList))
      np.random.shuffle(indices)
      self.imageList = np.array(self.imageList)[indices]
      self.maskList = np.array(self.maskList)[indices]

      if subset == "Train":
        self.imageNames = self.imageList[:int(np.ceil(len(self.imageList) * (1 - self.fraction)))]
        self.maskNames = self.maskList[:int(np.ceil(len(self.maskList) * (1 - self.fraction)))]
      else:
        self.imageNames = self.imageList[int(np.ceil(len(self.imageList) * (1 - self.fraction))):]
        self.maskNames = self.maskList[int(np.ceil(len(self.maskList) * (1 - self.fraction))):]

  # len method
  def __len__(self):
    return len(self.imageNames)

  ## get item method
  def __getitem__(self, index):
    imageName = self.imageNames[index]
    maskName = self.maskNames[index]
    image = cv2.imread(imageName, self.imageColorFlag).transpose(2, 0, 1)
    mask = cv2.imread(maskName, self.maskColorFlag)

    sample = {"image": image, "mask": mask}
    if self.transform:
      sample = self.transform(sample)
    return sample
