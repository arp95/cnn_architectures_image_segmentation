# header files
import torch
import torch.nn as nn
import torchvision
import numpy as np


# model (pretrained_net = vgg16)
class FCN32(torch.nn.Module):

  # init function
  def __init__(self, pretrained_net, num_classes=21):
    super(FCN32, self).__init__()

    # encoder
    self.encoder = torch.nn.Sequential(*list(pretrained_net.features.children()))

    # decoder
    self.decoder = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.BatchNorm2d(512),
        torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.BatchNorm2d(256),
        torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.BatchNorm2d(128),
        torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.BatchNorm2d(64),
        torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.BatchNorm2d(32),
        torch.nn.Conv2d(32, num_classes, kernel_size=1)
    )

  # forward function
  def forward(self, x):
    # apply encoder
    output = self.encoder(x)

    # apply decoder
    output = self.decoder(output)

    # return the predicted label image
    return output
