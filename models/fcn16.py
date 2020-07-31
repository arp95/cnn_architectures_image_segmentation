# header files
import torch
import torch.nn as nn
import torchvision
import numpy as np


# model
class FCN16(torch.nn.Module):

  # init function
  def __init__(self, pretrained_net, num_classes=21):
    super(FCN16, self).__init__()

    # enocder 1 and encoder 2
    self.encoder_1 = torch.nn.Sequential(*list(pretrained_net.features.children())[:-10])
    self.encoder_2 = torch.nn.Sequential(*list(pretrained_net.features.children())[-10:])

    # decoder 1 and decoder 2
    self.decoder_1 = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
        torch.nn.ReLU(inplace=True)
    )

    self.decoder_2 = torch.nn.Sequential(
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
    enc_output_1 = self.encoder_1(x)
    output = self.encoder_2(enc_output_1)

    # apply decoder
    dec_output_1 = self.decoder_1(output)
    output = dec_output_1 + enc_output_1
    output = self.decoder_2(output)

    # return the predicted label image
    return output
