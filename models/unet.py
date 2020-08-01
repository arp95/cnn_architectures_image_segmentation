# header files
import torch
import torch.nn as nn
import torchvision
import numpy as np


# model (pretrained_net = vgg16)
class UNet(torch.nn.Module):

  # init function
  def __init__(self, pretrained_net, num_classes=21):
    super(UNet, self).__init__()

    # encoder_1, encoder_2, encoder_3, encoder_4
    self.encoder_block_1 = torch.nn.Sequential(*list(pretrained_net.features.children())[:-37])
    self.encoder_block_2 = torch.nn.Sequential(*list(pretrained_net.features.children())[-37:-30])
    self.encoder_block_3 = torch.nn.Sequential(*list(pretrained_net.features.children())[-30:-20])
    self.encoder_block_4 = torch.nn.Sequential(*list(pretrained_net.features.children())[-20:-10])

    # center block
    self.center = torch.nn.Sequential(
        torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(1024),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(1024),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(inplace=True)
    )

    # decoder block 1
    self.decoder_block_1 = torch.nn.Sequential(
        torch.nn.Conv2d(1024, 512, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(inplace=True),
        torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(inplace=True)
    )

    # decoder block 2
    self.decoder_block_2 = torch.nn.Sequential(
        torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(inplace=True),
        torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(inplace=True)
    )

    # decoder block 3
    self.decoder_block_3 = torch.nn.Sequential(
        torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(inplace=True),
        torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace=True)
    )

    # decoder block 4
    self.decoder_block_4 = torch.nn.Sequential(
        torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace=True)
    )

    # final block
    self.final_block = torch.nn.Sequential(
        torch.nn.Conv2d(64, num_classes, kernel_size=1)
    )

  # forward function
  def forward(self, x):

    # apply encoder block 1, 2, 3, 4
    enc_1 = self.encoder_block_1(x)
    enc_2 = self.encoder_block_2(enc_1)
    enc_3 = self.encoder_block_3(enc_2)
    enc_4 = self.encoder_block_4(enc_3)

    # apply center block
    cen = self.center(enc_4)

    # apply decoder block 1, 2, 3, 4
    dec_1 = self.decoder_block_1(torch.cat([cen, torch.nn.functional.upsample_bilinear(enc_4, cen.size()[2:])], 1))
    dec_2 = self.decoder_block_2(torch.cat([dec_1, torch.nn.functional.upsample_bilinear(enc_3, dec_1.size()[2:])], 1))
    dec_3 = self.decoder_block_3(torch.cat([dec_2, torch.nn.functional.upsample_bilinear(enc_2, dec_2.size()[2:])], 1))
    dec_4 = self.decoder_block_4(torch.cat([dec_3, torch.nn.functional.upsample_bilinear(enc_1, dec_3.size()[2:])], 1))

    # apply final block
    final = self.final_block(dec_4)

    # upsample to image size
    output = torch.nn.functional.upsample_bilinear(final, [512, 1024])
    return output
