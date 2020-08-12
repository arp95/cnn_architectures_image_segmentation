# header files
import torch
import torch.nn as nn
import torchvision
import numpy as np


# model (pretrained_net = vgg16)
class SegNet(torch.nn.Module):

  def __init__(self, pretrained_net, num_classes=21):
    super(SegNet, self).__init__()

    # encoder 1, encoder 2, encoder 3, encoder 4, encoder 5
    self.encoder_block_1 = torch.nn.Sequential(*list(model.features.children())[:-38])
    self.encoder_block_2 = torch.nn.Sequential(*list(model.features.children())[-37:-31])
    self.encoder_block_3 = torch.nn.Sequential(*list(model.features.children())[-30:-21])
    self.encoder_block_4 = torch.nn.Sequential(*list(model.features.children())[-20:-11])
    self.encoder_block_5 = torch.nn.Sequential(*list(model.features.children())[-10:-1])

    # max-pool layer with return_indices as true
    self.max_pool_layer = torch.nn.Sequential(
        torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    )

    # decoder block 1
    self.decoder_block_1 = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(inplace=True),
        torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(inplace=True),
        torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(inplace=True)
    )

    # decoder block 2
    self.decoder_block_2 = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(inplace=True),
        torch.nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(512),
        torch.nn.ReLU(inplace=True),
        torch.nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(inplace=True)
    )

    # decoder block 3
    self.decoder_block_3 = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(inplace=True),
        torch.nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(inplace=True),
        torch.nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(inplace=True)
    )

    # decoder block 4
    self.decoder_block_4 = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(inplace=True),
        torch.nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace=True)
    )

    # decoder block 5
    self.decoder_block_5 = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace=True),
        torch.nn.ConvTranspose2d(64, num_classes, kernel_size=3, padding=1)
    )

  def forward(self, x):
    
    # apply encoder block 1
    enc_1 = self.encoder_block_1(x)
    enc_1, m_1 = self.max_pool_layer(enc_1)

    # apply encoder block 2
    enc_2 = self.encoder_block_2(enc_1)
    enc_2, m_2 = self.max_pool_layer(enc_2)

    # apply encoder block 3
    enc_3 = self.encoder_block_3(enc_2)
    enc_3, m_3 = self.max_pool_layer(enc_3)

    # apply encoder block 4
    enc_4 = self.encoder_block_4(enc_3)
    enc_4, m_4 = self.max_pool_layer(enc_4)

    # apply encoder block 5
    enc_5 = self.encoder_block_5(enc_4)
    enc_5, m_5 = self.max_pool_layer(enc_5)

    # apply decoder block 1
    dec_1 = self.decoder_block_1(torch.nn.functional.max_unpool2d(enc_5, m_5, kernel_size=2, stride=2, output_size=enc_4.size()))

    # apply decoder block 2
    dec_2 = self.decoder_block_2(torch.nn.functional.max_unpool2d(enc_4, m_4, kernel_size=2, stride=2, output_size=enc_3.size()))

    # apply decoder block 3
    dec_3 = self.decoder_block_3(torch.nn.functional.max_unpool2d(enc_3, m_3, kernel_size=2, stride=2, output_size=enc_2.size()))

    # apply decoder block 4
    dec_4 = self.decoder_block_4(torch.nn.functional.max_unpool2d(enc_2, m_2, kernel_size=2, stride=2, output_size=enc_1.size()))

    output = self.decoder_block_5(torch.nn.functional.max_unpool2d(dec_4, m_1, kernel_size=2, stride=2, output_size=x.size()))
    return output
