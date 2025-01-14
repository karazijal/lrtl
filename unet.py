from detectron2.structures import ImageList
from torch import nn


""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [torch.div(diffX, 2, rounding_mode='floor'),
                        diffX - torch.div(diffX, 2, rounding_mode='floor'),
                        torch.div(diffY, 2, rounding_mode='floor'),
                        diffY - torch.div(diffY, 2, rounding_mode='floor')])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, cfg, bilinear = True, device=torch.device('cuda')):
        super(UNet, self).__init__()
        n_channels = 3
        out_channels = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.n_channels = n_channels
        # self.bilinear = cfg.UNSUPVIDSEG.UNET_BILINEAR
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        self.device = device

        pixel_mean = cfg.MODEL.PIXEL_MEAN,
        pixel_std = cfg.MODEL.PIXEL_STD,
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

        print('pixel_mean:', self.pixel_mean)
        print('pixel_std:', self.pixel_std)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)
        self.to(device)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

    def forward_base(self, batched_inputs, keys, get_train=False, get_eval=False, raw_sem_seg=None):
        # key = keys[0]
        # images = [x[key].to(self.device) for x in batched_inputs]
        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # images = ImageList.from_tensors(images, self.size_divisibility)
        # return [{'sem_seg': x} for x in self.forward(images.tensor/255.)]

        images = (batched_inputs[keys[0]] - self.pixel_mean) / self.pixel_std

        r = self(images)
        ret = []
        for i in range(images.shape[0]):
            ret.append({
                'sem_seg': r[i],
            })
        return ret