import torch.nn as nn
import torch


class ConvNet(nn.Module):

    def __init__(self, in_channels=3, hid_channels=64, out_channels=64, global_pool=5):
        super().__init__()
        self.global_pool = global_pool
        self.layers = nn.Sequential(
            self.conv_block(in_channels,  hid_channels),
            self.conv_block(hid_channels, hid_channels),
            self.conv_block(hid_channels, hid_channels),
            self.conv_block(hid_channels, out_channels)
        )

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.layers(x)
        if self.global_pool:
            x = nn.MaxPool2d(self.global_pool)(x) # In feat, the global maxpool is added
        x = torch.flatten(x, 1)
        return x

