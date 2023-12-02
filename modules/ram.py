import torch
import torch.nn as nn
import torch.nn.functional as F


class RAM(nn.Module):
    def __init__(self, reduction):
        super(RAM, self).__init__()
        self.reduction = reduction
        in_channels = 2048
        channel = 2048

        self.conv1 = nn.Conv2d(in_channels, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1)
        )

        self.spatial_attention = nn.Conv2d(channel, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, input):
        u = F.relu(self.conv1(input))
        u = self.conv2(u)

        x = self.channel_attention(u).sigmoid()

        y = self.spatial_attention(u)

        z = torch.add(x, y)
        z = torch.mul(u, z)
        z = torch.add(input, z)

        return z
