import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, downsample=False):
    super(ResidualBlock, self).__init__()

    stride = 2 if downsample else 1

    # If batchnorm is applied, bias becomes redundant so set to false to remove redundant parameters
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

    if downsample:
      self.shortcut = nn.Sequential(
          nn.Conv2d(
              in_channels=in_channels,
              out_channels=out_channels,
              kernel_size=1,
              stride=stride,
              padding=0,
              bias=False
          ),
          nn.BatchNorm2d(num_features=out_channels)
      )
    else:
      self.shortcut = nn.Identity()

    self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    self.bn1 = nn.BatchNorm2d(num_features=out_channels)
    self.bn2 = nn.BatchNorm2d(num_features=out_channels)

  def forward(self, x):
    identity = self.shortcut(x)

    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)

    x += identity
    x = F.relu(x)

    return x


class ResNet18_CIFAR10(nn.Module):
  """
  Implementation of ResNet-18 module from scratch
  Based on the paper: Deep Residual Learning for Image Recognition

  Since CIFAR-10 Images are already 32x32, we skip the initial 7x7 conv and maxpooling layers to begin

  Just add one conv layer to increase channels from 3 to 64 before first res layer
  """
  def __init__(self, in_channels):
    super(ResNet18_CIFAR10, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)

    self.r1 = nn.Sequential(
        ResidualBlock(in_channels=64, out_channels=64),
        ResidualBlock(in_channels=64, out_channels=64)
    )

    self.r2 = nn.Sequential(
        ResidualBlock(in_channels=64, out_channels=128, downsample=True),
        ResidualBlock(in_channels=128, out_channels=128)
    )

    self.r3 = nn.Sequential(
        ResidualBlock(in_channels=128, out_channels=256, downsample=True),
        ResidualBlock(in_channels=256, out_channels=256)
    )

    self.r4 = nn.Sequential(
        ResidualBlock(in_channels=256, out_channels=512, downsample=True),
        ResidualBlock(in_channels=512, out_channels=512)
    )

    self.layers = nn.Sequential(
        self.r1,
        self.r2,
        self.r3,
        self.r4
    )

    self.gap = nn.AdaptiveAvgPool2d((1, 1))

    self.fc = nn.Linear(in_features=512, out_features=10)

    self.apply(self._init_weights)

  def _init_weights(self, layer):
    if isinstance(layer, nn.Conv2d):
      nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
      if layer.bias is not None:
        nn.init.zeros_(layer.bias)

    elif isinstance(layer, nn.Linear):
      nn.init.trunc_normal_(layer.weight, std=0.02)
      if layer.bias is not None:
        nn.init.zeros_(layer.bias)

  def forward(self, x):
    x = self.conv1(x)
    x = self.layers(x)
    x = self.gap(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x

