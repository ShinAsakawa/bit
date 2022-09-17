#https://gist.github.com/nikogamulin/7774e0e3988305a78fd73e1c4364aded
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self,
                 num_layers:int,
                 in_channels:int,
                 out_channels:int,
                 identity_downsample=None,
                 stride:int=1,
                 device:str="cuda" if torch.cuda.is_available() else "cpu",                 
                ):
        
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0).to(device=device)
        self.bn1 = nn.BatchNorm2d(out_channels).to(device=device)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels,
                                   out_channels,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=1).to(device=device)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=1).to(device=device)
        self.bn2 = nn.BatchNorm2d(out_channels).to(device=device)
        self.conv3 = nn.Conv2d(out_channels,
                               out_channels * self.expansion,
                               kernel_size=1,
                               stride=1,
                               padding=0).to(device=device)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion).to(device=device)
        self.relu = nn.ReLU().to(device=device)
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self,
                 num_layers:int,
                 block:int,
                 image_channels:int,
                 num_classes:int,
                 device:str="cuda" if torch.cuda.is_available() else "cpu",                 
                ):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3).to(device=device)
        self.bn1 = nn.BatchNorm2d(64).to(device=device)
        self.relu = nn.ReLU().to(device=device)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device=device)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers,
                                       block,
                                       layers[0],
                                       intermediate_channels=64,
                                       stride=1,
                                       device=self.device,
                                      )
        self.layer2 = self.make_layers(num_layers,
                                       block,
                                       layers[1],
                                       intermediate_channels=128,
                                       stride=2,
                                       device=self.device)
        self.layer3 = self.make_layers(num_layers,
                                       block,
                                       layers[2],
                                       intermediate_channels=256,
                                       stride=2,
                                       device=self.device)
        self.layer4 = self.make_layers(num_layers,
                                       block,
                                       layers[3],
                                       intermediate_channels=512,
                                       stride=2,
                                       device=self.device)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        self.fc = nn.Linear(512 * self.expansion, num_classes).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self,
                    num_layers,
                    block,
                    num_residual_blocks,
                    intermediate_channels,
                    stride,
                    device='cuda' if torch.cuda.is_available() else "cpu"):
        layers = []

        identity_downsample = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      intermediate_channels*self.expansion,
                      kernel_size=1,
                      stride=stride).to(device),
            nn.BatchNorm2d(intermediate_channels*self.expansion).to(device))
        layers.append(block(num_layers,
                            self.in_channels,
                            intermediate_channels,
                            identity_downsample,
                            stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers,
                                self.in_channels,
                                intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)


def ResNet18(img_channels=3,
             num_classes=1000,
             device='cuda' if torch.cuda.is_available() else "cpu"):
    return ResNet(num_layers=18,
                  block=Block,
                  image_channels=img_channels,
                  num_classes=num_classes,
                  device='cuda' if torch.cuda.is_available() else "cpu")

def ResNet34(img_channels=3,
             num_classes=1000,
             device='cuda' if torch.cuda.is_available() else "cpu"):
    return ResNet(num_layers=34,
                  block=Block,
                  image_channels=img_channels,
                  num_classes=num_classes,
                  device='cuda' if torch.cuda.is_available() else "cpu")


def ResNet50(img_channels=3,
             num_classes=1000,
             device='cuda' if torch.cuda.is_available() else "cpu"):
    return ResNet(num_layers=50,
                  block=Block,
                  image_channels=img_channels,
                  num_classes=num_classes,
                  device='cuda' if torch.cuda.is_available() else "cpu")

def ResNet101(img_channels=3, 
              num_classes=1000,
             device='cuda' if torch.cuda.is_available() else "cpu"):
    return ResNet(num_layers=101, 
                  bloc=Block, 
                  image_channels=img_channels, 
                  num_classes=num_classes,
                  device='cuda' if torch.cuda.is_available() else "cpu")


def ResNet152(img_channels=3, 
              num_classes=1000,
              device='cuda' if torch.cuda.is_available() else "cpu"):
    return ResNet(num_layers=152, 
                  block=Block, 
                  img_channels=img_channels, 
                  num_classes=num_classes,
                  device='cuda' if torch.cuda.is_available() else "cpu")

# def test():
#     net = ResNet18(img_channels=3, num_classes=1000)
#     y = net(torch.randn(4, 3, 224, 224)).to("cpu")
#     print(y.size())
#
# test()
