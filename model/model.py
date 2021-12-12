import torch
import torch.nn as nn
import torchvision
from torchvision import models
class ResNet(nn.Module):
    def __init__(self, inputchannl = 3, outputnode = 360):
        super(ResNet, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        # 修改第一层卷积层
        if inputchannl != 3:
            self.resnet.conv1 = nn.Conv2d(inputchannl, 64, kernel_size=7, stride=2, padding=3,
                                        bias=False)
        # 修改最后一层全连接层c
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, outputnode)

    def forward(self, x):
        y = self.resnet(x)
        return y