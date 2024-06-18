import torch.nn as nn
from torchvision import models


class FaceRecognitionResNet(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


def get_resnet_model(num_classes):
    return FaceRecognitionResNet(num_classes)
