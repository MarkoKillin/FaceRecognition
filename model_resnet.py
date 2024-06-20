import torch
import torch.nn as nn
from torchvision import models


class FaceRecognitionResNet(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionResNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        # Optional: Freeze earlier layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer3.parameters():
            param.requires_grad = True

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)



def get_resnet_model(num_classes):
    return FaceRecognitionResNet(num_classes)
