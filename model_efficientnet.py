import torch.nn as nn
import torchvision.models as models


class FaceRecognitionEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionEfficientNet, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.efficientnet(x)
        return x


def get_efficientnet_model(num_classes):
    return FaceRecognitionEfficientNet(num_classes)
