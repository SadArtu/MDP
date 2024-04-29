import torch.nn as nn
from torchvision import models


def resnet34(pretrained=True, num_classes=15):
    """
    Load the ResNet-34 model. Set pretrained=True to use the pretrained version.
    """
    model = models.resnet34(pretrained=pretrained)
    if pretrained:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model
