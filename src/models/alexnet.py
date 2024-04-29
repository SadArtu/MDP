import torch.nn as nn
from torchvision import models


def alexnet(pretrained=True, num_classes=1000):
    """
    Load the AlexNet model. Set pretrained=True to use the pretrained version.
    """
    model = models.alexnet(pretrained=pretrained)
    if pretrained:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return model
