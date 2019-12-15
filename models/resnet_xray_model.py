import torch
import torch.nn as nn
import torchvision

class ResnetXRayClassificationModel(nn.Module):
    def __init__(self, output_size=2):
        super(ResnetXRayClassificationModel, self).__init__()
        resnet = torchvision.models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(512, output_size)


    def forward(self, x):
        features = self.resnet(x)
        features = features.reshape(features.size(0), -1)
        return self.fc1(features)