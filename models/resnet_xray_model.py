import torch
import torch.nn as nn
import torchvision

class ResnetXRayClassificationModel(nn.Module):
    def __init__(self, output_size=2):
        super(ResnetXRayClassificationModel, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(512, output_size)


    def forward(self, x):
        features = self.resnet(x)
        features = features.reshape(features.size(0), -1)
        return self.fc1(features)

    def get_1x_lr_params(self):
        modules = [self.resnet]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.fc1]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
