import torch
from torch import nn
vgg_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=False)


def get_vgg_model(num_classes):
    model = vgg_model
    model.classifier[6] = nn.Linear(4096, num_classes+1)
    return model