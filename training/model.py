import torch
vgg_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)