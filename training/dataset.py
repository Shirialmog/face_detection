import os
import torch
import torchvision.datasets
from skimage import io, transform
import numpy as np

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt





class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.shape[1:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(sample, (float(3),new_h, new_w))

        img = torch.from_numpy(img)
        img.type(torch.DoubleTensor)
        return img


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         image, labels = sample[0], sample[1]
#
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C x H x W
#         image = image.transpose((2, 0, 1))
#         return {'image': torch.from_numpy(image),
#                 'labels': torch.from_numpy(labels)}

composed_transforms = transforms.Compose([Rescale((64,64))])



def get_datasets():
    ROOT = r'C:\Users\shiri\Documents\Galit\Data\VGG-Face2\data_small'
    train_dataset = torchvision.datasets.DatasetFolder(root =  r'C:\Users\shiri\Documents\School\Galit\Data\VGG-Face2\data_small\train', loader = torchvision.io.read_image, extensions =  ['jpg'],transform= composed_transforms)
    test_dataset = torchvision.datasets.DatasetFolder(root = r'C:\Users\shiri\Documents\School\Galit\Data\VGG-Face2\data\test\small_test', loader = torchvision.io.read_image, extensions =  ['jpg'], transform= composed_transforms )
    return train_dataset , test_dataset

# def get_datasets():
    #ROOT = r'C:\Users\shiri\Documents\Galit\Data\LFW'
#     training_dataset = datasets.LFWPeople(root=ROOT, transform=ToTensor(), split= 'train')
#     val_dataset = datasets.LFWPeople(root = ROOT, transform=ToTensor(),split = 'test')
#     return training_dataset, val_dataset



def loader(path):
    img = torchvision.io.read_image(path)
    img = img.resize((256,256))
    return img