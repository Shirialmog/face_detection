import os
import torch
import torchvision.datasets
from skimage import io, transform
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from PIL import Image

mtcnn = MTCNN()



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

composed_transforms = transforms.Compose([Rescale((224,224))])


class VGGDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the current exp csv file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df.reset_index()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.loc[idx, 'image_path']
        #image = torchvision.io.read_image(img_name)
        image = Image.open(img_name)
        try:
            image = mtcnn(image)
        except:
            print (img_name)
        if self.transform:
            image = composed_transforms(image)
        img_class = self.df.loc[idx, 'class']
        sample = {'image': image, 'img_class':img_class}

        return sample

def get_datasets(root_dir, df_path):
    df=pd.read_csv(df_path)
    train_dir = f'{root_dir}/train'
    train_dataset = VGGDataset(df[df['set']=='train'], train_dir, True)
    val_dataset = VGGDataset(df[df['set']=='val'], train_dir, True)

    return train_dataset , val_dataset





def loader(path):
    img = torchvision.io.read_image(path)
    #img = img.resize((256,256))
    return img