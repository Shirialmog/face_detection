import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchsummary import summary
from argparse import ArgumentParser

cos = torch.nn.CosineSimilarity()
mtcnn = MTCNN(image_size=224)

from torch import nn
vgg_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
#resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)

def get_vgg_model(num_classes):
    model = vgg_model
    #model.fc = nn.Linear(in_features = 2048, out_features=num_classes+1, bias = True)
    model.classifier[6] = nn.Linear(in_features = 4096, out_features=num_classes+1, bias = True)
    return model


def load_data(data_dir, instance_num_per_class):
    data_paths_list = []
    for c in os.listdir(data_dir):
        class_path = os.path.join(data_dir, c)
        imgs= os.listdir(class_path)
        for img in imgs[:instance_num_per_class]:
            im_path = os.path.join(class_path, img)
            data_paths_list.append(im_path)

    return data_paths_list


# get repr. from model
def get_rdm(dataset_paths_list, model):
    rdm = np.zeros((len(dataset_paths_list), len(dataset_paths_list)))
    for i, im_path in tqdm(enumerate(dataset_paths_list)):
        try:
            img = Image.open(im_path)
            img = mtcnn(img)
            img_embedding = model(img.unsqueeze(0).float())
            for j, second_im_path in enumerate(dataset_paths_list):
                if second_im_path == im_path:
                    second_img_embedding = img_embedding
                else:
                    second_img = Image.open(second_im_path)
                    second_img = mtcnn(second_img)
                    second_img_embedding = model(second_img.unsqueeze(0).float())
                rdm[i, j] = cos(img_embedding, second_img_embedding)
        except:
            print(f'error calculating similarity for: {im_path}')

    return rdm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-data_dir", "--data_dir", dest="data_dir", help="folder with data")

    args = parser.parse_args()
    data_paths_list = load_data(args.data_dir, instance_num_per_class=1)
    model = get_vgg_model(num_classes=len(data_paths_list))
    rdm = get_rdm(data_paths_list, model)
    plt.imshow(rdm)
    plt.show()
