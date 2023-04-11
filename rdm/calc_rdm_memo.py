import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from numpy.linalg import norm
from PIL import Image

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from training.dataset_utils import Rescale

mtcnn = MTCNN(image_size=160)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
composed_transforms = transforms.Compose([Rescale((160, 160)), normalize])
cos = torch.nn.CosineSimilarity()


def get_resnet_model(pretrain='vggface2'):
    # options are: vggface2, casia-webface https://github.com/timesler/facenet-pytorch
    model = InceptionResnetV1(pretrained=pretrain).eval()
    return model


def load_data(data_dir, num_classes, instance_num_per_class):
    data_paths_list = []
    for c in os.listdir(data_dir)[:num_classes]:
        class_path = os.path.join(data_dir, c)
        imgs = os.listdir(class_path)
        for img in imgs[:instance_num_per_class]:
            im_path = os.path.join(class_path, img)
            data_paths_list.append(im_path)

    return data_paths_list


def get_embeddings(data, model, transform=False):
    embeddings = np.zeros((len(data), 512))
    for i, im_path in enumerate(data):
        img = Image.open(im_path)
        img = mtcnn(img)
        if transform:
            img = composed_transforms(img)

        img_embedding = model(img.unsqueeze(0).float())
        embeddings[i] = img_embedding.detach().numpy()

    return embeddings


def get_rdm(embeddings):
    rdm = np.zeros((len(embeddings), len(embeddings)))
    for i, first_em in enumerate(embeddings):
        for j, second_em in enumerate(embeddings):
            rdm[i, j] = np.dot(first_em, second_em) / (norm(first_em) * norm(second_em))
    return rdm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-data_dir", "--data_dir", dest="data_dir", help="folder with data")
    args = parser.parse_args()

    data_paths_list = load_data(args.data_dir, num_classes=10, instance_num_per_class=10)
    model = get_resnet_model()
    embeddings = get_embeddings(data_paths_list, model)
    rdm = get_rdm(embeddings)
    plt.imshow(rdm)
    plt.show()
