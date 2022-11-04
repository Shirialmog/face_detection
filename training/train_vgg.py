import torch
import pandas as pd
from face_detection.training.dataset import get_datasets
from face_detection.training.model import get_vgg_model
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from face_detection.training.config import TrainConfig
from face_detection.prepare_data.create_df import create_unique_df

def train_vgg(config, df_path):
    epochs = config.epochs
    writer = SummaryWriter()

    train_dataset, test_dataset = get_datasets(config.root_dir, df_path)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    model = get_vgg_model(config.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    acc=torchmetrics.Accuracy()
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss, train_acc = [], []
        for i, data in enumerate(train_dataloader, 0):
            inputs = data['image']
            labels = data['img_class']




            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)
            thresh_probabilities = torch.argmax(outputs, dim= 1)
            cur_train_acc = acc(thresh_probabilities, labels)

            loss = criterion(probabilities, labels)
            loss.backward()
            optimizer.step()


            # print statistics
            running_loss.append(loss.item())
            train_acc.append(cur_train_acc)
            if i % 10 == 0:  # print every x mini-batches
                print(f'[epoch: {epoch + 1}/{epochs},step: {i + 1:5d}/{len(train_dataloader)}] loss: {np.mean(running_loss):.3f}, acc: {np.mean(train_acc)}')
                writer.add_scalar('Accuracy/train', np.mean(np.array(train_acc)), i)




        model.eval()
        val_running_loss, val_acc = [], []
        for i, data in enumerate(test_dataloader, 0):
            inputs = data['image']
            labels = data['img_class']
            outputs = model(inputs.float())
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)
            thresh_probabilities =  torch.argmax(outputs, dim= 1)
            cur_val_acc = acc(thresh_probabilities, labels)

            loss = criterion(probabilities, labels)
            val_running_loss.append(loss.item())
            val_acc.append(cur_val_acc)

        print(f'[validation epoch: {epoch + 1}/{epochs}] loss: {np.mean(val_running_loss):.3f}, acc: {np.mean(val_acc)}')
        writer.add_scalar('Accuracy/val', np.mean(np.array(val_acc)), i)


    print('Finished Training')

if __name__ == '__main__':
    config = TrainConfig()
    num_classes = config.num_classes
    num_instances_per_class = config.train_num_instances_per_class
    df_path = f'{config.root_dir}/dataset_splits/exp_{config.exp_name}_{num_classes}_{num_instances_per_class}.csv'
    try:
        df = pd.read_csv(df_path)
    except:
        df = create_unique_df(config)
        df.to_csv(df_path)

    train_vgg(config, df_path)