import torch
from dataset import get_datasets
from model import vgg_model
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

def train_vgg(epochs):
    writer = SummaryWriter()
    train_dataset, test_dataset = get_datasets()
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    model = vgg_model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    acc=torchmetrics.Accuracy()
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        model.train()
        running_loss = 0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            if i==10:
                train_acc = acc.compute()
                writer.add_scalar('Accuracy/train', train_acc, i)
                print(f"acc: {train_acc}")
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())
            #probabilities = torch.nn.functional.softmax(outputs, dim=-1)
            thresh_probabilities = torch.argmax(outputs, dim= 1)
            train_acc = acc(thresh_probabilities, labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:  # print every 2000 mini-batches
                print(f'[epoch: {epoch + 1}/{epochs},step: {i + 1:5d}/{len(train_dataloader)}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        #     batch_acc = torch.sum(thresh_probabilities == labels)/len(inputs)
        #     train_acc += batch_acc
        # train_acc = train_acc/len(train_dataloader)
        # print (f'acc epoch {epoch}: {train_acc}')

        model.eval()
        running_val_loss = 0
        val_batch_acc=0
        if epoch ==5:
            a=1
        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            outputs = model(inputs.float())
            #probabilities = torch.nn.functional.softmax(outputs, dim=-1)
            thresh_probabilities =  torch.argmax(outputs, dim= 1)

            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            batch_acc = torch.sum(thresh_probabilities == labels) / len(inputs)
            val_batch_acc += batch_acc

        print (f'Val loss epoch {epoch}: {running_val_loss}')

        val_acc = val_batch_acc / len(test_dataloader)
        print(f'acc epoch {epoch}: {val_acc}')


    print('Finished Training')

if __name__ == '__main__':
    train_vgg(epochs=10)