import argparse
import torch
import torch.nn as nn
import numpy as np
from chest_xray_data_loader import ChestXRayDataset
import os
from torch.utils.data import Dataset, DataLoader
from models.resnet_xray_model import ResnetXRayClassificationModel
from utils import metrics
import time

import warnings
# TODO remove this check 
warnings.filterwarnings("ignore")

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(model):
    val_dataset = ChestXRayDataset('data/chest_xray', 'val', ['normal', 'pneumonia'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        gt = []
        pred = []
        print(len(val_loader))
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            for i in range(0, len(labels)):
                gt.append(labels[i].cpu().numpy().item())
                if outputs[i, 1] > outputs[i, 0]:
                    pred.append(1)
                else:
                    pred.append(0)
        print(pred)
        split_accuracy, split_precision, split_recall = metrics.get_performance_metrics(gt, pred)
        print('Accuracy : {} | Precision : {} | Recall : {}'.format(
            split_accuracy, split_precision, split_recall))
    return split_accuracy, split_precision, split_recall

def train(args):
    train_dataset = ChestXRayDataset('data/chest_xray', 'train', ['normal', 'pneumonia'])
    model = ResnetXRayClassificationModel()
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Define model, Loss, and optimizer
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    train_params = [{'params': model.get_1x_lr_params(), 'lr': 1E-3},
                    {'params': model.get_10x_lr_params(), 'lr': 0}]
    optimizer = torch.optim.SGD(model.fc1.parameters(), lr=1E-3, momentum=0.9, weight_decay=1E-2)
    best_acc = -np.inf
    for epoch in range(args.num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            label = labels.to(device)
            output = model(images)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if i % args.log_step == 0:
                print('Epoch {}/{} : Step {}/{}, Loss: {:.4f}'
                      .format(epoch+1, args.num_epochs, i+1, len(train_loader), loss.item()))
        accuracy, precision, recall = validate(model)
        if accuracy > best_acc:
            print('Saving better model with accuracy : {}'.format(accuracy))
            torch.save(model, 'best_weights.pth')
            best_acc = accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_step', type=int, default=500)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()
    print(args)
train(args)
