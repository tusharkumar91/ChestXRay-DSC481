import argparse
import torch
import torch.nn as nn
import numpy as np
from chest_xray_data_loader import ChestXRayDataset
import os
import pickle
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from models.resnet_xray_model import ResnetXRayClassificationModel
from models.build_model import *
from utils import metrics
import time

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(model):
    val_dataset = ChestXRayDataset('data', 'data/data_split_70_15_15.json', split='val', labels=['normal', 'pneumonia'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Define model, Loss, and optimizer
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    val_loss = 0.0
    with torch.no_grad():
        gt = []
        pred = []
        for i, (images, labels) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            for i in range(0, len(labels)):
                gt.append(labels[i].cpu().numpy().item())
                if outputs[i, 1] > outputs[i, 0]:
                    pred.append(1)
                else:
                    pred.append(0)
        split_accuracy, split_precision, split_recall = metrics.get_performance_metrics(gt, pred)
        print('Val Loss : {} | Accuracy : {} | Precision : {} | Recall : {}'.format(
            val_loss/len(val_loader), split_accuracy, split_precision, split_recall))
        return val_loss/len(val_loader), split_accuracy, split_precision, split_recall

def train(args):
    train_dataset = ChestXRayDataset('data', 'data/data_split_70_15_15.json', split='train', labels=['normal', 'pneumonia'])
    
    # Resnet Model
    model = ResnetXRayClassificationModel()
    
    # VGG 19 Model
    #model = models.vgg19(pretrained=True)
    #features = list(model.classifier.children())[:-1]
    #features.extend([torch.nn.Linear(4096, 2)])
    #model.classifier = torch.nn.Sequential(*features)

    # UNet model
    #model =  UNet(3,1)
    #model.load_state_dict(torch.load('models/cp_bce_flip_lr_04_no_rot57_0.027226628065109254.pth.tar'), strict=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    
    # Define model, Loss, and optimizer
    model.to(device)
    val_losses = []
    train_losses = []
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)
    best_val_acc = -np.inf
    best_val_recall = -np.inf
    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        print('Epoch : {}'.format(epoch+1))
        model.train()
        t1 = time.time()
        train_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            # scheduler(optimizer, i, epoch, best_F1)
            optimizer.zero_grad()
            # mini-batch
            images = images.to(device)
            label = labels.to(device)
            output = model(images)
            loss = criterion(output, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        val_loss, acc, prec, recall = validate(model)
        if acc > best_val_acc:
            torch.save(model.state_dict(), 'best_weights_resnet_oversample_70_15_15_intensity_0.1_rot10_crop_original_data_acc.pth')
            best_val_acc = acc
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), 'best_weights_resnet_oversample_70_15_15_intensity_0.1_rot10_crop_original_data_loss.pth')
            best_val_loss = val_loss
        if recall > best_val_recall and acc > 0.7:
            torch.save(model.state_dict(), 'best_weights_resnet18_oversasmple_70_15_15_intensity_0.1_rot10_recall.pth')
            best_val_recall = recall
        print('Best val acc : ', best_val_acc)
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss)
        #pickle.dump(train_losses, open('train_losses_pretrain_aug.pkl', 'wb'))
        #pickle.dump(val_losses, open('val_losses_pretrain_aug.pkl', 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_step', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    print(args)
    train(args)
