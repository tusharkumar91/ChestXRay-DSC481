import argparse
import torch
import torch.nn as nn
import numpy as np
from chest_xray_data_loader import ChestXRayDataset
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from models.resnet_xray_model import ResnetXRayClassificationModel
from models.build_model import *
from utils import metrics
import time

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model):
    model = ResnetXRayClassificationModel()
    model.load_state_dict(torch.load('best_weights_resnet_oversample_70_15_15_intensity_0.1_rot10_crop_original_data_loss.pth'))
    test_dataset = ChestXRayDataset('data', 'data/original_data.json', split='test', labels=['normal', 'pneumonia'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    # Define model, Loss, and optimizer
    model.to(device)
    model.eval()
    with torch.no_grad():
        gt = []
        pred = []
        print(len(test_loader))
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            for i in range(0, len(labels)):
                gt.append(labels[i].cpu().numpy().item())
                if outputs[i, 1] > outputs[i, 0]:
                    pred.append(1)
                else:
                    pred.append(0)
        split_accuracy, split_precision, split_recall = metrics.get_performance_metrics(gt, pred)
        print('Accuracy : {} | Precision : {} | Recall : {}'.format(
            split_accuracy, split_precision, split_recall))
        return split_accuracy, split_precision, split_recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_step', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()
    print(args)
    evaluate(args)
