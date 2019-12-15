import json
import torch
import torch.utils.data as data
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import os
from utils import augmentation
import random
import cv2

class ChestXRayDataset(data.Dataset):

    def __init__(self, data_root, data_json_path, split, labels):
        r"""
        Data loader for chest xray data.
        Returns list of image, label tuples for given split
        """
        self.data_json_path = data_json_path
        self.data_root = data_root
        self.split = split
        self.valid_labels = [label.lower() for label in labels]
        #self.valid_img_types = valid_img_types
        self.images, self.labels = self.get_image_data()


    def get_image_data(self):
        data = json.load(open(self.data_json_path))
        assert self.split in data, '{} directory not valid split'.format(self.split)
        imgs = []
        labels = []
        for label in data[self.split].keys():
            for img_path in data[self.split][label]:
                dups = 1
                if label == 'normal' and self.split == 'train': 
                    dups = 3
                for _ in range(dups):
                    labels.append(self.valid_labels.index(label))
                    imgs.append(os.path.join(self.data_root, img_path))
        return imgs, labels


    def __getitem__(self, index):
        image = cv2.imread(self.images[index], 1).astype(np.float32) / 255
        image = cv2.resize(image, (256, 256))
        image = image[17:241, 17:241, :]
        #image = augmentation.random_blur(image)
        #image = augmentation.random_flip(image)
        image = augmentation.rotation(image, [-10, 10])
        image = augmentation.random_intensity(image, 0.1)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
            ])
        image = transform(image)
        label = self.labels[index]
        return image, label


    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    dataset = ChestXRayDataset('data', 'data/original_data.json', split='train', labels=['normal', 'pneumonia'])
    img, label = dataset[0]
    #plt.imshow(dataset[0][0].numpy().transpose(1, 2, 0))
    #plt.show()
