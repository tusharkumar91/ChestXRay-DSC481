import torch
import torch.utils.data as data
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2

class ChestXRayDataset(data.Dataset):

    def __init__(self, data_root, split, labels, valid_img_types = ('jpeg', 'jpg', 'png')):
        r"""
        Data loader for chest xray data.
        Returns list of image, label tuples for given split
        """
        self.data_root = data_root
        self.split = split
        self.valid_labels = [label.lower() for label in labels]
        self.valid_img_types = valid_img_types
        self.images, self.labels = self.get_image_data()

    def get_image_data(self):
        assert self.split in os.listdir(self.data_root), '{} directory not in root path'.format(self.split)
        img_dirs = os.listdir(self.data_root)
        imgs = []
        labels = []
        for img_dir in os.listdir(os.path.join(self.data_root, self.split)):
            if img_dir.lower() in self.valid_labels:
                for img_file in os.listdir(os.path.join(self.data_root, self.split, img_dir)):
                    ext = img_file.split('.')[-1]
                    if ext in self.valid_img_types:
                        imgs.append(os.path.join(self.data_root, self.split, img_dir, img_file))
                        labels.append(self.valid_labels.index(img_dir.lower()))
        return imgs, labels


    def __getitem__(self, index):
        image = cv2.imread(self.images[index], 3).astype(np.float32) / 255
        image = cv2.resize(image, (224, 224))
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
    dataset = ChestXRayDataset('data/chest_xray', 'train', ['normal', 'pneumonia'])
    img = dataset[0][0]
    plt.imshow(dataset[0][0].numpy().transpose(1, 2, 0))
    plt.show()
