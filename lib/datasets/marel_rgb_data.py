import torch
import torchvision
import random

import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset, DataLoader




class RGB_Fillets(Dataset):
    ''' Marel RGB_Fillet Dataset'''

    def __init__(self, dataset_path, mode='train', cropsize=[512,512]):
        
        self.images = None
        self.labels = None
        self.dataset_path = dataset_path
        self.mode = mode
        self.cropsize = cropsize

        def load_data_paths(dataset_path, mode):
            dataset_pairs = pd.read_csv(f'{dataset_path}/{mode}.txt', sep=',', header=None)
            self.images = dataset_pairs[0].tolist()
            self.labels = dataset_pairs[1].tolist()
        
        load_data_paths(dataset_path, mode)

    def transform(self, image, label):
        # Resize
        resize = torchvision.transforms.Resize(size=(self.cropsize[0], self.cropsize[1]))
        image = resize(image)
        label = resize(label)

        # Random crop
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            image, output_size=(self.cropsize[0], self.cropsize[1]))
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # Transform to tensor
        image = TF.to_tensor(image).float()
        label = TF.to_tensor(label).long()
        return image, label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = Image.open(f'{self.dataset_path}/{self.images[idx]}')
        labels = Image.open(f'{self.dataset_path}{self.labels[idx]}')

        return self.transform(img, labels)

def get_dataset(batch_size=20, mode='train', num_workers=0, shuffle=True, drop_last=True, cropsize=[512,512]):
    rgb_fillets = RGB_Fillets('./datasets/marel/', mode=mode, cropsize)
    dataloader = DataLoader(rgb_fillets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return dataloader


if __name__ == "__main__":
    dataset = get_dataset()
    for i, x in enumerate(dataset):
        stuff = x[0].numpy()
