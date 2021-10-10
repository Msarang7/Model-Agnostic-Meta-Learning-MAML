# create an indexed csv file with paths to images and corresponding labels

import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import numpy as np
import matplotlib.pyplot as plt

def get_indexed_data(root_dir):

    img_paths = []
    img_labels = []
    img_labels_dict = {}


    i = 0
    for (root, dirs, files) in os.walk(root_dir):

        for f in files :
            if f.endswith('png'):
                img_path = root + str('/') + f
                img_path = img_path.replace("\\" , "/")
                img_paths.append(img_path)
                if root not in img_labels_dict :
                    img_labels_dict[root] = len(img_labels_dict) # indexing classes
                img_labels.append(img_labels_dict[root])


    df = pd.DataFrame()
    df['paths'] = img_paths
    df['labels'] = img_labels
    df.to_csv('data.csv')


if 'data.csv' not in os.listdir():
    get_indexed_data()



class Omniglot(Dataset):

    def __init__(self, h, w, csv_file = None):

        self.data = pd.read_csv(csv_file)
        self.h = h
        self.w = w

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = self.data.iloc[idx,1]
        label = self.data.iloc[idx, 2]
        image = Image.open(img_path).convert('L') # L is mode for reading black and white 8 bit pixels
        image = image.resize((self.h, self.w))
        image = np.reshape(image, (self.h, self.w, 1))
        image = np.transpose(image, (2,0,1))
        image = image/255.0

        return image, label


def show_batch(batch):

    img_batch = batch[0]
    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy()[::-1].transpose(1,2,0))
    plt.title('Batch from DataLoader')

if __name__ == "__main__" :

    data = Omniglot(csv_file = 'data.csv', h = 28, w = 28)

    dataloader = DataLoader(data, batch_size = 4, shuffle = True)

    print(" batch index, image size and Label size below : ")
    for i, (image, label) in enumerate(dataloader):

        print(i, image.size(), label.size())

        # observe 4th batch
        if i == 3:
            plt.figure()
            show_batch((image, label))
            plt.axis('off')
            plt.ioff()
            plt.show()
            break












