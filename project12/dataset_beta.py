#matplotlib inline
import glob
from math import floor
import os

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from torch import utils

import random
from math import floor

class WasteInWild(torch.utils.data.Dataset):
    def __init__(self,id_list, transform, data_path='/dtu/datasets1/02514/data_wastedetection'):
        'Initialization'
        self.transform = transform
        self.data_path = data_path
        anns_file_path = data_path + '/' + 'annotations.json'
        self.id_list = id_list
        with open(anns_file_path, 'r') as f:
            dataset = json.loads(f.read())

        self.categories = dataset['categories']
        self.anns = dataset['annotations']
        self.imgs = dataset['images']

    def __len__(self):
        'Returns the total number of samples'
        return len(self.id_list)

    def __getitem__(self, idx):
        id = self.id_list[idx]
        image_filepath = self.imgs[id]['file_name']
        image = Image.open(self.data_path + '/' + image_filepath)
        bbox = []
        labels = []
        for d in self.anns:
            if d['image_id'] == id:
                labels.append(d['category_id'])
                bbox.append(d['bbox'])
        bbox = torch.as_tensor(bbox)
        labels = torch.as_tensor(labels)
        target = {}
        target["boxes"] = bbox
        target["labels"] = labels
        X = self.transform(image)
        return X, target
    
def collate_fn(batch):
    return tuple(zip(*batch))

def get_data(batch_size=16):
    size = 256 

    fullIdList = list(range(1500))
    random.shuffle(fullIdList)

    train = fullIdList[0:floor(1500*0.6)]

    val = fullIdList[floor(1500*0.6):(floor(1500*0.6)+floor(1500*0.2))]

    test = fullIdList[floor(1500*0.8):(floor(1500*0.8)+floor(1500*0.2))]

    train_transform = transforms.Compose([
                                        transforms.Resize((size, size)), 
                                        transforms.ToTensor(), 
                                        ])

    test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                        transforms.ToTensor()
                                        ])

    trainset = WasteInWild(id_list=train, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3, collate_fn=collate_fn)
    testset = WasteInWild(id_list=test, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3, collate_fn=collate_fn)

    return train_loader, test_loader