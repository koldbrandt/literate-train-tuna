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
import cv2

import random
from math import floor

class WasteInWild(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform):
        'Initialization'
        self.transform = transform
        self.images = images
        self.labels = labels

    def __len__(self):
        'Returns the total number of samples'
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        X = self.transform(image)
        y = self.labels[idx][0]
        return  X, y  

def get_data(train_images, train_labels, batch_size=16):
    size = 32 


    train_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((size, size)), 
                                        transforms.ToTensor(), 
                                        ])

    test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                        transforms.ToTensor()
                                        ])

    trainset = WasteInWild(train_images, train_labels, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)

    return train_loader



