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
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from torch import utils

import random
from math import floor
import cv2
import pickle

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

def load_full_data(data_path = 'project12/data/'):
    print('[*] Loading dataset from', data_path)
    with open(data_path + 'train_images.pkl', 'rb') as f:
        train_images=pickle.load(f)
    with open(data_path + 'train_labels.pkl', 'rb') as f:
        train_labels=pickle.load(f)
    with open(data_path + 'val_images.pkl', 'rb') as f:
        val_images=pickle.load(f)
    with open(data_path + 'val_labels.pkl', 'rb') as f:
        val_labels=pickle.load(f)
        
    return train_images, train_labels, val_images, val_labels
    
def get_loaders(train_images, train_labels, val_images, val_labels, batch_size=16):
    size = 112 


    train_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((size, size)), 
                                        transforms.ToTensor(), 
                                        ])
    val_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((size, size)), 
                                        transforms.ToTensor(), 
                                        ])

    trainset = WasteInWild(train_images, train_labels, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    
    valset = WasteInWild(val_images, val_labels, transform=val_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=3)

    return train_loader, val_loader