#matplotlib inline
import glob
from math import floor
import os

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def load_image_data(id_img, basewidth=400, data_path='/dtu/datasets1/02514/data_wastedetection'):
    
    anns_file_path = data_path + '/' + 'annotations.json'

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    cat_names = []
    super_cat_names = []
    super_cat_ids = {}
    super_cat_last_name = ''
    nr_super_cats = 0
    for cat_it in categories:
        cat_names.append(cat_it['name'])
        super_cat_name = cat_it['supercategory']
        # Adding new supercat
        if super_cat_name != super_cat_last_name:
            super_cat_names.append(super_cat_name)
            super_cat_ids[super_cat_name] = nr_super_cats
            super_cat_last_name = super_cat_name
            nr_super_cats += 1
    
    image_filepath = imgs[id_img]['file_name']
    image = Image.open(data_path + '/' + image_filepath)
    bbox = []
    labels = []
    for d in anns:
        if d['image_id'] == id_img:
            cat = [cat for cat in categories if cat['id'] == d['category_id']][0]
            label = super_cat_ids[cat['supercategory']]
            labels.append(label)
            bbox.append(d['bbox'])
    bbox = torch.as_tensor(bbox)
    wpercent = (basewidth/float(image.size[0]))
    hsize = int((float(image.size[1])*float(wpercent)))
    image = image.resize((basewidth,hsize), Image.ANTIALIAS)
    bbox_gt = (bbox*wpercent).numpy().astype(int)
    
    return np.array(image), bbox_gt, labels


