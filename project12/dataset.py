#matplotlib inline
import glob
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

class WasteInWild(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='dtu/datasets1/02514/data_wastedetection'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        anns_file_path = data_path + '/' + 'annotations.json'

        with open(anns_file_path, 'r') as f:
            dataset = json.loads(f.read())

        self.categories = dataset['categories']
        self.anns = dataset['annotations']
        self.imgs = dataset['images']

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getparams__(self):
        categories = self.categories
        anns = self.anns
        imgs = self.imgs
        nr_cats = len(categories)
        nr_annotations = len(anns)
        nr_images = len(imgs)
        # Load categories and super categories
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
        return categories, anns, imgs, nr_cats, nr_super_cats, nr_annotations, nr_images, super_cat_names, cat_names, super_cat_ids 
    
def get_data(batch_size=16):
    size = 256  
    trainset = WasteInWild(train=True)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    testset = WasteInWild(train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

    return train_loader, test_loader