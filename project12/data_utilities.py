#matplotlib inline
import glob
from math import floor
import os

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from project12.proposals import get_proposals, load_image_data
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

def prepare_datasets(categories, anns, imgs, data_path = '', seed = 4):
    
    random.seed(seed)
    fullIdList = list(range(len(imgs)))
    random.shuffle(fullIdList)

    train_ids = fullIdList[0:floor(len(imgs)*0.6)]
    val_ids = fullIdList[floor(len(imgs)*0.6):(floor(len(imgs)*0.6)+floor(len(imgs)*0.2))]
    test_ids = fullIdList[floor(len(imgs)*0.8):(floor(len(imgs)*0.8)+floor(len(imgs)*0.2))]
    print(f'train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}')
    
    train_ims = []
    train_labs = []
    for id_im in train_ids:
        im, bbox_gt, labels = load_image_data(id_im=id_im, categories=categories, anns=anns, imgs=imgs)
        train_images, train_labels = get_proposals(im, bbox_gt, labels, id_im, IoU_threshold=0.5)
        train_ims += train_images
        train_labs += train_labels
    print(f'train len: {len(train_ims)}')
    
    val_ims = []
    val_labs = []
    for id_im in val_ids:
        im, bbox_gt, labels = load_image_data(id_im=id_im, categories=categories, anns=anns, imgs=imgs)
        val_images, val_labels = get_proposals(im, bbox_gt, labels, id_im, IoU_threshold=0.5)
        val_ims += val_images
        val_labs += val_labels
    print(f'val len: {len(val_ims)}')
    
    print('[*] Dataset generated! Saving labels to', data_path)
    with open(data_path + 'train_labels.pkl', 'wb') as f:
        pickle.dump(train_labs, f)
    with open(data_path + 'train_images.pkl', 'wb') as f:
        pickle.dump(train_ims, f)  
    with open(data_path + 'val_labels.pkl', 'wb') as f:
        pickle.dump(val_labs, f)
    with open(data_path + 'val_images.pkl', 'wb') as f:
        pickle.dump(val_ims, f)   
    