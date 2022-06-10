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
import cv2

def selective_search(image):
    # return region proposals of selective searh over an image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return ss.process()

def calculate_IoU(bb1, bb2):
# calculate IoU(Intersection over Union) of 2 boxes 
# **IoU = Area of Overlap / Area of Union
# https://github.com/Hulkido/RCNN/blob/master/RCNN.ipynb

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    # if there is no overlap output 0 as intersection area is zero.
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # calculate Overlapping area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    union_area = bb1_area + bb2_area - intersection_area

    return intersection_area / union_area   

def load_image_data(id_im, categories, anns, imgs, basewidth=400, data_path='/dtu/datasets1/02514/data_wastedetection'):
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
        
    image_filepath = imgs[id_im]['file_name']
    image = Image.open(data_path + '/' + image_filepath)
    bbox = []
    labels = []
    for d in anns:
        if d['image_id'] == id_im:
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

def weirdbbox2bbox(x,y,w,h):
    x1, y1 = x, y
    x2, y2 = x+w, y+h
    return {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}

def get_proposals(image, bbox_gt, labels, IoU_threshold=0.5):
    obj_counter = 0
    bg_counter = 0
    train_images=[]
    train_labels=[]



    rects = selective_search(image)
    random.shuffle(rects)

    for (x, y, w, h) in rects:
        # apply padding
        best_iou = 0
        bbox_est = weirdbbox2bbox(x,y,w,h)
        for i, gt_bbox in enumerate(bbox_gt):
            [x,y,w,h]=gt_bbox
            gt_bbox = weirdbbox2bbox(x,y,w,h)
            iou = calculate_IoU(gt_bbox, bbox_est)

            if iou >= IoU_threshold: # if object(RoI > 0.5)
                if iou > best_iou:
                    best_iou = iou
                    best_i = i

        if best_iou > 0:
            obj_counter += 1
            cropped = image[bbox_est['y1']:bbox_est['y2'], bbox_est['x1']:bbox_est['x2'], :]
            train_images.append(cropped)
            train_labels.append([labels[best_i], bbox_gt[best_i], bbox_est])

        else:
            bg_counter+=1
            cropped = image[bbox_est['y1']:bbox_est['y2'], bbox_est['x1']:bbox_est['x2'], :]
            train_images.append(cropped)
            train_labels.append([28, np.array([1,1,1,1]), bbox_est])
    
    return train_images, train_labels
