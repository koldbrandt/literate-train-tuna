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


from project12.dataset import  get_loaders, load_full_data
import utilities as ut
import proposals

from model import FinetuneResnet50


def train(model, optimizer, train_loader, val_loader, device, num_epochs=50, patience = 10):
   
        
    def loss_fun(output, target):
        return F.cross_entropy(output, target)
    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}
    for epoch in range(num_epochs):
        
        model.train()
        #For each epoch
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss = loss_fun(output, target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            #Compute how many were correctly classified
            predicted = F.softmax(output, dim=1).argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
        #Comput the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted =  F.softmax(output, dim=1).argmax(1)
            test_correct += (target==predicted).sum().cpu().item()
        out_dict['train_acc'].append(train_correct/len(train_loader.dataset))
        out_dict['test_acc'].append(test_correct/len(val_loader.dataset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%\t",
              f"Memory allocated: {torch.cuda.memory_allocated(device=device)/1e9:.1f} GB")
    # Early stopping 
#         if epoch > 10 and out_dict['test_acc'][-1] < out_dict['test_acc'][-2]:
#             patience -= 1
#             if patience == 0:
#                 print("Early stopping")
#                 break

    return out_dict


def main():

    data_path='/dtu/datasets1/02514/data_wastedetection'
    anns_file_path = data_path + '/' + 'annotations.json'
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())
        
    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']   

    # takes about 1 hour, run once
    # prepare_datasets(categories=categories, anns=anns, imgs=imgs)
    
    device = ut.get_device()

    train_images, train_labels, val_images, val_labels = load_full_data()
    train_loader, val_loader = get_loaders(train_images, train_labels, val_images, val_labels, batch_size=512)


    model = FinetuneResnet50(29)
    # model = ResNet(3,16, num_res_blocks=8)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    training_stats = train(model, optimizer, train_loader, val_loader,  device, 5)

    torch.save(model.state_dict(), 'model.pkl')
    
if __name__ == "__main__":
    main()