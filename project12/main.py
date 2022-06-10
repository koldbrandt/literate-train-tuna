import glob
import os
import sys

import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import dataset
import utilities as ut

# training loop with trainset and testset
# return dict with accuracy and loss
def training_loop(train_loader, test_loader, model, criterion, optimizer, epochs, device, patience = 10):
    # initialize dictionary for storing results
    out_dict = {'train_acc': [],
                'test_acc': [],
                'train_loss': [],
                'test_loss': []}
    # initialize variables for storing results
    train_correct = 0
    train_loss = []
    test_correct = 0
    test_loss = []
    # set model to training mode
    model.train()
    # for each epoch
    for epoch in range(epochs):
        # for each minibatch
        for minibatch_no, (data, target) in enumerate(train_loader):
            # move data to device
            data, target = data.to(device), target.to(device)
            # zero the gradients
            optimizer.zero_grad()
            # forward pass
            output = model(data)
            # compute loss
            loss = criterion(output, target)
            # backward pass
            loss.backward()
            # update weights
            optimizer.step()
            # compute accuracy
            predicted = F.softmax(output, dim=1).argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
            # store loss and accuracy
            train_loss.append(loss.item())
        # set model to evaluation mode
        model.eval()
        # for each minibatch
        for data, target in test_loader:
            # move data to device
            data, target = data.to(device), target.to(device)
            # forward pass
            output = model(data)
            # compute loss
            loss = criterion(output, target)
            # compute accuracy
            predicted = F.softmax(output, dim=1).argmax(1)
            test_correct += (target==predicted).sum().cpu().item()
            # store loss and accuracy
            test_loss.append(loss.item())
        # store results
        out_dict['train_acc'].append(train_correct/len(train_loader.dataset))
        out_dict['test_acc'].append(test_correct/len(test_loader.dataset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        # print results and ram usage
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%\t",
              f"Memory allocated: {torch.cuda.memory_allocated(device=device)/1e9:.1f} GB")

    
        # Early stopping with patience
        if epoch > 10 and np.mean(out_dict['test_acc'][-patience:]) < np.mean(out_dict['test_acc'][-patience-1:-1]):
            patience -= 1
            if patience == 0:
                print('Early stopping')
                break
        

    return out_dict



    

        






def main():
    device = ut.get_device()

    # ds, categories, anns, imgs, nr_super_cats, nr_cats, nr_annotations, nr_images, super_cat_ids, super_cat_names, cat_names= dataset.get_data()
    # print('Number of super categories:', nr_super_cats)
    # print('Number of categories:', nr_cats)
    # print('Number of annotations:', nr_annotations)
    # print('Number of images:', nr_images)
    # ut.show_dataset_stats(ds,categories, nr_cats,nr_super_cats,cat_names, super_cat_names, anns, super_cat_ids)
    # ut.dataset_graph(categories)

if __name__ == "__main__":
    main()