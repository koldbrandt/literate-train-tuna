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
import proposals

from model import FinetuneResnet50

def Validation(model, test_loader, criterion, device, test_im_ids):
    # set model to evaluation mode
    model.eval()
    # initialize variables for storing results
    test_correct = 0
    test_loss = []

    for id_img in test_im_ids:
        img, bbox_gt, labels = load_image_data(id_img=id_img)


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


# training loop with trainset and testset
# return dict with accuracy and loss
def train(model, optimizer, train_ids, test_ids, device, num_epochs=50, patience = 10):
    def loss_fun(output, target):
        return F.cross_entropy(output, target)
    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}
    for epoch in range(num_epochs):
        print(epoch)
        for id_img in train_ids:
            img, bbox_gt, labels = dataset.load_image_data(id_img=id_img)
            train_images, train_labels = proposals.get_proposals(img, bbox_gt, labels, IoU_threshold=0.5)
            train_loader = dataset.get_data(train_images, train_labels)
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
    #         #Comput the test accuracy
    #         test_loss = []
    #         test_correct = 0
    #         model.eval()
    #         for data, target in test_loader:
    #             data, target = data.to(device), target.to(device)
    #             with torch.no_grad():
    #                 output = model(data)
    #             test_loss.append(loss_fun(output, target).cpu().item())
    #             predicted =  F.softmax(output, dim=1).argmax(1)
    #             test_correct += (target==predicted).sum().cpu().item()
            out_dict['train_acc'].append(train_correct/len(train_loader.dataset))
    #         out_dict['test_acc'].append(test_correct/len(test_loader.dataset))
            out_dict['train_loss'].append(np.mean(train_loss))
    #         out_dict['test_loss'].append(np.mean(test_loss))
    #         print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
    #               f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%\t",
    #               f"Memory allocated: {torch.cuda.memory_allocated(device=device)/1e9:.1f} GB")
            print(f"Loss train: {np.mean(train_loss):.3f}\t",f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%")
        # Early stopping 
#         if epoch > 10 and out_dict['test_acc'][-1] < out_dict['test_acc'][-2]:
#             patience -= 1
#             if patience == 0:
#                 print("Early stopping")
#                 break

    return out_dict












def main():
    device = ut.get_device()

    model = FinetuneResnet50(29)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    train_ids = [0,1,2]
    training_stats = train(model, optimizer, train_ids, train_ids, device, 5)

    # ds, categories, anns, imgs, nr_super_cats, nr_cats, nr_annotations, nr_images, super_cat_ids, super_cat_names, cat_names= dataset.get_data()
    # print('Number of super categories:', nr_super_cats)
    # print('Number of categories:', nr_cats)
    # print('Number of annotations:', nr_annotations)
    # print('Number of images:', nr_images)
    # ut.show_dataset_stats(ds,categories, nr_cats,nr_super_cats,cat_names, super_cat_names, anns, super_cat_ids)
    # ut.dataset_graph(categories)

if __name__ == "__main__":
    main()