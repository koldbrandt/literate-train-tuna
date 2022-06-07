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
from model import FinetuneResnet50, Network, ResNet


def train(model, optimizer, train_loader, test_loader, device, num_epochs=50, patience = 10):
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
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
        #Comput the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted = output.argmax(1)
            test_correct += (target==predicted).sum().cpu().item()
        out_dict['train_acc'].append(train_correct/len(train_loader.dataset))
        out_dict['test_acc'].append(test_correct/len(test_loader.dataset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%\t",
              f"Memory allocated: {torch.cuda.memory_allocated(device=device)/1e9:.1f} GB")

        # Early stopping 
        if epoch > 10 and out_dict['test_acc'][-1] < out_dict['test_acc'][-2]:
            patience -= 1
            if patience == 0:
                print("Early stopping")
                break

    return out_dict



def main():
    train_data, test_data = dataset.get_data(64)

    device = ut.get_device()
    model = FinetuneResnet50(2)
    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    training_stats = train(model, optimizer, train_data, test_data, device, 100)
    
    ut.plot_training_stats(training_stats)

    torch.save(model.state_dict(), 'models/model.pt')
    ut.save_training_stats(training_stats, 'Resnet50-no-transfer.csv')


if __name__ == "__main__":
    main()

