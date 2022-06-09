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