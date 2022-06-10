#matplotlib inline
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ExifTags
#from pycocotools.coco import COCO
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import colorsys
import random
import pylab

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

def get_device():
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU")
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def show_dataset_stats(dataset, categories, nr_cats,nr_super_cats,cat_names, super_cat_names, anns, super_cat_ids):
    # Dataset statistics (shows the # of annotations per category)

    cat_histogram = np.zeros(nr_cats,dtype=int)
    for ann in anns:
        cat_histogram[ann['category_id']] += 1

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(5,15))

    # Convert to DataFrame
    df = pd.DataFrame({'Categories': cat_names, 'Number of annotations': cat_histogram})
    df = df.sort_values('Number of annotations', 0, False)

    # Plot the histogram
    # sns.set_color_codes("pastel")
    # sns.set(style="whitegrid")
    plot_1 = sns.barplot(x="Number of annotations", y="Categories", data=df,
                label="Total", color="b")

    # fig = plot_1.get_figure()
    # fig.savefig("output.png")

    #(shows the # of annotations per supercategory):
    cat_ids_2_supercat_ids = {}
    for cat in categories:
        cat_ids_2_supercat_ids[cat['id']] = super_cat_ids[cat['supercategory']]

    # Count annotations
    super_cat_histogram = np.zeros(nr_super_cats,dtype=int)
    for ann in anns:
        cat_id = ann['category_id']
        super_cat_histogram[cat_ids_2_supercat_ids[cat_id]] +=1
        
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(5,10))

    # Convert to DataFrame
    d ={'Super categories': super_cat_names, 'Number of annotations': super_cat_histogram}
    df = pd.DataFrame(d)
    df = df.sort_values('Number of annotations', 0, False)

    # sns.set_color_codes("pastel")
    # sns.set(style="whitegrid")
    plot_1 = sns.barplot(x="Number of annotations", y="Super categories", data=df,
                label="Total", color="b")
    plot_1.set_title('Annotations per super category',fontsize=20)  

    #Background stats:
    # Get scene cat names
    scene_cats = dataset['scene_categories']
    scene_name = []
    for scene_cat in scene_cats:
        scene_name.append(scene_cat['name'])

    nr_scenes = len(scene_cats)
    scene_cat_histogram = np.zeros(nr_scenes,dtype=int)

    for scene_ann in dataset['scene_annotations']:    
        scene_ann_ids = scene_ann['background_ids']
        for scene_ann_id in scene_ann_ids:
            if scene_ann_id<len(scene_cats):
                scene_cat_histogram[scene_ann_id]+=1

    # Convert to DataFrame
    df = pd.DataFrame({'scene_cats': scene_cats, 'nr_annotations': scene_cat_histogram})
    
    # Plot
    colors = ['white','black','gray', 'gold', 'red','green','lightskyblue']
    plt.pie(scene_cat_histogram, labels=scene_name, colors = colors, shadow=False, startangle=-120)
    
    plt.axis('equal')
    plt.show()

def dataset_graph(categories):
    #g = Digraph('G', filename='hello.gv')
    dot = Digraph('Dataset graph', filename='asd.gv')
    dot.attr(rankdir='LR', size='8,10')

    for cat_it in categories:
        dot.node(cat_it['name'])
        if cat_it['name']==cat_it['supercategory']:
            dot.node(cat_it['supercategory'])
        else:
            dot.edge(cat_it['supercategory'], cat_it['name'])
    dot




def nms(boxes, threshold):
    """
    Non-maximum suppression.
    :param boxes: np.array of bounding boxes (x1, y1, x2, y2)
    :param threshold: overlap threshold
    :return: list of indices of bounding boxes to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(areas)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep

    # Uncomment next line to print pdf
    #dot.view()

# def annotated_images(image_filepath = 'batch_11/000028.jpg'):
#     # User settings
#     pylab.rcParams['figure.figsize'] = (28,28)
#     ####################

#     # Obtain Exif orientation tag code
#     for orientation in ExifTags.TAGS.keys():
#         if ExifTags.TAGS[orientation] == 'Orientation':
#             break

#     # Loads dataset as a coco object
#     coco = COCO(anns_file_path)

#     # Find image id
#     img_id = -1
#     for img in imgs:
#         if img['file_name'] == image_filepath:
#             img_id = img['id']
#             break

#     # Show image and corresponding annotations
#     if img_id == -1:
#         print('Incorrect file name')
#     else:

#         # Load image
#         print(image_filepath)
#         I = Image.open(dataset_path + '/' + image_filepath)

#         # Load and process image metadata
#         if I._getexif():
#             exif = dict(I._getexif().items())
#             # Rotate portrait and upside down images if necessary
#             if orientation in exif:
#                 if exif[orientation] == 3:
#                     I = I.rotate(180,expand=True)
#                 if exif[orientation] == 6:
#                     I = I.rotate(270,expand=True)
#                 if exif[orientation] == 8:
#                     I = I.rotate(90,expand=True)

#         # Show image
#         fig,ax = plt.subplots(1)
#         plt.axis('off')
#         plt.imshow(I)

#         # Load mask ids
#         annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
#         anns_sel = coco.loadAnns(annIds)

#         # Show annotations
#         for ann in anns_sel:
#             color = colorsys.hsv_to_rgb(np.random.random(),1,1)
#             for seg in ann['segmentation']:
#                 poly = Polygon(np.array(seg).reshape((int(len(seg)/2), 2)))
#                 p = PatchCollection([poly], facecolor=color, edgecolors=color,linewidths=0, alpha=0.4)
#                 ax.add_collection(p)
#                 p = PatchCollection([poly], facecolor='none', edgecolors=color, linewidths=2)
#                 ax.add_collection(p)
#             [x, y, w, h] = ann['bbox']
#             rect = Rectangle((x,y),w,h,linewidth=2,edgecolor=color,
#                             facecolor='none', alpha=0.7, linestyle = '--')
#             ax.add_patch(rect)

#         plt.show()

# def filter_by_cat(category_name, nr_img_2_display = 10):
#     # User settings
#     pylab.rcParams['figure.figsize'] = (14,14)
#     ####################

#     # Obtain Exif orientation tag code
#     for orientation in ExifTags.TAGS.keys():
#         if ExifTags.TAGS[orientation] == 'Orientation':
#             break

#     # Loads dataset as a coco object
#     coco = COCO(anns_file_path)

#     # Get image ids
#     imgIds = []
#     catIds = coco.getCatIds(catNms=[category_name])
#     if catIds:
#         # Get all images containing an instance of the chosen category
#         imgIds = coco.getImgIds(catIds=catIds)
#     else:
#         # Get all images containing an instance of the chosen super category
#         catIds = coco.getCatIds(supNms=[category_name])
#         for catId in catIds:
#             imgIds += (coco.getImgIds(catIds=catId))
#         imgIds = list(set(imgIds))

#     nr_images_found = len(imgIds) 
#     print('Number of images found: ',nr_images_found)

#     # Select N random images
#     random.shuffle(imgIds)
#     imgs = coco.loadImgs(imgIds[0:min(nr_img_2_display,nr_images_found)])

#     for img in imgs:
#         image_path = dataset_path + '/' + img['file_name']
#         # Load image
#         I = Image.open(image_path)
        
#         # Load and process image metadata
#         if I._getexif():
#             exif = dict(I._getexif().items())
#             # Rotate portrait and upside down images if necessary
#             if orientation in exif:
#                 if exif[orientation] == 3:
#                     I = I.rotate(180,expand=True)
#                 if exif[orientation] == 6:
#                     I = I.rotate(270,expand=True)
#                 if exif[orientation] == 8:
#                     I = I.rotate(90,expand=True)
        
#         # Show image
#         fig,ax = plt.subplots(1)
#         plt.axis('off')
#         plt.imshow(I)

#         # Load mask ids
#         annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
#         anns_sel = coco.loadAnns(annIds)
        
#         # Show annotations
#         for ann in anns_sel:
#             color = colorsys.hsv_to_rgb(np.random.random(),1,1)
#             for seg in ann['segmentation']:
#                 poly = Polygon(np.array(seg).reshape((int(len(seg)/2), 2)))
#                 p = PatchCollection([poly], facecolor=color, edgecolors=color,linewidths=0, alpha=0.4)
#                 ax.add_collection(p)
#                 p = PatchCollection([poly], facecolor='none', edgecolors=color, linewidths=2)
#                 ax.add_collection(p)
#             [x, y, w, h] = ann['bbox']
#             rect = Rectangle((x,y),w,h,linewidth=2,edgecolor=color,
#                             facecolor='none', alpha=0.7, linestyle = '--')
#             ax.add_patch(rect)

#         plt.show()