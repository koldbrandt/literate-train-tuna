
import os
import numpy as np
import glob
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchcam.methods import SmoothGradCAMpp, GradCAM, GradCAMpp, XGradCAM, LayerCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image

import sys

import dataset
import utilities as ut

from model import Network, ResNet, FinetuneResnet50

train_data, test_data = dataset.get_data(64)
model = FinetuneResnet50(2)
device = ut.get_device()
model.to(device)
model.load_state_dict(torch.load('models/model.pt'))

# Methods related to gradient-based class activation maps:
#cam_extractor = SmoothGradCAMpp(model)
cam_extractor = GradCAM(model)
#cam_extractor = GradCAMpp(model)
#cam_extractor = XGradCAM(model)
#cam_extractor = LayerCAM(model)

it = iter(test_data)
first_batch = next(it)
first_img = first_batch[0][5].to(device)


out = model(first_img.unsqueeze(0))
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
# Visualize the raw CAM
plt.imshow(activation_map[0].squeeze(0).cpu().numpy()); plt.axis('off'); plt.tight_layout(); plt.savefig('img1.jpg')   
result = overlay_mask(to_pil_image(first_img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.savefig('img2.jpg')