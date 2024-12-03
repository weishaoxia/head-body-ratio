import os
import cv2
print(cv2.__version__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom as dicom
import shutil
import glob
from tqdm import tqdm
import random

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp

from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from PIL import Image

def copy_pres(source_dir, target_dir, filenames):
    os.makedirs(target_dir, exist_ok=True)
    
    for filename in filenames:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        if os.path.exists(source_path):
            if not os.path.exists(target_path):
                shutil.copyfile(source_path, target_path)
                # print(f"Copied {filename} to {target_dir}")
        else:
            print(f"File {filename} not found in {source_dir}")

# Distinguish between white and black backgrounds
# Remove oversized images

bak_white_size = (960, 384)
bak_black_size = (864, 288)

input_path = "/storageC/shiwei/work/DXA/3_crop_predicted_npy/"
input_list = os.listdir(input_path)

print(len(input_list))

bak_black_list = []
bak_black_os_list = []
bak_white_list = []
bak_white_os_list = []

for filename in tqdm(input_list):
    file_size = os.path.getsize(os.path.join("/storageC/shiwei/work/DXA/3_crop_predicted_png/", filename.replace('.npy', '.png')))
    file_size_kb = file_size / 1024 

    if file_size_kb > 40:
        data = np.load(os.path.join(input_path, filename))
        height, width = data.shape[:2]

        arr = data[0:9,0:9]
        unique_elements, counts = np.unique(arr, return_counts=True)
        mode_index = np.argmax(counts)
        bak_col = unique_elements[mode_index]
        
        if bak_col == 0:
            bak = "black"
            if (height > bak_black_size[0]) | (width > bak_black_size[1]):
                bak_black_os_list.append(filename)
            else:
                bak_black_list.append(filename)

        elif bak_col == 252:
            bak = "white"
            if (height > bak_white_size[0]) | (width > bak_white_size[1]):
                bak_white_os_list.append(filename)
            else:
                bak_white_list.append(filename)
        else:
            print(filename, bak_col,flush=True)

        # plt.imshow(data, cmap='gray')
        # plt.show()
    else:
        print(filename, bak_col,flush=True)
          
print(len(bak_black_list), len(bak_black_os_list),flush=True)
print(len(bak_white_list), len(bak_white_os_list),flush=True)

# print(pd.Series(bak_black_list).str.split("_").str[0].value_counts().shape,flush=True)
# print(pd.Series(bak_white_list).str.split("_").str[0].value_counts().shape,flush=True)

if True:
    copy_pres("/storageC/shiwei/work/DXA/3_crop_predicted_npy", "/storageC/shiwei/work/DXA/4_bak_black_npy", bak_black_list)
    copy_pres("/storageC/shiwei/work/DXA/3_crop_predicted_npy", "/storageC/shiwei/work/DXA/4_bak_white_npy", bak_white_list)

    copy_pres("/storageC/shiwei/work/DXA/3_crop_predicted_png", "/storageC/shiwei/work/DXA/4_bak_black_png", [item.replace('.npy', '.png') for item in bak_black_list])
    copy_pres("/storageC/shiwei/work/DXA/3_crop_predicted_png", "/storageC/shiwei/work/DXA/4_bak_white_png", [item.replace('.npy', '.png') for item in bak_white_list])