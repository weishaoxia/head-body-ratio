# %%
import datetime
from tqdm import tqdm
import random
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import shutil
import json

import math
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset, Subset, DataLoader, random_split
import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import save_image

from sklearn.metrics import roc_auc_score
import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw 

import cv2

if True:

    def scale_img(data, transf, target_size=(960, 384)):

        height, width = data.shape[:2]

        if (height > target_size[0]) & (width > target_size[1]):

            crop_height = max(0, height - target_size[0])
            crop_top = crop_height // 2
            crop_bottom = height - (crop_height - crop_top)
        
            crop_width = max(0, width - target_size[1])
            crop_left = crop_width // 2
            crop_right = width - (crop_width - crop_left)
        
            # Crop the image from the center.
            cropped_image = data[crop_top:crop_bottom, crop_left:crop_right, :]
            padded_image = cropped_image

        elif (height > target_size[0]) & (width <= target_size[1]):
            crop_height = max(0, height - target_size[0])
            crop_top = crop_height // 2
            crop_bottom = height - (crop_height - crop_top)
        
            pad_width = max(0, target_size[1] - width)
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
        
            cropped_image = data[crop_top:crop_bottom, :]
        
            fill_color = (data[0,0,0], data[0,0,0])
            padded_image = np.pad(cropped_image, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=fill_color)
        
        elif (height <= target_size[0]) & (width > target_size[1]):
            crop_width = max(0, width - target_size[1])
            crop_left = crop_width // 2
            crop_right = width - (crop_width - crop_left)
        
            pad_height = max(0, target_size[0] - height)
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
        
            cropped_image = data[:, crop_left:crop_right]
        
            fill_color = (data[0,0,0], data[0,0,0])
            padded_image = np.pad(cropped_image, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=fill_color)

        else:
            pad_height = max(0, target_size[0] - height)
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
        
            pad_width = max(0, target_size[1] - width)
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            # print("Before padding, data shape:", data.shape)
            fill_color = (data[0,0,0], data[0,0,0])
            padded_image = np.pad(data, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=fill_color)

        if transf:
            transform_image = transform(padded_image)
            return transform_image
        else:
            return padded_image

    class Segmentation_test_Dataset(Dataset):
        def __init__(self, image_dir, transform=None):
            self.image_dir = image_dir
            self.transform = transform
            self.images = [file for file in os.listdir(image_dir) if file.endswith('.png')]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = os.path.join(self.image_dir, self.images[idx])
            image = cv2.imread(img_path) # Image.open(img_path).convert("RGB")
            image = scale_img(image, transf=False)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                image = self.transform(image)
            return image

    transform=transforms.Compose({
        # Convert to tensor
        transforms.ToTensor()
    })

    # %%
    def segment_thresh(seg_img, threshold):
        outputs_t= np.where(seg_img > threshold, 1, seg_img)
        outputs_t= np.where(outputs_t < threshold, 0, outputs_t)

        return outputs_t


    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# dcm_table.sort_values("height")
# unique_counts = dcm_table.groupby('folder')['Age'].nunique()
# unique_counts.loc[unique_counts == 2]

# %%
# "head","shoulder_left","shoulder_right","ankle_left","ankle_right",
label_list = ["head","shoulder_left","shoulder_right","ankle_left","ankle_right","hip_left","hip_right"]

bak = str(sys.argv[1]) #"black"
threshood = 0
model_type = str(sys.argv[2]) #"resnet152" #"tu-hrnet_w32"
suffix = str(sys.argv[3]) #"2024-07-08"
pose = "direct"

if model_type == "resnet152":
    # resnet152
    black_label_model_dict = {
        "head":"127", # 0.0125
        "shoulder_left":"145", # 0.0625
        "shoulder_right":"113", # 0.059
        "ankle_left":"110", # 0.0812
        "ankle_right":"132", # 0.0748
        "hip_left":"113", # 0.064
        "hip_right":"148" # 0.083
    }
    white_label_model_dict = {
        "head":"110", # 0.0114
        "shoulder_left":"122", # 0.0877
        "shoulder_right":"138", # 0.0694
        "ankle_left":"116", # 0.0828
        "ankle_right":"130", # 0.0856
        "hip_left":"135", # 0.0803
        "hip_right":"127" # 0.0785
    }
elif model_type == "tu-hrnet_w32":
    # tu-hrnet_w32
    black_label_model_dict = {
        "head":"130", # 0.013
        "shoulder_left":"119", # 0.0678
        "shoulder_right":"138", # 0.0657
        "ankle_left":"131", # 0.0812
        "ankle_right":"136", # 0.0748
        "hip_left":"101",
        "hip_right":"101"
    }
    white_label_model_dict = {
        "head":"150", # 0.0114
        "shoulder_left":"125", # 0.0714
        "shoulder_right":"142", # 0.0823
        "ankle_left":"139", # 0.0936
        "ankle_right":"113", # 0.0925
        "hip_left":"101",
        "hip_right":"101"
    }


# %%
# out put
table_path = "/storageC/shiwei/work/DXA/8_segment_result/segment_table_"+bak+"_"+suffix+".tsv"

if os.path.isfile(table_path):
    dcm_table = pd.read_table(table_path,sep="\t")
else:
    dcm_table = pd.read_table("/storageC/shiwei/work/DXA/dcm_table.tsv",sep="\t")

for label in label_list:

    if bak == "black":
        bestmodel = black_label_model_dict[label]
    elif bak == "white":
        bestmodel = white_label_model_dict[label]
    else: 
        print("background error")
        sys.exit()

    set_seed(int(bestmodel))
    print(label, flush=True)
    
    # Apply the model to all images for segmentation
    if True:
        best_model_name = "/storageC/shiwei/work/DXA/7_segment_trains_"+label+"_"+bak+"_direct/models_test_"+model_type+"_"+bestmodel+"/"+model_type+"_model_best_"+bestmodel+".pth"
        # best_model_name = "/storageC/shiwei/work/DXA/segment_trains_"+label+"_"+bak+"_direct/models_test_"+bestmodel+"/"+model_type+"_model_best_"+bestmodel+".pth" # old
        model = smp.Unet(
            encoder_name=model_type,  
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
        )
        
        model.load_state_dict(torch.load(best_model_name))
        model.to(device)
        model.eval()

        input_path = "/storageC/shiwei/work/DXA/6_pose_predicted_"+bak+"_png1/"
        input_list = os.listdir(input_path)
        # 只取.png
        input_list = [file for file in input_list if file.endswith('.png')]

        output_path = "/storageC/shiwei/work/DXA/8_segment_predicted_"+label+"_"+ bak +"/"+ suffix +"/"
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

        all_dataset = Segmentation_test_Dataset(image_dir=input_path, transform=transform)

        part_size = 100
        num_parts = len(all_dataset) // part_size
        parts = [Subset(all_dataset, range(i * part_size, (i + 1) * part_size)) for i in range(num_parts)]
        last_part = Subset(all_dataset, range(num_parts * part_size, len(all_dataset)) )
        parts = parts + [last_part]

        img_index = 0
        bs=1
        for i, part in enumerate(parts):
            # print(f"Processing Part {i + 1}:", flush=True)
            part_loader = DataLoader(part, batch_size=bs, shuffle=False)
            with torch.no_grad():
                for inputs in part_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    outputs = outputs[0].permute(1,2,0).cpu().numpy()
                    np.save(os.path.join(output_path ,input_list[img_index].replace('.png', '.npy')), outputs)
                    img_index += 1


    input_path = "/storageC/shiwei/work/DXA/8_segment_predicted_"+label+"_"+ bak +"/"+ suffix +"/"
    input_list = os.listdir(input_path)
    print(len(input_list), flush=True)
    for input_ in input_list:

        data = np.load(input_path+input_)
        data = segment_thresh(data, threshood)
        image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # plt.imshow(image)
        # plt.show()

        _, binary_image = cv2.threshold(image.astype(np.uint8), threshood, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
        except:
            x, y, w, h = -1, -1, -1, -1
            
        dcm_table.loc[dcm_table.Image == input_.split(".")[0], [label+"_x", label+"_y", label+"_w", label+"_h"]] = x, y, w, h

    dcm_table.to_csv("/storageC/shiwei/work/DXA/8_segment_result/segment_table_"+bak+"_"+suffix+".tsv",sep="\t",index=False)
    # shutil.rmtree(input_path) # During testing, the data is deleted after processing; in the final version, it will not be deleted.

    del bestmodel
print("Done")
