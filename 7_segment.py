##
# Image segmentation to identify head and landmarks

import time
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import shutil
import json
import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, Subset, DataLoader, random_split
import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import save_image

from sklearn.metrics import roc_auc_score
import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw 

import cv2

# Convert label points into a binary mask according to the shape type

def shape_to_mask(img_shape, points, shape_type=None, line_width=10, point_size=5):

    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]

    if shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool) 
    return mask

# Resize and pad/crop images
def scale_img(data, transf, target_size=(960, 384)):

    height, width = data.shape[:2]

    if (height > target_size[0]) & (width > target_size[1]):

        crop_height = max(0, height - target_size[0])
        crop_top = crop_height // 2
        crop_bottom = height - (crop_height - crop_top)
    
        crop_width = max(0, width - target_size[1])
        crop_left = crop_width // 2
        crop_right = width - (crop_width - crop_left)
    
        # Center crop the image
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
    # Pad the image
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

# Apply thresholding to segmentation outputs
def segment_thresh(seg_img, threshold):
    outputs_t= torch.where(seg_img >=threshold, torch.tensor(1,dtype=torch.float).to(device), seg_img)
    outputs_t= torch.where(outputs_t < threshold, torch.tensor(0,dtype=torch.float).to(device), outputs_t)
    return outputs_t

# Dataset for applying trained model to test images
class Segmentation_test_Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [file for file in os.listdir(image_dir) if file.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path) 
        image = scale_img(image, transf=False)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)
        return image

# Default image transform: convert to tensor only
def get_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])

# Dataset for segmentation training and validation
# Loads paired image and mask
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])  # 确保掩码和图像文件名匹配
        image = cv2.imread(img_path) 
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path) # 掩码是单通道图像
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY) #掩码转为灰度图

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
   
# Dice loss for binary segmentation.
def dice_loss(logits, target):
    smooth = 1.
    prob  = torch.sigmoid(logits)
    batch = prob.size(0)
    prob   = prob.view(batch,1,-1)
    target = target.view(batch,1,-1)
    intersection = torch.sum(prob*target, dim=2)
    denominator  = torch.sum(prob, dim=2) + torch.sum(target, dim=2)
    dice = (2*intersection + smooth) / (denominator + smooth)
    dice = torch.mean(dice)
    dice_loss = 1. - dice
    return dice_loss

# Calculate Intersection over Union between predicted and true masks
def calculate_iou(preds, labels):
    intersection = np.logical_and(preds, labels).sum()
    union = np.logical_or(preds, labels).sum()
    iou = intersection / union
    return iou

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
torch.set_num_threads(32)


# json2png
label_list = ["head","shoulder_left","shoulder_right","ankle_left","ankle_right","hip_left","hip_right"]
label_index = {"head":0, "shoulder_left":1,"shoulder_right":2,"ankle_left":3,"ankle_right":4,"hip_left":5,"hip_right":6}
label_shape = {"head":"polygon", "shoulder_left":"point","shoulder_right":"point","ankle_left":"point","ankle_right":"point","hip_left":"point","hip_right":"point"}

bak = str(sys.argv[1])
label = str(sys.argv[2])
pose = "direct"
seed = 42
print(label, flush=True)

device = torch.device("cuda:0")

set_seed(seed)
torch.cuda.empty_cache()

# Input data paths
png_path = "/home/shiwei/work/DXA/1_Total_Body_png"
mask_path = "/home/shiwei/work/DXA/7_segment_masked_" + bak + "_"+ pose + "/in"

# Output data paths
outmask_path = "/home/shiwei/work/DXA/7_segment_trains_"+label+"_"+ bak + "_"+ pose +"/mask"
outimg_path = "/home/shiwei/work/DXA/7_segment_trains_"+label+"_"+ bak+ "_"+ pose +"/image"
model_path = "/home/shiwei/work/DXA/7_segment_trains_"+label+"_"+ bak+ "_"+ pose +"/models"

os.makedirs(outmask_path, exist_ok=True)
os.makedirs(outimg_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# Copy images and masks into training directory
masked_list = [f for f in os.listdir(mask_path) if f.endswith('.json')]
for sample in masked_list:
    sample = sample.split(".")[0]
    if not (os.path.exists(os.path.join(outimg_path ,sample+".png")) and os.path.exists(os.path.join(outmask_path ,sample+".png"))):
        json_path = mask_path +'/'+sample+'.json'
        img_path =  png_path + '/'+sample+'.png'
     
        labelme_json = json.load(open(json_path, encoding='utf-8'))
        orgin_image = cv2.imread(img_path)
        
        mask_image = shape_to_mask(orgin_image.shape, labelme_json['shapes'][label_index[label]]['points'], label_shape[label])
        mask_image = cv2.cvtColor(mask_image.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)

        orgin_image_scaled = scale_img(orgin_image, transf=False)
        mask_image_scaled = scale_img(mask_image, transf=False)

        cv2.imwrite(os.path.join(outimg_path ,sample+".png"), orgin_image_scaled)
        cv2.imwrite(os.path.join(outmask_path ,sample+".png"), mask_image_scaled)


dataset = SegmentationDataset(image_dir=outimg_path, mask_dir=outmask_path, transform=get_transform())

bs=4
# Split dataset into training, validation and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)


model = smp.Unet(
        encoder_name="resnet152",  
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
    )

# Model hyperparameters
lr=1e-4
weight_decay=1e-4
num_epochs=50

torch.cuda.empty_cache()


# Define loss function and optimizer
criterion = dice_loss
optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)

# Train model
model.to(device)

best_val_loss = float('inf')
early_stop=0
best_model_name=model_path+"/model_best_"+str(seed)+".pth"

if not os.path.exists(best_model_name):
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}", flush=True)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        # val_aucs = []
        val_ious = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                preds = segment_thresh(outputs, 0)
                iou = calculate_iou(preds.cpu().numpy().flatten(), labels.cpu().numpy().flatten())
                val_ious.append(iou)
        
        val_loss /= len(val_dataset)
        print(f"Validation Loss: {val_loss:.4f}", flush=True)

        val_iou = np.mean(val_ious)
        print(f"Validation IoU: {val_iou:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            early_stop=0
        else:
            early_stop += 1
            if early_stop == 10:
                break;

    torch.save(best_model, best_model_name)

    
    # Evaluate best model on test set
    model.load_state_dict(torch.load(best_model_name))
    model.to(device)
    model.eval()

    test_loss = 0.0
    # test_aucs = []
    test_ious = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
            preds = segment_thresh(outputs, 0.5)
            # auc = roc_auc_score(labels[:,0].cpu().numpy().flatten(), preds.cpu().numpy().flatten())
            # test_aucs.append(auc)
            iou = calculate_iou(preds.cpu().numpy().flatten(), labels.cpu().numpy().flatten())
            test_ious.append(iou)

    test_loss /= len(test_dataset)
    print(f"Test Loss: {test_loss:.4f}", flush=True)
    test_ious = np.mean(test_ious)
    print(f"Test IoU: {test_ious:.4f}", flush=True)

    print("Done!", flush=True)


# Apply model to all images for segmentation
output_path = "/home/shiwei/work/DXA/7_segment_predicted_" + label+"_" + bak + "/"

os.makedirs(output_path, exist_ok=True)
model.load_state_dict(torch.load(best_model_name))
model.to(device)
model.eval()

input_path = "/home/shiwei/work/DXA/6_pose_"+bak+"_predicted_png/"
input_list = os.listdir(input_path)

# Only use .png files
input_list = [file for file in input_list if file.endswith('.png')]

all_dataset = Segmentation_test_Dataset(
    image_dir=input_path, transform=get_transform())

part_size = 500
num_parts = len(all_dataset) // part_size
parts = [Subset(all_dataset, range(i * part_size, (i + 1) * part_size))
    for i in range(num_parts)]
last_part = Subset(all_dataset, range(
    num_parts * part_size, len(all_dataset)))
parts = parts + [last_part]

img_index = 0
bs = 4
for i, part in enumerate(parts):
    part_loader = DataLoader(part, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for inputs in part_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # [B, 1, H, W]

            for b in range(outputs.shape[0]):
                out = outputs[b].permute(1, 2, 0).cpu().numpy()
                filename = input_list[img_index].replace('.png', '.npy')
                np.save(os.path.join(output_path, filename), out)
                img_index += 1