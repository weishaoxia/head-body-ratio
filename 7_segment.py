# %%
from tqdm import tqdm
import random
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import shutil
import json

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

# %%
def shape_to_mask(img_shape, points, shape_type=None, line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]

    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool) 
    return mask

# %%
def scale_img(data, transf, target_size=(960, 384)):

    height, width = data.shape[:2]

    if (height > target_size[0]) & (width > target_size[1]):

        crop_height = max(0, height - target_size[0])
        crop_top = crop_height // 2
        crop_bottom = height - (crop_height - crop_top)
    
        crop_width = max(0, width - target_size[1])
        crop_left = crop_width // 2
        crop_right = width - (crop_width - crop_left)
    
        # 从中心裁剪图像
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

# %%
def segment_thresh(seg_img, threshold):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_t= torch.where(seg_img >=threshold, torch.tensor(1,dtype=torch.float).to(device), seg_img)
    outputs_t= torch.where(outputs_t < threshold, torch.tensor(0,dtype=torch.float).to(device), outputs_t)

    return outputs_t

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
    #转化为Tensor
    transforms.ToTensor()
})

# %%
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
        image = cv2.imread(img_path) # Image.open(img_path).convert("RGB")
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path) # Image.open(mask_path).convert("L")  # 掩码通常是单通道图像
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)#掩膜转为灰度图

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
    
transform=transforms.Compose({
    #转化为Tensor
    transforms.ToTensor()
})

    
# %%
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
    
def calculate_iou(preds, labels):
    intersection = np.logical_and(preds, labels).sum()
    union = np.logical_or(preds, labels).sum()
    iou = intersection / union
    return iou


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# %%
# json2png
label_list = ["head","shoulder_left","shoulder_right","ankle_left","ankle_right"]
label_index = {"head":0, "shoulder_left":1,"shoulder_right":2,"ankle_left":3,"ankle_right":4}
label_shape = {"head":"polygon", "shoulder_left":"point","shoulder_right":"point","ankle_left":"point","ankle_right":"point"}

bak = str(sys.argv[1]) # "black" "white"
pose = "direct"
iftrain = True
model_type = str(sys.argv[2]) # tu-hrnet_w48 resnet152 tu-hrnet_w32
suffix = str(sys.argv[3]) # run suffix

seed = int(sys.argv[4])
set_seed(seed)

for label in ["head","shoulder_left","shoulder_right","ankle_left","ankle_right"]:
    print(label, flush=True)
    torch.cuda.empty_cache()

    mask_path = "/storageC/shiwei/work/DXA/7_segment_masked_" + bak + "_"+ pose 
    outmask_path = "/storageC/shiwei/work/DXA/7_segment_trains_"+label+"_"+ bak + "_"+ pose +"/mask"+ "_" + suffix
    outimg_path = "/storageC/shiwei/work/DXA/7_segment_trains_"+label+"_"+ bak+ "_"+ pose +"/image"+ "_" + suffix
    model_path = "/storageC/shiwei/work/DXA/7_segment_trains_"+label+"_"+ bak+ "_"+ pose +"/models"+ "_" + suffix
    png_path = "/storageC/shiwei/work/DXA/6_pose_predicted_"+bak+"_png"

    if os.path.exists(outmask_path):
        shutil.rmtree(outmask_path)
    os.makedirs(outmask_path, exist_ok=True)

    if os.path.exists(outimg_path):
        shutil.rmtree(outimg_path)
    os.makedirs(outimg_path, exist_ok=True)

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)

    # %%
    # Copy the images to the training folder.

    masked_list=os.listdir(mask_path)

    for sample in masked_list:
        sample = sample.split(".")[0]

        json_path = mask_path +'/'+sample+'.json'
        img_path =  png_path + '/'+sample+'.png'
     
        labelme_json = json.load(open(json_path, encoding='utf-8'))
        orgin_image = cv2.imread(img_path)

        shutil.copyfile(img_path, os.path.join(outimg_path ,sample+".png"))
        
        mask_image = shape_to_mask(orgin_image.shape, labelme_json['shapes'][label_index[label]]['points'], label_shape[label])
        mask_image = cv2.cvtColor(mask_image.astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)

        orgin_image_scaled = scale_img(orgin_image, transf=False)
        mask_image_scaled = scale_img(mask_image, transf=False)

        cv2.imwrite(os.path.join(outimg_path ,sample+".png"), orgin_image_scaled)
        cv2.imwrite(os.path.join(outmask_path ,sample+".png"), mask_image_scaled)

    # %%
    dataset = SegmentationDataset(image_dir=outimg_path, mask_dir=outmask_path, transform=transform)

    bs=1
    # Split the dataset.
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    # %%
    model = smp.Unet(
            encoder_name=model_type,  
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
        )

    # Configure the model hyperparameters.
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr=1e-4
    weight_decay=1e-4
    num_epochs=50
    
    # %%
    torch.cuda.empty_cache()


    # Define the loss function and optimizer.
    criterion = dice_loss # dice_loss nn.BCELoss() nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)

    # Train the model.
    model.to(device)

    best_val_loss = float('inf')
    early_stop=0

    if iftrain:
        best_model_name=model_path+"/"+model_type+"_model_best_"+str(seed)+".pth"
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

            # Evaluate the model on the validation set.
            model.eval()
            val_loss = 0.0
            # val_aucs = []
            val_ious = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0) #* batch_size

                    preds = segment_thresh(outputs, 0)
                    # auc = roc_auc_score(labels[:,0].cpu().numpy().flatten(), preds.cpu().numpy().flatten())
                    # val_aucs.append(auc)
                    iou = calculate_iou(preds.cpu().numpy().flatten(), labels.cpu().numpy().flatten())
                    val_ious.append(iou)
            
            val_loss /= len(val_dataset)
            print(f"Validation Loss: {val_loss:.4f}", flush=True)

            # val_auc = np.mean(val_aucs)
            # print(f"Validation AUC: {val_auc:.4f}", flush=True)

            val_iou = np.mean(val_ious)
            print(f"Validation IoU: {val_iou:.4f}", flush=True)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                early_stop=0
            else:
                early_stop += 1
                if early_stop == 10:
                    break;

        torch.save(best_model, best_model_name)
    else:
        best_model_name=model_path+"/"+model_type+"_model_best_"+str(seed)+".pth"


    # %%
    # Evaluate the best model on the test set.
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

    # test_auc = np.mean(test_aucs)
    # print(f"Test Auc: {test_auc:.4f}", flush=True)
    test_loss /= len(test_dataset)
    print(f"Test Loss: {test_loss:.4f}", flush=True)
    test_ious = np.mean(test_ious)
    print(f"Test IoU: {test_ious:.4f}", flush=True)

    print("Done!", flush=True)


    # ply to all images for segmentation.
    if True:

        model.load_state_dict(torch.load(best_model_name))
        model.to(device)
        model.eval()

        input_path = "/storageC/shiwei/work/DXA/6_pose_predicted_"+bak+"_png/"
        input_list = os.listdir(input_path)

        input_list = [file for file in input_list if file.endswith('.png')]

        output_path = "/storageC/shiwei/work/DXA/7_segment_predicted_" + \
            label+"_" + bak + "/" + suffix + "/"
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

        all_dataset = Segmentation_test_Dataset(
            image_dir=input_path, transform=transform)

        part_size = 100
        num_parts = len(all_dataset) // part_size
        parts = [Subset(all_dataset, range(i * part_size, (i + 1) * part_size))
            for i in range(num_parts)]
        last_part = Subset(all_dataset, range(
            num_parts * part_size, len(all_dataset)))
        parts = parts + [last_part]

        img_index = 0
        bs = 1
        for i, part in enumerate(parts):
            # print(f"Processing Part {i + 1}:", flush=True)
            part_loader = DataLoader(part, batch_size=bs, shuffle=False)
            with torch.no_grad():
                for inputs in part_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    outputs = outputs[0].permute(1, 2, 0).cpu().numpy()
                    np.save(os.path.join(
                        output_path, input_list[img_index].replace('.png', '.npy')), outputs)
                    img_index += 1