# %%
import os
import cv2
import sys
# print(cv2.__version__)
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
from sklearn.metrics import roc_auc_score

from PIL import Image
current_dir = os.getcwd() 


# %%
# Extract the full-body skeletal image from each sample.
bak = str(sys.argv[1]) #"black" "white"

if bak == "white":
    label_nrow = 850
elif bak == "black":
    label_nrow = 850
else:
    label_nrow = 0
directory = "/storageC/shiwei/work/DXA/4_bak_"+bak+"_npy"
img_files = glob.glob(os.path.join(directory, '*.npy'))

df = pd.DataFrame(columns=['File_Name'])
df['File_Name'] = [os.path.splitext(os.path.basename(file))[0] for file in img_files]
df["Label"] = 0

df["Artificial"] = 0
df["Amputation"] = 0

df = df.sort_values(by="File_Name")
# print(df.shape)

label_df_train = pd.read_table("/storageC/shiwei/work/DXA/5_contrast_"+bak+"_label_my.tsv",nrows=label_nrow)

df.loc[df.File_Name.isin(label_df_train.File_Name),"Label"] = label_df_train.loc[label_df_train.File_Name.isin(df.File_Name),"Label"].values
df.to_csv("/storageC/shiwei/work/DXA/5_contrast_"+bak+"_label.tsv",sep="\t",index=False,header=True)

# %%
def copy_pres(source_dir, target_dir, filenames):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
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

# %%
def copy_npys(source_dir, target_dir, filenames):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    for filename in filenames:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        if os.path.exists(source_path):
            if filename.endswith('.npy'):
                if not os.path.exists(target_path):
                    shutil.copyfile(source_path, target_path)
                # print(f"Copied {filename} to {target_dir}")
            else:
                print(f"Ignored {filename}: not a .npy file")
        else:
            print(f"File {filename} not found in {source_dir}")

# %%
# Extract the labeled images and copy them to the training dataset.
label_df_train = pd.read_table("/storageC/shiwei/work/DXA/5_contrast_"+bak+"_label_my.tsv",nrows=label_nrow)
label_df_train["File_Name_npy"] = label_df_train.File_Name + ".npy"
df_label_1 = label_df_train[label_df_train['Label'] == 1]
num_label_1 = df_label_1.shape[0]
df_remaining = label_df_train[label_df_train['Label'] != 1]
df_random_sample = df_remaining.sample(n=num_label_1, random_state=123)
label_df_train = pd.concat([df_label_1, df_random_sample])

copy_npys("/storageC/shiwei/work/DXA/4_bak_"+bak+"_npy", "/storageC/shiwei/work/DXA/5_contrast_"+bak+"_trains", label_df_train.File_Name_npy.to_list())
print(label_df_train.Label.value_counts(),flush=True)

# %%
class NpyDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None, target_size=(960, 384)):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        filename = self.dataframe.iloc[idx, -1]
        label = self.dataframe.iloc[idx, 1]
        # print(f"Filename: {filename}, Label: {label}")
        data = np.load(os.path.join(self.data_dir, filename))

        height, width = data.shape[:2]

        if (height > self.target_size[0]) & (width > self.target_size[1]):
            print(filename)
            crop_height = max(0, height - self.target_size[0])
            crop_top = crop_height // 2
            crop_bottom = height - (crop_height - crop_top)
    
            crop_width = max(0, width - self.target_size[1])
            crop_left = crop_width // 2
            crop_right = width - (crop_width - crop_left)
    
            # Crop the image from the center.
            cropped_image = data[crop_top:crop_bottom, crop_left:crop_right]
            if self.transform:
                transform_image = self.transform(cropped_image)

        elif (height > self.target_size[0]) & (width <= self.target_size[1]):
            print(filename)
            crop_height = max(0, height - self.target_size[0])
            crop_top = crop_height // 2
            crop_bottom = height - (crop_height - crop_top)
    
            pad_width = max(0, self.target_size[1] - width)
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
    
            cropped_image = data[crop_top:crop_bottom, :]
    
            fill_color = (data[0,0], data[0,0])
            padded_image = np.pad(cropped_image, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=fill_color)

            if self.transform:
                transform_image = self.transform(padded_image)
    
        elif (height <= self.target_size[0]) & (width > self.target_size[1]):
            print(filename)
            crop_width = max(0, width - self.target_size[1])
            crop_left = crop_width // 2
            crop_right = width - (crop_width - crop_left)
    
            pad_height = max(0, self.target_size[0] - height)
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
    
            cropped_image = data[:, crop_left:crop_right]
    
            fill_color = (data[0,0], data[0,0])
            padded_image = np.pad(cropped_image, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=fill_color)

            if self.transform:
                transform_image = self.transform(padded_image)

        else:
            pad_height = max(0, self.target_size[0] - height)
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
    
            pad_width = max(0, self.target_size[1] - width)
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            # print("Before padding, data shape:", data.shape)
            fill_color = (data[0,0], data[0,0])
            padded_image = np.pad(data, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=fill_color)
            # print("After padding, data shape:", data.shape)

            if self.transform:
                transform_image = self.transform(padded_image)
        
        return transform_image, label
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# %%
def myDataset(dataset):
    train_data = []
    train_labels = []

    for data, label in dataset:
        train_data.append(data)
        train_labels.append(label)
        
    train_data = torch.stack(train_data)
    train_labels = torch.tensor(train_labels)
    return TensorDataset(train_data, train_labels)

# %%
train_df, test_df = train_test_split(label_df_train, test_size=0.2, random_state=123)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=123)

data_dir = "/storageC/shiwei/work/DXA/5_contrast_"+bak+"_trains"
train_dataset = NpyDataset(train_df, data_dir, transform=transform)
val_dataset = NpyDataset(val_df, data_dir, transform=transform)
test_dataset = NpyDataset(test_df, data_dir, transform=transform)

batch_size = 4
train_loader = DataLoader(myDataset(train_dataset), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(myDataset(val_dataset), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(myDataset(test_dataset), batch_size=batch_size, shuffle=False)

# %%
# Build a ResNet model.
torch.cuda.empty_cache()
model_path="/storageC/shiwei/work/DXA/5_contrast_"+bak+"_models/"
if not os.path.exists(model_path):
    os.makedirs(model_path)

model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 2)

# Define the loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Train the model.
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_val_loss = float('inf')

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
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            probas = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            predictions.extend(probas)
            true_labels.extend(labels.cpu().numpy())
    
    val_loss /= len(val_dataset)
    print(f"Validation Loss: {val_loss:.4f}", flush=True)

    val_auc = roc_auc_score(true_labels, predictions)
    print(f"Validation AUC: {val_auc:.4f}", flush=True)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()

torch.save(best_model, model_path+'best_model_resnet152.pth')


# %%
# Evaluate the best model on the test set.
model.load_state_dict(torch.load(model_path+'/best_model_resnet152.pth'))
model.eval()

true_labels = []
predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        
        probas = torch.softmax(outputs, dim=1)[:, 1]
        
        predictions.extend(probas.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

auc_score = roc_auc_score(true_labels, predictions)
print(f"Test AUC: {auc_score:.4f}", flush=True)
    

# %%
# Apply to all images to predict the classification.

model.load_state_dict(torch.load(model_path+'/best_model_resnet152.pth'))
model.eval()
predicted_labels = np.array([])

label_df_all = pd.read_table("/storageC/shiwei/work/DXA/5_contrast_"+bak+"_label_my.tsv")
# label_df_all["predict"] = 0
label_df_all["File_Name_npy"] = label_df_all.File_Name + ".npy"
print(label_df_all.shape, flush=True)
data_dir = "/storageC/shiwei/work/DXA/4_bak_"+bak+"_npy"

num_parts = 1000
label_df_all_parts = np.array_split(label_df_all, num_parts)

for i, part in enumerate(label_df_all_parts):
    if i % 50 ==0 :
        print(f"Processing Part {i + 1}:", flush=True)
        # print(part.shape)

    pred_dataset = NpyDataset(part, data_dir, transform=transform)
    pred_loader = DataLoader(myDataset(pred_dataset), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for inputs, labels in pred_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted_labels = np.append(predicted_labels, predicted.cpu().numpy())
    np.save("/storageC/shiwei/work/DXA/5_contrast_"+bak+"_pred.npy", predicted_labels)

# %%
# Extract the images classified as having a contrast of 0.

label_df_all = pd.read_table("/storageC/shiwei/work/DXA/5_contrast_"+bak+"_label_my.tsv")
print(len(label_df_all), flush=True)
class_pred = np.load("/storageC/shiwei/work/DXA/5_contrast_"+bak+"_pred.npy")
print(len(class_pred), flush=True)

label_df_all = label_df_all.iloc[0:len(class_pred)]
label_df_all["predict"] = class_pred #[0:len(label_df_all)]
label_df_all["predict"] = label_df_all["predict"].astype(int)
# (label_df_all.Crop == label_df_all.predict).value_counts()

# Images with normal contrast.
label_df_pre = label_df_all.loc[label_df_all.predict == 0].copy().reset_index(drop=True)
label_df_pre["File_Name_npy"] = label_df_pre.File_Name + ".npy"
label_df_pre["File_Name_png"] = label_df_pre.File_Name + ".png"
print(len(label_df_pre), flush=True)

copy_pres("/storageC/shiwei/work/DXA/4_bak_"+bak+"_npy", "/storageC/shiwei/work/DXA/5_contrast_"+bak+"_predicted_npy", label_df_pre.File_Name_npy.to_list())
copy_pres("/storageC/shiwei/work/DXA/4_bak_"+bak+"_png", "/storageC/shiwei/work/DXA/5_contrast_"+bak+"_predicted_png", label_df_pre.File_Name_png.to_list())
