##
# Extract complete total-body bone images

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import cv2
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
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from PIL import Image

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Copy training files
def copy_file(source_path, target_path):
    if os.path.exists(source_path):
        if not os.path.exists(target_path):
            shutil.copyfile(source_path, target_path)
    else:
        print(f"File {os.path.basename(source_path)} not found in {os.path.dirname(source_path)}")

# Parallel copy for predictions
def copy_pres_parallel(source_dir, target_dir, filenames, max_workers=8):
    os.makedirs(target_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for filename in filenames:
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            futures.append(executor.submit(copy_file, source_path, target_path))

        # Use tqdm to show progress bar
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Copying"):
            pass

# Build dataset class
class NpyDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None, target_size=(960, 384)):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        fileprefix = self.dataframe.iloc[idx, 0] # first column is image ID
        label = self.dataframe.iloc[idx, 1]     # second column is label
        filename = self.dataframe.iloc[idx, 2]  # third column is filename
        # print(f"Filename: {filename}, Label: {label}")
        data = np.load(os.path.join(self.data_dir, filename))

        height, width = data.shape[:2]

        if (height > self.target_size[0]) & (width > self.target_size[1]):
            # print("BothOversize:"+filename, flush=True)
            crop_height = max(0, height - self.target_size[0])
            crop_top = crop_height // 2
            crop_bottom = height - (crop_height - crop_top)
    
            crop_width = max(0, width - self.target_size[1])
            crop_left = crop_width // 2
            crop_right = width - (crop_width - crop_left)
    
            # Center crop the image
            cropped_image = data[crop_top:crop_bottom, crop_left:crop_right]
            if self.transform:
                transform_image = self.transform(cropped_image)

        elif (height > self.target_size[0]) & (width <= self.target_size[1]):
            # print("HeightOversize:"+filename, flush=True)

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
            # print("WidthOversize:"+filename, flush=True)

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
        # Pad the image
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
        
        return transform_image, label, fileprefix
    
# Define transforms
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

# Aggregates per-sample tensors and labels
def myDataset(dataset):
    train_data = []
    train_labels = []

    for data, label, filenames in dataset:
        train_data.append(data)
        train_labels.append(label)
        
    train_data = torch.stack(train_data)
    train_labels = torch.tensor(train_labels)
    return TensorDataset(train_data, train_labels)

# Predict images in batches
def process_part_pool(args):
    part_df, part_idx, transform, raw_path, save_dir = args

    # Load model (each process loads independently)）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model (grayscale input + binary classification)
    model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Load weights
    model.load_state_dict(torch.load('/storageC/shiwei/work/DXA/3_crop_models/best_model_resnet152.pth', map_location=device))
    model.to(device)
    model.eval()

    dataset = NpyDataset(part_df, raw_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    results = []

    with torch.no_grad():
        for inputs, labels, filenames in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for fname, pred in zip(filenames, predicted.cpu().numpy()):
                results.append((fname, pred))

    # Each subprocess saves its own prediction results
    part_output_path = os.path.join(save_dir, f"preds_part_{part_idx}.npy")
    np.save(part_output_path, results)

    return part_output_path  # convenient for the main process to record

# Model building, training, validation and testing
def train_model():
    set_seed(seed)
    # Read manually annotated labels for constructing complete total-body images
    label_df_train = pd.read_table("/storageC/shiwei/work/DXA/3_crop_label.tsv")
    train_path = "/storageC/shiwei/work/DXA/3_crop_trains"
    transform = get_transform()
    batch_size = 8

    # Split dataset into training, validation and test sets
    train_df, test_df = train_test_split(label_df_train, stratify=label_df_train['Label'], test_size=0.2, random_state=seed)
    val_df, test_df = train_test_split(test_df, stratify=test_df['Label'], test_size=0.5, random_state=seed)

    train_dataset = NpyDataset(train_df, train_path, transform=transform)
    val_dataset = NpyDataset(val_df, train_path, transform=transform)
    test_dataset = NpyDataset(test_df, train_path, transform=transform)

    train_loader = DataLoader(myDataset(train_dataset), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(myDataset(val_dataset), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(myDataset(test_dataset), batch_size=batch_size, shuffle=False)

    
    # Build ResNet model
    torch.cuda.empty_cache()
    model_path='/storageC/shiwei/work/DXA/3_crop_models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Load pretrained weights
    model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # grayscale
    model.fc = nn.Linear(model.fc.in_features, 2)  # binary classification

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train model
    num_epochs = 30
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

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                probas = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                predictions.extend(probas)
                true_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_dataset)
        print(f"Validation Loss: {val_loss:.4f}", flush=True)
        # val_accuracy = correct / total
        # print(f"Validation Accuracy: {val_accuracy:.4f}")
        val_auc = roc_auc_score(true_labels, predictions)
        print(f"Validation AUC: {val_auc:.4f}", flush=True)

        if (val_loss < best_val_loss) & (epoch_loss < 0.01):
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path+'/best_model_resnet152.pth')

    
    # Evaluate best model on test set
    model.load_state_dict(torch.load(model_path+'/best_model_resnet152.pth'))
    model.eval()

    true_labels = []
    predictions = []
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            probas = torch.softmax(outputs, dim=1)[:, 1]
            
            # Save predicted probabilities and true labels
            predictions.extend(probas.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
    auc_score = roc_auc_score(true_labels, predictions)
    print(f"Test AUC: {auc_score:.4f}", flush=True)
    # test_accuracy = correct / total
    # print(f"Test Accuracy: {test_accuracy:.4f}")
    test_loss /= len(test_dataset)
    print(f"Test Loss: {test_loss:.4f}", flush=True)

    torch.cuda.empty_cache()

# Apply the model to all images for prediction
def predict_all_data():
    # Get list of all .npy files in the directory
    npy_files = glob.glob(os.path.join('/storageC/shiwei/work/DXA/2_class_predicted_npy', '*.npy'))
    # Create an initial dataframe to hold file names and labels
    label_df_all = pd.DataFrame(columns=['File_Name'])
    label_df_all['File_Name'] = [os.path.splitext(os.path.basename(file))[0] for file in npy_files]
    label_df_all["Label"] = 0
    label_df_all["File_Name_npy"] = label_df_all.File_Name + ".npy"
    raw_path = "/storageC/shiwei/work/DXA/2_class_predicted_npy"

    num_parts = 1000
    label_df_all_parts = np.array_split(label_df_all, num_parts)

    save_dir = "/storageC/shiwei/work/DXA/3_crop_pred_parts"
    os.makedirs(save_dir, exist_ok=True)

    args_list = [
        (part_df, i, get_transform(), raw_path, save_dir)
        for i, part_df in enumerate(label_df_all_parts)
    ]

    with Pool(processes=8) as pool:
        output_files = list(tqdm(pool.imap_unordered(process_part_pool, args_list), total=len(args_list)))

    all_results = []
    for file in sorted(output_files):
        results = np.load(file, allow_pickle=True)
        all_results.extend(results)
    np.save("/storageC/shiwei/work/DXA/3_crop_pred.npy", all_results)

# Extract complete total-body images classified as 0
def filter_and_copy_predicted():
    # Get list of all .npy files in the directory
    npy_files = glob.glob(os.path.join('/storageC/shiwei/work/DXA/2_class_predicted_npy', '*.npy'))
    # Create an initial dataframe to hold file names and labels
    label_df_all = pd.DataFrame(columns=['File_Name'])
    label_df_all['File_Name'] = [os.path.splitext(os.path.basename(file))[0] for file in npy_files]

    crop_pred = np.load("/storageC/shiwei/work/DXA/3_crop_pred.npy")
    print(len(crop_pred), flush=True)
    if len(label_df_all) != len(crop_pred):
        raise ValueError(f"Length mismatch: label_df_all ({len(label_df_all)}) vs crop_pred ({len(crop_pred)})")
        
    label_df_all = label_df_all.merge(pd.DataFrame(crop_pred, columns=["File_Name", "predict"]), on="File_Name", how="left")
    label_df_all["predict"] = label_df_all["predict"].astype(int)

    label_df_pre = label_df_all.loc[label_df_all.predict == 0].copy().reset_index(drop=True)
    label_df_pre["File_Name_npy"] = label_df_pre.File_Name + ".npy"
    label_df_pre["File_Name_png"] = label_df_pre.File_Name + ".png"
    print(len(label_df_pre), flush=True)

    copy_pres_parallel("/storageC/shiwei/work/DXA/2_class_predicted_npy", "/storageC/shiwei/work/DXA/3_crop_predicted_npy", label_df_pre.File_Name_npy.to_list(), max_workers=32)
    copy_pres_parallel("/storageC/shiwei/work/DXA/2_class_predicted_png", "/storageC/shiwei/work/DXA/3_crop_predicted_png", label_df_pre.File_Name_png.to_list(), max_workers=32)

def main():
    # Train model and test
    train_model()

    # Apply model to all images for prediction
    predict_all_data()

    # Extract complete total-body images classified as 0
    filter_and_copy_predicted()

    print("Done.")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    seed = 42
    set_seed(seed)
    main()
