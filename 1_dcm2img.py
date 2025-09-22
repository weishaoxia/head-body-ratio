import os
import zipfile
import time
import cv2
print(cv2.__version__)
import pandas as pd
import numpy as np
import pydicom as dicom
import shutil
from tqdm import tqdm

# Construct output dataframe
dcm_table = pd.DataFrame(columns=["folder","ID","Sex","Age","height","Weight","Ethnic","DeviceSoftware","Image"])
# Original UKB DXA data folder, each sample is a zip file, total 76320 files
DXA_data_folder = "/home/shiwei/work/DXA/0_DXA_data"
# Each sample is unzipped into a subdirectory; each sample may contain multiple images
DXA_data_unzip = "/home/shiwei/work/DXA/0_DXA_data_unzip"
DXA_data_folder_list = [file for file in os.listdir(DXA_data_folder) if file.endswith(".zip")]

# Output .npy directory
out_path_npy = "/home/shiwei/work/DXA/1_Total_Body_npy"
if not os.path.exists(out_path_npy):
    os.makedirs(out_path_npy)
# Output .png directory
out_path_png = "/home/shiwei/work/DXA/1_Total_Body_png"
if not os.path.exists(out_path_png):
    os.makedirs(out_path_png)
# Image counter
all_images = 0

for folder_path in DXA_data_folder_list:
    folder = folder_path.split(".")[0]
    # Unzip
    shutil.unpack_archive(os.path.join(DXA_data_folder,folder_path), os.path.join(DXA_data_unzip,folder), 'zip')

    images_list = [file for file in os.listdir(os.path.join(DXA_data_unzip,folder)) if file.endswith(".dcm")]
    all_images += len(images_list)
    N_images = 0
    for n, image in enumerate(images_list):
        ds = dicom.dcmread(os.path.join(DXA_data_unzip, folder, image))
        if ds.get(ds[0x0018, 0x1030].keyword) == 'Total Body':
            image_prefix = "Total_Body"
        else:
            continue

        # Get image information from the .dcm
        # print(ds.get(ds[0x0020, 0x000e].keyword))
        ID = ds.get(ds[0x0008, 0x0018].keyword)
        height = float(ds.get(ds[0x0010, 0x1020].keyword))
        Weight = ds.get(ds[0x0010, 0x1030].keyword)
        Ethnic = ds.get(ds[0x0010, 0x2160].keyword)
        Sex = ds.get(ds[0x0010, 0x0040].keyword)
        Age = ds.get(ds[0x0010, 0x1010].keyword)
        Device = ds.get(ds[0x0018, 0x1000].keyword)
        Software = ds.get(ds[0x0018, 0x1020].keyword)
        DeviceSoftware = Device+"_"+Software
        
        try:
            # ValueError: The length of the pixel data in the dataset doesn't match the expected length . 
            # Save pixel array as .png
            cv2.imwrite(os.path.join(out_path_png ,folder+"_"+str(N_images)+".png"), ds.pixel_array)
            file_size = os.path.getsize(os.path.join(out_path_png ,folder+"_"+str(N_images)+".png"))
            file_size_kb = file_size / 1024 
            if file_size_kb < 0:
                os.remove(os.path.join(out_path_png ,folder+"_"+str(N_images)+".png"))
            else:
                # Save pixel array as .npy
                np.save(os.path.join(out_path_npy ,folder+"_"+str(N_images)+".npy"), ds.pixel_array)

                temp = pd.DataFrame({"folder":[folder],"ID":[ID],"height":[height],"Sex":[Sex],"Age":[Age],"Weight":[Weight],"Ethnic":[Ethnic],"DeviceSoftware":[DeviceSoftware],"Image":[folder+"_"+str(N_images)]})

                if len(dcm_table) == 0:
                    dcm_table = temp
                else:
                    dcm_table = pd.concat([dcm_table, temp], ignore_index=True)

                del ds
                N_images += 1
                # time.sleep(0.01)
        except:
            # Print sample ID and image ID for failed reads
            print(folder+"_"+image, flush=True)
            del ds
            continue

# Save statistics table for the .dcm images
dcm_table = dcm_table.sort_values(by="Image", ascending =True)
dcm_table.to_csv("/home/shiwei/work/DXA/dcm_table.tsv",index=False,header=True,sep="\t")
print("num of all images "+str(all_images), flush=True)