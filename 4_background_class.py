##
# Distinguish between white and black backgrounds

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from threading import Lock

# Set random seed for reproducibility
def copy_file(source_path, target_path):
    if os.path.exists(source_path):
        if not os.path.exists(target_path):
            shutil.copyfile(source_path, target_path)
    else:
        print(f"File {os.path.basename(source_path)} not found in {os.path.dirname(source_path)}")

# Parallel copy for output
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

bak_size = (960, 384)

input_path = "/home/shiwei/work/DXA"
input_list = os.listdir(os.path.join(input_path, "3_crop_predicted_npy"))

print(len(input_list), flush=True)
# Global lock for writing shared lists
lock = Lock()

bak_black_list = []
bak_white_list = []
bak_os_list = [] # oversize
unexpected_list = []
less_size_list = []

# Used for batch background color determination
def process_file(filename, input_path, bak_size):
    png_path = os.path.join(input_path, "3_crop_predicted_png", filename.replace('.npy', '.png'))
    npy_path = os.path.join(input_path, "3_crop_predicted_npy", filename)

    file_size = os.path.getsize(png_path)
    file_size_kb = file_size / 1024

    # Remove undersized images
    if file_size_kb < 40:
        less_size_list.append(filename)

    data = np.load(npy_path)
    height, width = data.shape[:2]


    arr = data[0:9, 0:9] # Top-left pixel block
    unique_elements, counts = np.unique(arr, return_counts=True)
    mode_index = np.argmax(counts)
    bak_col = unique_elements[mode_index]

    with lock:
        if (height > bak_size[0]) or (width > bak_size[1]):
            bak_os_list.append(filename)
        else:
            if bak_col == 0:
                    bak_black_list.append(filename)
            elif bak_col == 252:
                    bak_white_list.append(filename)
            else:
                print(f"{filename} → Unexpected background color: {bak_col}", flush=True)
                unexpected_list.append(filename)
# Batch processing
def parallel_process(input_list, input_path, bak_size, max_workers=16):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, filename, input_path, bak_size)
            for filename in input_list
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            pass
parallel_process(
    input_list=input_list,
    input_path=input_path,
    bak_size=bak_size,
    max_workers=24
)


print(len(bak_black_list), len(bak_white_list), flush=True)
print("Over image size ", len(bak_os_list), flush=True)
print("Less file size", len(less_size_list), flush=True)
print("Unexpected background color ", unexpected_list, flush=True)

# output
copy_pres_parallel("/home/shiwei/work/DXA/3_crop_predicted_npy", "/home/shiwei/work/DXA/4_bak_black_npy", bak_black_list, max_workers=32)
copy_pres_parallel("/home/shiwei/work/DXA/3_crop_predicted_npy", "/home/shiwei/work/DXA/4_bak_white_npy", bak_white_list, max_workers=32)

copy_pres_parallel("/home/shiwei/work/DXA/3_crop_predicted_png", "/home/shiwei/work/DXA/4_bak_black_png", [item.replace('.npy', '.png') for item in bak_black_list], max_workers=32)
copy_pres_parallel("/home/shiwei/work/DXA/3_crop_predicted_png", "/home/shiwei/work/DXA/4_bak_white_png", [item.replace('.npy', '.png') for item in bak_white_list], max_workers=32)