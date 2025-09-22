## Extract segmentation results

import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# List of labels to process
label_list = ["head", "shoulder_left", "shoulder_right", "ankle_left", "ankle_right", "hip_left", "hip_right"]

bak = str(sys.argv[1]) # black white
threshood = 0
pose = "direct"

os.makedirs("/home/shiwei/work/DXA/8_segment_result", exist_ok=True)
table_path = f"/home/shiwei/work/DXA/8_segment_result/segment_table_{bak}.tsv"
# Load existing results table if available, otherwise load base DICOM table
if not os.path.exists(table_path):
    dcm_table = pd.read_table("/home/shiwei/work/DXA/dcm_table.tsv", sep="\t")
else:
    dcm_table = pd.read_table(table_path, sep="\t")

# Add placeholder columns for bounding boxes of each label if missing
for label in label_list:
    for dim in ['x', 'y', 'w', 'h']:
        col = f"{label}_{dim}"
        if col not in dcm_table.columns:
            dcm_table[col] = -1

# Apply thresholding to segmentation outputs            
def segment_thresh(seg_img, threshold):
    outputs_t= np.where(seg_img > threshold, 1, seg_img)
    outputs_t= np.where(outputs_t < threshold, 0, outputs_t)

    return outputs_t

# Process a single predicted segmentation, compute bounding box
def process_single_image(args):
    label, input_path, input_filename, threshood = args
    try:
        full_path = os.path.join(input_path, input_filename)
        data = np.load(full_path)
        data = segment_thresh(data, threshood)
        image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary_image = cv2.threshold(image.astype(np.uint8), threshood, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
    except:
        x, y, w, h = -1, -1, -1, -1

    return input_filename.split(".")[0], label, x, y, w, h


if __name__ == "__main__":
    # # Loop through all labels and extract bounding boxes
    for label in label_list:
        print(f"Processing label: {label}")

        input_path = f"/home/shiwei/work/DXA/7_segment_predicted_{label}_{bak}/"
        input_list = os.listdir(input_path)

        args_list = [(label, input_path, fname, threshood) for fname in input_list]

        with ProcessPoolExecutor(max_workers=80) as executor:
            results = list(tqdm(executor.map(process_single_image, args_list), total=len(args_list), desc=label))

        # Update main results table
        update_df = pd.DataFrame(results, columns=["Image", "label", "x", "y", "w", "h"])
        update_df[f"{label}_x"] = update_df["x"]
        update_df[f"{label}_y"] = update_df["y"]
        update_df[f"{label}_w"] = update_df["w"]
        update_df[f"{label}_h"] = update_df["h"]
        update_df = update_df[["Image", f"{label}_x", f"{label}_y", f"{label}_w", f"{label}_h"]].set_index("Image")

        dcm_table.set_index("Image", inplace=True)
        dcm_table.update(update_df)
        dcm_table.reset_index(inplace=True)

        # Save updated results
        dcm_table.to_csv(table_path, sep="\t", index=False)
    print("Done.")
