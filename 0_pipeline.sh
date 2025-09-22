#!/bin/bash

# --- Step 1: DICOM Format Conversion ---
# Filter for 'Total Body' images and convert original medical images from .dcm format 
# to general-purpose .png and .npy formats, and extract metadata.
#############################################################################
nohup python 1_dcm2img.py > 1_dcm2img.py.log &

# --- Step 2: Whole-Body Skeletal Image Classification ---
# Classification to extract whole-body skeleton DXA images.
#############################################################################
nohup python 2_body_class.py > 2_body_class.py.log &

# --- Step 3: Cropping Quality Classification ---
# Removal of poorly cropped DXA images.
#############################################################################
nohup python 3_crop_class.py > 3_crop_class.py.log &

# --- Step 4: Background Colour Classification ---
# Image background differentiation.
#############################################################################
nohup python 4_background_class.py > 4_background_class.py.log &

# --- Step 5: Contrast Classification ---
# Removal of DXA images with abnormal contrast..
#############################################################################
nohup python 5_contrast_class.py white > 5_contrast_class.py_white.log &
nohup python 5_contrast_class.py black > 5_contrast_class.py_black.log &

# --- Step 6: Head Pose Classification ---
# Classification of head pose.
#############################################################################
nohup python 6_pose_class.py white > 6_pose_class.py_white.log &
nohup python 6_pose_class.py black > 6_pose_class.py_black.log &

# --- Step 7: Create Directories for Segmentation Model Training Data ---
# Create directories to store manually annotated .json files (masks), which are
# the input for training the segmentation model. The user needs to place the
# annotated files into these directories after this step.
#############################################################################
mkdir 7_segment_masked_white_direct
mkdir 7_segment_masked_black_direct

# --- Step 7: Image Segmentation ---
# Image segmentation model to identify head and anatomical landmarks..
#############################################################################
# Define the list of body parts to be segmented
label_list=("head" "shoulder_left" "shoulder_right" "ankle_left" "ankle_right" "hip_left" "hip_right")
# Iterate through each label in the list
for label in "${label_list[@]}"; do
    nohup python 7_segment.py black ${label} > 7_segment_black_${label}.log &
    nohup python 7_segment.py white ${label} > 7_segment_white_${label}.log &
    wait
done

# --- Step 8: Calculate Coordinates from Segmentation Results ---
# Calculate the coordinates of the bounding box of the segmented head and 
# anatomical landmarks.
#############################################################################
nohup python 8_cal_segment.py black > 8_cal_segment.py_black.log &
nohup python 8_cal_segment.py white > 8_cal_segment.py_white.log &
