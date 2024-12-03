# Image Processing and Segmentation Pipeline for head-body-ratio(HBR)

This repository contains a collection of Python scripts for image processing and segmentation, focused on tasks such as background extraction, contrast enhancement, head pose classification, and image segmentation. The project also includes a pipeline script to execute the entire process.

## Files Overview

- **0_pipeline.sh**: A shell script that runs the entire image processing pipeline by executing all the required Python scripts in sequence.
- **1_dcm2img.py**: Converts DICOM images to a suitable image format for processing.
- **2_body_class.py**: Classification to extract whole body skeleton DXA images.
- **3_crop_class.py**: Removal of poorly cropped x-rays.
- **4_background_class.py**: Image standardization and background differentiation.
- **5_contrast_class.py**: Removal of x-rays with abnormal contrast.
- **6_pose_class.py**: Classification of head pose.
- **7_segment.py**: Deep learning-based image segmentation model to identify head and landmarks.
- **8_cal_segment.py**: Calculate the coordinates of the bounding box of the segmented head and the landmarks.
- **environment.yaml**: Contains the environment dependencies required to run the scripts.

## Setup

1. Clone the repository or download the files.
2. Set up the environment using the provided `environment.yaml`. You can create a virtual environment and install the necessary dependencies using:

   ```bash
   conda env create -f environment.yaml
