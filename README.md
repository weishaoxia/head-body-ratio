# Image Processing and Segmentation Pipeline for head-body-ratio(HBR)

This repository contains a collection of Python scripts for image processing and segmentation, focused on tasks such as background extraction, contrast enhancement, body and pose analysis, and image segmentation. The project also includes a pipeline script to execute the entire process.

## Files Overview

- **0_pipeline.sh**: A shell script that runs the entire image processing pipeline by executing all the required Python scripts in sequence.
- **1_dcm2img.py**: Converts DICOM images to a suitable image format for processing.
- **2_body_class.py**: Processes body-related features or performs body segmentation.
- **3_crop_class.py**: Crops the image based on predefined parameters.
- **4_background_class.py**: Extracts and processes the background from images.
- **5_contrast_class.py**: Enhances or adjusts the contrast of the images.
- **6_pose_class.py**: Analyzes and processes body pose information.
- **7_segment.py**: Performs image segmentation to separate different regions in the image.
- **8_cal_segment.py**: Calibration script for the segmentation process to improve accuracy.
- **environment.yaml**: Contains the environment dependencies required to run the scripts, including specific Python packages.

## Setup

1. Clone the repository or download the files.
2. Set up the environment using the provided `environment.yaml`. You can create a virtual environment and install the necessary dependencies using:

   ```bash
   conda env create -f environment.yaml
