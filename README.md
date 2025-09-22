
---

# **Image Processing and Segmentation Pipeline for head-body-ratio(HBR)**

This repository contains a collection of Python scripts that form a comprehensive pipeline for processing Dual-energy X-ray absorptiometry (DXA) images. The pipeline starts with raw DICOM files and performs a series of filtering, focused classification, and segmentation steps to identify human head and key anatomical landmarks. The project also includes a pipeline script to execute the entire process.

---

## Setup

1. Clone the repository or download the files.
2. Set up the environment using the provided `environment.yaml`. You can create a virtual environment and install the necessary dependencies using:

   ```bash
   conda env create -f environment.yaml
   ```

---

## **Files Overview**

This section details the purpose, inputs, and outputs of each script in the pipeline.

- **0_pipeline.sh**: A shell script that runs the entire image processing pipeline by executing all the required Python scripts in sequence.
- **1_dcm2img.py**: Convert original medical images from .dcm format to general-purpose .png and .npy formats, and extract metadata.
- **2_body_class.py**: Classification to extract whole body skeleton DXA images.
- **3_crop_class.py**: Removal of poorly cropped DXA images.
- **4_background_class.py**: Image background differentiation.
- **5_contrast_class.py**: Removal of DXA images with abnormal contrast.
- **6_pose_class.py**: Classification of head pose.
- **7_segment.py**: Image segmentation model to identify head and anatomical landmarks.
- **8_cal_segment.py**: Calculate the coordinates of the bounding box of the segmented head and anatomical landmarks.
- **environment.yaml**: Contains the environment dependencies required to run the scripts.


---

### **0\_pipeline.sh**

A shell script that orchestrates the entire workflow by executing the Python scripts in the correct sequence. It handles the flow of data from one step to the next and manages logging for each process.

* **Functionality**: Executes scripts 1 through 8 in order. It runs steps 5, 6, 7, and 8 separately for images with 'white' and 'black' backgrounds.  

---

### **1\_dcm2img.py**

Converts medical DICOM images into standard image and data formats for processing.

* **Functionality**: Unzips archives of DICOM (.dcm) files, extracts patient metadata (ID, sex, age, etc.) and pixel data. It filters for 'Total Body' images and saves them as both .png and .npy files..  
* **Input**:  
  * **Directory**: A folder containing zipped patient data. Each .zip file contains one or more .dcm files.  
* **Output**:  
  * **Directories**:  
    * 1\_Total\_Body\_npy: Contains raw pixel data saved as NumPy arrays (.npy).  
    * 1\_Total\_Body\_png: Contains the same images saved in PNG format.  
  * **File**: dcm\_table.tsv, a tab-separated file containing the extracted metadata for every processed image.

---

### **2\_body\_class.py**

A deep learning classifier to extract whole-body skeletal DXA images.

* **Functionality**: Trains and applies a ResNet152 model to perform binary classification. It filters out images that are not whole-body skeletal.  
* **Input**:  
  * **Training Data**: Manually labeled images specified in 2\_class\_label.tsv. Label=1 indicates a whole-body skeletal image.  
  * **Prediction Data**: All images from 1\_Total\_Body\_npy/.  
* **Output**:  
  * **Model**: A trained PyTorch model file (2\_class\_models/best\_model\_resnet152.pth).  
  * **Directories**:  
    * 2\_class\_predicted\_npy: Contains .npy files classified as whole-body skeletal images.  
    * 2\_class\_predicted\_png: Contains .png files classified as whole-body skeletal images.

---

### **3\_crop\_class.py**

A classifier to remove whole-body skeletal images that are poorly framed or where parts of the body are cut off.

* **Functionality**: Trains and applies a ResNet152 model to identify and discard improperly cropped images.  
* **Input**:  
  * **Training Data**: Manually labeled images specified in 3\_crop\_label.tsv. Label=0 indicates a correctly framed image.  
  * **Prediction Data**: All images from 2\_class\_predicted\_npy/.  
* **Output**:  
  * **Model**: A trained PyTorch model file (3\_crop\_models/best\_model\_resnet152.pth).  
  * **Directories**:  
    * 3\_crop\_predicted\_npy/: Contains complete-limb .npy images.  
    * 3\_crop\_predicted\_png/: Contains complete-limb .png images.

---

### **4\_background\_class.py**

A non-ML script to standardize images by differentiating their background color.

* **Functionality**: Examines the pixels in each image to determine if the background is black (pixel value \~0) or white (pixel value \~252). It then sorts the files accordingly.  
* **Input**: Images from 3\_crop\_predicted\_npy/ and 3\_crop\_predicted\_png/.  
* **Output**:  
  * **Directories**:  
    * 4\_bak\_black\_npy/ & 4\_bak\_black\_png/: Images with a black background.  
    * 4\_bak\_white\_npy/ & 4\_bak\_white\_png/: Images with a white background.

---

### **5\_contrast\_class.py**

A classifier to remove images with abnormal contrast, which can hinder analysis.

* **Functionality**: Trains and applies a ResNet152 model to identify images with normal contrast. The script is run twice, once for black-background images and once for white-background images.  
* **Input**:  
  * **Argument**: black or white.  
  * **Training Data**: Labeled images from 5\_contrast\_black\_label.tsv or 5\_contrast\_white\_label.tsv. Label=0 indicates normal contrast.  
  * **Prediction Data**: Images from 4\_bak\_black\_npy/ or 4\_bak\_white\_npy/.  
* **Output**:  
  * **Models**: Separate trained models for black and white backgrounds.  
  * **Directories**: Filtered images with normal contrast (e.g., 5\_contrast\_black\_predicted\_npy/).

---

### **6\_pose\_class.py**

A classifier to select images where the participant's head is in a standard, front-facing posture.

* **Functionality**: Trains and applies a ResNet152 model to classify head posture. It filters out images with non-standard poses. This script is also run separately for black and white backgrounds.  
* **Input**:  
  * **Argument**: black or white.  
  * **Training Data**: Labeled images from 6\_pose\_black\_label.tsv or 6\_pose\_white\_label.tsv. Label=0 indicates correct posture.  
  * **Prediction Data**: Images from the previous contrast-filtering step.  
* **Output**:  
  * **Models**: Separate trained models for black and white backgrounds.  
  * **Directories**: The final set of quality-controlled images ready for segmentation (e.g., 6\_pose\_black\_predicted\_npy/).

---

### **7\_segment.py**

A deep learning model to perform semantic segmentation on the cleaned images to identify anatomical regions.

* **Functionality**: Trains and applies a U-Net model (with a ResNet152 encoder) to segment specific body parts: head, shoulder\_left, shoulder\_right, ankle\_left, ankle\_right, hip\_left, and hip\_right. The script is run in a loop for each landmark and for each background color.  
* **Input**:  
  * **Arguments**: black or white, and a landmark name (e.g., head).  
  * **Training Data**: Manual annotations in LabelMe (.json) format. The script converts these into binary mask images for training.  
  * **Prediction Data**: The final filtered images from 6\_pose\_...\_predicted\_png/.  
* **Output**:  
  * **Models**: A separate trained U-Net model for each landmark and background color.  
  * **Directories**: For each landmark, a directory containing .npy files that are segmentation masks (probability maps) for each input image (e.g., 7\_segment\_predicted\_head\_black/).

---

### **8\_cal\_segment.py**

Calculates the bounding box coordinates for each segmented landmark.

* **Functionality**: Loads the segmentation masks (.npy files) produced by the previous step, applies a threshold to create a binary mask, finds the largest contour (the landmark), and computes its bounding box (x, y, width, height).  
* **Input**:  
  * **Argument**: black or white.  
  * **Prediction Data**: All segmentation mask directories from the previous step (e.g., 7\_segment\_predicted\_head\_black/).  
* **Output**:  
  * **File**: A final, comprehensive data table named segment\_table\_{bak}.tsv. This file contains all the original metadata from dcm\_table.tsv plus new columns for the bounding box coordinates of every landmark (e.g., head\_x, head\_y, head\_w, head\_h).

---
## **Input Data Formats**

For the classification and segmentation models to be trained, data must be provided in specific formats.

### **Label Files for Classification (.tsv)**

The classification scripts (2\_body\_class.py, 3\_crop\_class.py, etc.) require a tab-separated values file for training. These files (e.g., 2\_class\_label.tsv, 3\_crop\_label.tsv) must have the following three columns:

1. **File\_Name**: The unique identifier for the image, without any file extension.  
2. **Label**: The ground-truth class, typically 0 for the desired category and 1 for the category to be excluded (or vice-versa depending on the script).  
3. **File\_Name\_npy**: The full filename of the corresponding NumPy array file (e.g., image1.npy).


**Example (2\_class\_label.tsv):**


```tsv
File_Name	Label	File_Name_npy  
image1	0	image1.npy  
image2	1	image2.npy  
image3	0	image3.npy
```

### **Annotation Files for Segmentation (.json)**

The segmentation script (7\_segment.py) requires manual annotations to be created using a tool like [LabelMe](https://github.com/labelmeai/labelme). The annotations should be saved as .json files and placed in the appropriate input directory (e.g., 7\_segment\_masked\_white\_direct/in/). The script will automatically parse these files to create training masks. Landmarks like the head should be annotated as polygons, while point landmarks (shoulders, ankles, hips) are annotated as points.

---
