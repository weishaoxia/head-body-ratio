#!/bin/bash

#############################################################################
nohup python 1_dcm2img.py > 1_dcm2img.py.log &

#############################################################################
nohup python 2_body_class.py > 2_body_class.py.log &

#############################################################################
nohup python 3_crop_class.py > 3_crop_class.py.log &

#############################################################################
nohup python 4_background_class.py > 4_background_class.py.log &

#############################################################################
nohup python 5_contrast_class.py white > 5_contrast_class.py_white.log &
nohup python 5_contrast_class.py black > 5_contrast_class.py_black.log &

#############################################################################
nohup python 6_pose_class.py white > 6_pose_class.py_white.log &
nohup python 6_pose_class.py black > 6_pose_class.py_black.log &

#############################################################################
mkdir 7_segment_masked_white_direct
mkdir 7_segment_masked_black_direct
cp 6_pose_predicted_white_png/*.json 7_segment_masked_white_direct
cp 6_pose_predicted_black_png/*.json 7_segment_masked_black_direct

#############################################################################
# copy exsit mask to 6_pose_predicted_white_png and 6_pose_predicted_black_png
nohup sh 7_segment_model_black_test.sh > 7_segment_model_black_test.sh.log &
nohup sh 7_segment_model_white_test.sh > 7_segment_model_white_test.sh.log &

#############################################################################
# set best runs in 8_cal_segment.py
nohup python 8_cal_segment.py black resnet152 resnet152_test > 8_cal_segment.py_black_resnet152.log &
nohup python 8_cal_segment.py white resnet152 resnet152_test > 8_cal_segment.py_white_resnet152.log &
