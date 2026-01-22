# nohup python pred_pipeline.py --input_dir=/storageC/shiwei/work/DXA/input_DXA_dir --model_path=/storageC/shiwei/work/DXA/all_models/ > pred_pipeline.py.log
import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import segmentation_models_pytorch as smp
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
BAK_SIZE = (960, 384)

def scale_img(data, transf=False, target_size=(960, 384), transform=None):
    height, width = data.shape[:2]
    pad_height = max(0, target_size[0] - height)
    pad_width = max(0, target_size[1] - width)
    pad_top, pad_bottom = pad_height // 2, pad_height - pad_height // 2
    pad_left, pad_right = pad_width // 2, pad_width - pad_width // 2
    crop_height = min(height, target_size[0])
    crop_width = min(width, target_size[1])
    start_h = (height - crop_height)//2
    start_w = (width - crop_width)//2
    cropped = data[start_h:start_h+crop_height, start_w:start_w+crop_width]
    fill_value = (0,) if len(cropped.shape)==2 else tuple(cropped[0,0])
    padded = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)) if len(cropped.shape)==3 else ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=fill_value)
    if transf and transform is not None:
        return transform(padded)
    return padded

# ------------------------------
# Npy Dataset
# ------------------------------
class NpyDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        fileprefix = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        filename = self.dataframe.iloc[idx, 2]
        data = np.load(os.path.join(self.data_dir, filename))
        image = scale_img(data, transf=True, transform=self.transform)
        return image, label, fileprefix

# ------------------------------
# 背景检测
# ------------------------------
def process_file(filename, input_path):
    npy_path = os.path.join(input_path, "npy", f"{filename}.npy")
    if not os.path.exists(npy_path):
        return filename, None
    data = np.load(npy_path)
    arr = data[:9, :9]
    unique_elements, counts = np.unique(arr, return_counts=True)
    bak_col = unique_elements[np.argmax(counts)]
    if bak_col == 0: return filename, "black"
    elif bak_col == 252: return filename, "white"
    return filename, "other"

def parallel_process(input_list, input_path, max_workers=4):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, f, input_path) for f in input_list]
        for future in as_completed(futures):
            filename, background = future.result()
            results[filename] = background
    return results

# ------------------------------
# 模型预测函数
# ------------------------------
def get_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

def class_pred_model(temp_in_df, model_name, batch_size, npy_dir, model_map):
    torch.cuda.empty_cache()
    model = models.resnet152()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_map[model_name], map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    transform = get_transform()
    dataset = NpyDataset(temp_in_df, npy_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    results = []
    with torch.no_grad():
        for inputs, labels, filenames in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for fname, pred in zip(filenames, predicted.cpu().numpy()):
                results.append((fname, pred))
    return dict(results)

# ------------------------------
# Segment Dataset & 预测
# ------------------------------
class Segmentation_test_Dataset(Dataset):
    def __init__(self, npy_dir, id_list, transform=None):
        """
        npy_dir: npy 文件存放目录
        id_list: 需要读取的 ID 列表
        transform: 可选的 torchvision transform
        """
        self.npy_dir = npy_dir
        self.transform = transform
        self.files = [f"{fid}.npy" for fid in id_list]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        npy_path = os.path.join(self.npy_dir, filename)
        data = np.load(npy_path)  # (H, W) 或 (H, W, C)

        # 如果是灰度 npy，扩展为 (H, W, 1)
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
        
        # scale_img 保持和之前一致
        data = scale_img(data, transf=False)

        # 转 RGB（复制灰度到3通道）
        data_rgb = np.repeat(data, 3, axis=2)  # (H, W, 3)

        if self.transform is not None:
            data_rgb = self.transform(data_rgb)

        return data_rgb, filename

def segment_pred_model(temp_in_df, model_name, batch_size, npy_dir, output_path):
    torch.cuda.empty_cache()

    # 创建模型
    model = smp.Unet(
        encoder_name="resnet152",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(MODEL_MAP[model_name], map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    id_list = temp_in_df.ID.tolist()
    transform = transforms.ToTensor()
    dataset = Segmentation_test_Dataset(npy_dir, id_list, transform=transform)

    # 分块处理
    part_size = 500 if len(temp_in_df) > 10000 else 100
    dataset_len = len(dataset)
    num_parts = max(1, math.ceil(dataset_len / part_size))
    parts = [Subset(dataset, range(i*part_size, min((i+1)*part_size, dataset_len))) for i in range(num_parts)]

    for i, part in enumerate(parts):
        print(f"Segmenting {model_name} part {i+1}/{num_parts}...", flush=True)
        loader = DataLoader(part, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        with torch.no_grad():
            for inputs, filenames in loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)  # [B, 1, H, W] 或 [B, 1, H, W, 1]
                for b in range(outputs.shape[0]):
                    out = outputs[b].permute(1,2,0).cpu().numpy()  # [H, W, 1]
                    np.save(os.path.join(output_path, filenames[b].replace('.npy','')), out)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="""
        DXA HBR Prediction Pipeline

        该脚本用于对 DXA 全身灰度影像进行自动化质量控制、分类、分割，
        并计算头身比（HBR）等体型表型指标。

        【输入要求】
        1. 输入目录 (--input_dir) 必须包含：
           - 所有待预测的 DXA 灰度 PNG 图像（8-bit 单通道）
             * 文件名格式：<ID>.png
             * ID 必须与 input_height.tsv 中第一列一致
           - 一个 input_height.tsv 文件：
             * 制表符分隔（TSV）
             * 无表头
             * 第一列：ID
             * 第二列：身高（单位：cm）

           示例结构：
           input_dir/
           ├── 000001.png
           ├── 000002.png
           ├── ...
           ├── input_height.tsv

        2. PNG 图像要求：
           - 单通道灰度图（grayscale）
           - 背景为纯黑（0）或纯白（252）
           - 图像内容为完整人体 DXA 扫描

        【模型目录要求】
        --model_path 必须包含所有已训练完成的模型权重文件（.pth），
        文件名需与脚本中 MODEL_MAP 定义严格一致。

        示例结构：
        model_path/
        ├── 2_class_model_resnet152.pth
        ├── 3_crop_model_resnet152.pth
        ├── 5_contrast_black_model_resnet152.pth
        ├── 5_contrast_white_model_resnet152.pth
        ├── 6_pose_black_model_resnet152.pth
        ├── 6_pose_white_model_resnet152.pth
        ├── 7_head_black_model.pth
        ├── 7_head_white_model.pth
        ├── 7_shouler_left_black_model.pth
        ├── 7_shouler_left_white_model.pth
        ├── 7_shouler_right_black_model.pth
        ├── 7_shouler_right_white_model.pth
        ├── 7_hip_left_black_model.pth
        ├── 7_hip_left_white_model.pth
        ├── 7_hip_right_black_model.pth
        ├── 7_hip_right_white_model.pth
        ├── 7_ankle_left_black_model.pth
        ├── 7_ankle_left_white_model.pth
        ├── 7_ankle_right_black_model.pth
        └── 7_ankle_right_white_model.pth

        注意：
        - model_path 需以 “/” 结尾，或确保路径拼接正确
        - 所有模型缺一不可，否则流程会报错

        【输出说明】
        脚本运行完成后，将在 input_dir 下生成：
        - npy/                PNG 转换后的中间文件
        - segment/<region>/   各身体部位的分割结果（.npy）
        - output_HBRs.tsv     最终表型结果表

        【运行示例】
        nohup python pred_pipeline.py \\
            --input_dir=/path/to/input_DXA_dir \\
            --model_path=/path/to/all_models/ \\
            --num_workers=4 \\
            > pred_pipeline.log 2>&1 &

        【硬件与环境】
        - 支持 CPU / GPU（自动检测 CUDA）
        - 推荐 GPU 显存 ≥ 12GB
        - Python ≥ 3.8
        """
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="DXA 输入目录，必须包含所有 <ID>.png 灰度影像及 input_height.tsv"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型权重目录，需包含所有 ResNet152 / UNet 的 .pth 文件"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="并行线程数（默认：4）"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    model_path = args.model_path
    NUM_WORKERS = args.num_workers

    MODEL_MAP = {
        "body_classifier": model_path+"2_class_model_resnet152.pth",
        "crop_classifier": model_path+"3_crop_model_resnet152.pth", 
        "contrast_black": model_path+"5_contrast_black_model_resnet152.pth", 
        "contrast_white": model_path+"5_contrast_white_model_resnet152.pth", 
        "pose_black": model_path+"6_pose_black_model_resnet152.pth",
        "pose_white": model_path+"6_pose_white_model_resnet152.pth",
        "segment_unet_head_black": model_path+"7_head_black_model.pth", 
        "segment_unet_head_white": model_path+"7_head_white_model.pth",
        "segment_unet_shoulder_left_black": model_path+"7_shouler_left_black_model.pth",
        "segment_unet_shoulder_left_white": model_path+"7_shouler_left_white_model.pth", 
        "segment_unet_shoulder_right_black": model_path+"7_shouler_right_black_model.pth",
        "segment_unet_shoulder_right_white": model_path+"7_shouler_right_white_model.pth",
        "segment_unet_hip_left_black": model_path+"7_hip_left_black_model.pth",
        "segment_unet_hip_left_white": model_path+"7_hip_left_white_model.pth", 
        "segment_unet_hip_right_black": model_path+"7_hip_right_black_model.pth", 
        "segment_unet_hip_right_white": model_path+"7_hip_right_white_model.pth", 
        "segment_unet_ankle_left_black": model_path+"7_ankle_left_black_model.pth",
        "segment_unet_ankle_left_white": model_path+"7_ankle_left_white_model.pth", 
        "segment_unet_ankle_right_black": model_path+"7_ankle_right_black_model.pth",
        "segment_unet_ankle_right_white": model_path+"7_ankle_right_white_model.pth",
    }

    print(f"Running pipeline with input_dir={input_dir}, model_path={model_path}, NUM_WORKERS={NUM_WORKERS}", flush=True)

    # ------------------------------
    # 1. 生成 npy 文件
    # ------------------------------
    npy_dir = os.path.join(input_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)

    input_height_path = os.path.join(input_dir, "input_height.tsv")
    input_df = pd.read_csv(input_height_path, sep="\t", header=None, names=["ID","Height"])

    for _, row in input_df.iterrows():
        id_ = row["ID"]
        png_path = os.path.join(input_dir, f"{id_}.png")
        npy_path = os.path.join(npy_dir, f"{id_}.npy")
        if os.path.exists(png_path) and not os.path.exists(npy_path):
            img = Image.open(png_path)
            np.save(npy_path, img)

    output_df = input_df.copy()
    output_df["npy"] = output_df.ID + ".npy"

    # ------------------------------
    # 2. 背景检测
    # ------------------------------
    background_dict = parallel_process(output_df["ID"].tolist(), input_dir)
    output_df["Background"] = output_df["ID"].map(background_dict)

    # ------------------------------
    # 3. 分类预测
    # ------------------------------
    temp_in_df = output_df

    # body_classifier
    id2value = class_pred_model(temp_in_df, "body_classifier", batch_size=8, npy_dir=npy_dir, model_map=MODEL_MAP)
    output_df["Bone_class"] = output_df["ID"].map(id2value)

    # crop_classifier
    temp_in_df = output_df.loc[output_df.Bone_class == 1].reset_index(drop=True)
    id2value = class_pred_model(temp_in_df, "crop_classifier", batch_size=8, npy_dir=npy_dir, model_map=MODEL_MAP)
    output_df["Crop_class"] = output_df["ID"].map(id2value)

    # contrast_black/white
    temp_in_df = output_df.loc[(output_df.Crop_class == 0) & (output_df.Background == "black")]
    id2value_black = class_pred_model(temp_in_df, "contrast_black", batch_size=4, npy_dir=npy_dir, model_map=MODEL_MAP) if len(temp_in_df) > 0 else {}
    temp_in_df = output_df.loc[(output_df.Crop_class == 0) & (output_df.Background == "white")]
    id2value_white = class_pred_model(temp_in_df, "contrast_white", batch_size=4, npy_dir=npy_dir, model_map=MODEL_MAP) if len(temp_in_df) > 0 else {}
    output_df["Contrast_class"] = output_df["ID"].map(id2value_black | id2value_white)

    # pose_black/white 
    temp_in_df = output_df.loc[(output_df.Contrast_class == 0) & (output_df.Background == "black")]
    id2value_black = class_pred_model(temp_in_df, "pose_black", batch_size=4, npy_dir=npy_dir, model_map=MODEL_MAP) if len(temp_in_df) > 0 else {}
    temp_in_df = output_df.loc[(output_df.Contrast_class == 0) & (output_df.Background == "white")]
    id2value_white = class_pred_model(temp_in_df, "pose_white", batch_size=4, npy_dir=npy_dir, model_map=MODEL_MAP) if len(temp_in_df) > 0 else {}
    output_df["Pose_class"] = output_df["ID"].map(id2value_black | id2value_white)

    # ------------------------------
    # 4. 分割预测
    # ------------------------------
    regions = ["head","shoulder_left","shoulder_right","ankle_left","ankle_right","hip_left","hip_right"]

    temp_in_black = output_df.loc[(output_df.Pose_class == 0) & (output_df.Background == "black")].reset_index(drop=True)
    temp_in_white = output_df.loc[(output_df.Pose_class == 0) & (output_df.Background == "white")].reset_index(drop=True)

    for region in regions:
        if len(temp_in_black) > 0:
            seg_dir = os.path.join(input_dir, "segment", region)
            os.makedirs(seg_dir, exist_ok=True)
            segment_pred_model(temp_in_black, f"segment_unet_{region}_black", batch_size=4, npy_dir=npy_dir, output_path=seg_dir)
        if len(temp_in_white) > 0:
            seg_dir = os.path.join(input_dir, "segment", region)
            os.makedirs(seg_dir, exist_ok=True)
            segment_pred_model(temp_in_white, f"segment_unet_{region}_white", batch_size=4, npy_dir=npy_dir, output_path=seg_dir)

    # ------------------------------
    # 5. 计算身体测量值和头身比表型
    # ------------------------------
    def segment_thresh(seg_img, threshold):
        outputs_t= np.where(seg_img > threshold, 1, seg_img)
        outputs_t= np.where(outputs_t < threshold, 0, outputs_t)
        return outputs_t

    def get_height_length(ankle_right_y, ankle_left_y, ankle_right_h,ankle_left_h, head_y ):
        return ( (ankle_right_y+(ankle_right_h/2)) + (ankle_left_y+(ankle_left_h/2)) )/2 - head_y

    def get_shoulder_width(shoulder_right_x, shoulder_right_w, shoulder_left_x, shoulder_left_w, shoulder_right_y, shoulder_right_h, shoulder_left_y, shoulder_left_h, pixel_length):
        right_x = shoulder_right_x + (shoulder_right_w/2)
        left_x = shoulder_left_x - (shoulder_left_w/2)
        right_y = shoulder_right_y + (shoulder_right_h/2)
        left_y = shoulder_left_y + (shoulder_left_h/2)
        return math.sqrt( (right_x - left_x) ** 2 + (right_y - left_y) ** 2 ) * pixel_length

    def get_hip_width(hip_right_x, hip_right_w, hip_left_x, hip_left_w, hip_right_y, hip_right_h, hip_left_y, hip_left_h, pixel_length):
        right_x = hip_right_x + (hip_right_w/2)
        left_x = hip_left_x - (hip_left_w/2)
        right_y = hip_right_y + (hip_right_h/2)
        left_y = hip_left_y + (hip_left_h/2)
        return math.sqrt( (right_x - left_x) ** 2 + (right_y - left_y) ** 2 ) * pixel_length

    threshold = 0.5

    for pred in output_df["npy"]:
        for region in regions:
            npy_path = os.path.join(input_dir, "segment", region, pred)

            if not os.path.exists(npy_path):
                x, y, w, h = -1, -1, -1, -1
            else:
                data = np.load(npy_path)
                data = segment_thresh(data, threshold)
                image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

                _, binary_image = cv2.threshold(image.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                else:
                    x, y, w, h = -1, -1, -1, -1

            for col, val in zip([region+"_x", region+"_y", region+"_w", region+"_h"], [x, y, w, h]):
                output_df.loc[output_df["npy"] == pred, col] = val

    print("All segment result",len(output_df))
    print("Failed head: ", (output_df.head_x == -1).value_counts().get(True))
    print("Failed shoulder_left:", (output_df.shoulder_left_x == -1).value_counts().get(True))
    print("Failed shoulder_right:", (output_df.shoulder_right_x == -1).value_counts().get(True))
    print("Failed ankle_left:", (output_df.ankle_left_x == -1).value_counts().get(True))
    print("Failed ankle_right:", (output_df.ankle_right_y == -1).value_counts().get(True))
    print("Failed hip_left:", (output_df.hip_left_x == -1).value_counts().get(True))
    print("Failed hip_right:", (output_df.hip_right_y == -1).value_counts().get(True))

    output_df["height_length"] = output_df.apply(lambda row: get_height_length(row.ankle_right_y, row.ankle_left_y, row.ankle_right_h,row.ankle_left_h, row.head_y ),axis=1)
    output_df["pixel_length"] = output_df.Height / output_df.height_length # 身高/像素长度

    output_df["head_width"] = output_df.head_w * output_df.pixel_length
    output_df["head_length"] = output_df.head_h * output_df.pixel_length

    output_df["shoulder_width"] = output_df.apply(lambda row: get_shoulder_width(row.shoulder_right_x, row.shoulder_right_w, row.shoulder_left_x, row.shoulder_left_w, 
                                                                                 row.shoulder_right_y, row.shoulder_right_h, row.shoulder_left_y, row.shoulder_left_h, row.pixel_length),axis=1)

    output_df["hip_width"] = output_df.apply(lambda row: get_hip_width(row.hip_right_x, row.hip_right_w, row.hip_left_x, row.hip_left_w, 
                                                                       row.hip_right_y, row.hip_right_h, row.hip_left_y, row.hip_left_h, row.pixel_length),axis=1)

    output_df["trunk_length"] = (((output_df.hip_right_y + (output_df.hip_right_h/2) +  output_df.hip_left_y + (output_df.hip_left_h/2) ) /2) - (
        (output_df.shoulder_right_y + (output_df.shoulder_right_h/2) +  output_df.shoulder_left_y + (output_df.shoulder_left_h/2) ) /2) ) * output_df.pixel_length

    output_df["leg_length"]= (((output_df.ankle_right_y + (output_df.ankle_right_h/2) +  output_df.ankle_left_y + (output_df.ankle_left_h/2) ) /2) - (
        (output_df.hip_right_y + (output_df.hip_right_h/2) +  output_df.hip_left_y + (output_df.hip_left_h/2) ) /2) ) * output_df.pixel_length


    output_df["trunk_left_length"] = ((output_df.hip_left_y + output_df.hip_left_h/2) - (output_df.shoulder_left_y + output_df.shoulder_left_h/2 ) ) * output_df.pixel_length
    output_df["trunk_right_length"] = ((output_df.hip_right_y + output_df.hip_right_h/2) - (output_df.shoulder_right_y + output_df.shoulder_right_h/2 ) ) * output_df.pixel_length

    output_df["leg_left_length"]= ((output_df.ankle_left_y + output_df.ankle_left_h/2) - ( output_df.hip_left_y + output_df.hip_left_h/2) ) * output_df.pixel_length
    output_df["leg_right_length"]= ((output_df.ankle_right_y + output_df.ankle_right_h/2) - ( output_df.hip_right_y + output_df.hip_right_h/2) ) * output_df.pixel_length

    output_df["LHR"] = output_df.head_length / output_df.Height # 头长, 身高
    output_df["WHR"] = output_df.head_width / output_df.Height # 头宽, 身高
    # output_df["SHR"] = output_df.shoulder_width / output_df.Height # 肩宽, 身高
    output_df["LSR"] = output_df.head_length / output_df.shoulder_width # 头长，肩宽
    output_df["WSR"] = output_df.head_width / output_df.shoulder_width # 头宽，肩宽
    #output_df["LWR"] = output_df.head_length / output_df.head_width # 头长/头宽

    output_df["LTR"] = output_df.head_length / output_df.trunk_length # 头长，躯干长
    output_df["WTR"] = output_df.head_width / output_df.trunk_length # 头宽，躯干长
    output_df["LLeR"] = output_df.head_length / output_df.leg_length # 头长，腿长
    output_df["WLeR"] = output_df.head_width / output_df.leg_length # 头宽，腿长
    output_df["LHiR"] = output_df.head_length / output_df.hip_width # 头长，臀宽
    output_df["WHiR"] = output_df.head_width / output_df.hip_width # 头宽，臀宽

    # ------------------------------
    # 6. 保存初步输出
    # ------------------------------
    output_df.to_csv(os.path.join(input_dir, "output_HBRs.tsv"), sep="\t", index=False)
    print("Pipeline finished. Output saved to output_HBRs.tsv")
