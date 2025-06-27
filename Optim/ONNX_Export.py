from pathlib import Path
import open3d as o3d
import os
import sys

sys.path.append(str(Path.cwd().parent))

from pytorch_lightning import seed_everything
from src.model_utils import Model, Model_internal
from src.mvdream_utils import load_mvdream_model
import argparse
from PIL import Image


import numpy as np
import seaborn as sns
import pandas as pd

import time
import torch
import os
import onnx
import onnxscript

os.environ["XFORMERS_DISABLED"] = "1"

scale = 1.3
diffusion_rescale_timestep = 8


### Single_View
MODEL_CONFIG_URI_SV = "s3://dream-shape-output-2/Wave_Geometry_Net_all_Wavelet_General_Encoder_Down_2_Wavelet_General_Decoder_Up_2_original_4_1024_1_0.25_256_bior6.8_constant_2_2_2_e_r_0_d_r_0_ema_True_all_batched_threshold_use_sample_training_bf16_1.0_1/filter_single_image_udit_1152_32_16/args.json"
MODEL_CHECKPOINT_URI_SV = "s3://dream-shape-output-2/Wave_Geometry_Net_all_Wavelet_General_Encoder_Down_2_Wavelet_General_Decoder_Up_2_original_4_1024_1_0.25_256_bior6.8_constant_2_2_2_e_r_0_d_r_0_ema_True_all_batched_threshold_use_sample_training_bf16_1.0_1/filter_single_image_udit_1152_32_16/checkpoints/step=step=3250000.ckpt"


#### Multi_View Depth
MODEL_CONFIG_URI_MV_depth = "s3://dream-shape-output-2/Wave_Geometry_Net_all_Wavelet_General_Encoder_Down_2_Wavelet_General_Decoder_Up_2_original_4_1024_1_0.25_256_bior6.8_constant_2_2_2_e_r_0_d_r_0_ema_True_all_batched_threshold_use_sample_training_bf16_1.0_1/filter_6_depth_udit_1152_32_16/args.json"
MODEL_CHECKPOINT_URI_MV_depth = "s3://dream-shape-output-2/Wave_Geometry_Net_all_Wavelet_General_Encoder_Down_2_Wavelet_General_Decoder_Up_2_original_4_1024_1_0.25_256_bior6.8_constant_2_2_2_e_r_0_d_r_0_ema_True_all_batched_threshold_use_sample_training_bf16_1.0_1/filter_6_depth_udit_1152_32_16/checkpoints/step=step=2400000.ckpt"

#### Multi_View RGB 
MODEL_CONFIG_URI_MV_RGB = "s3://dream-shape-output-2/Wave_Geometry_Net_all_Wavelet_General_Encoder_Down_2_Wavelet_General_Decoder_Up_2_original_4_1024_1_0.25_256_bior6.8_constant_2_2_2_e_r_0_d_r_0_ema_True_all_batched_threshold_use_sample_training_bf16_1.0_1/filter_multiview_udit_1152_32_16/args.json"
MODEL_CHECKPOINT_URI_MV_RGB = "s3://dream-shape-output-2/Wave_Geometry_Net_all_Wavelet_General_Encoder_Down_2_Wavelet_General_Decoder_Up_2_original_4_1024_1_0.25_256_bior6.8_constant_2_2_2_e_r_0_d_r_0_ema_True_all_batched_threshold_use_sample_training_bf16_1.0_1/filter_multiview_udit_1152_32_16/checkpoints/step=step=2840000.ckpt"


### Voxel
MODEL_CONFIG_URI_Voxel = "s3://dream-shape-output-2/Wave_Geometry_Net_all_Wavelet_General_Encoder_Down_2_Wavelet_General_Decoder_Up_2_original_4_1024_1_0.25_256_bior6.8_constant_2_2_2_e_r_0_d_r_0_ema_True_all_batched_threshold_use_sample_training_bf16_1.0_1/filter_voxel_udit_1152_30_16/args.json"
MODEL_CHECKPOINT_URI_Voxel = "s3://dream-shape-output-2/Wave_Geometry_Net_all_Wavelet_General_Encoder_Down_2_Wavelet_General_Decoder_Up_2_original_4_1024_1_0.25_256_bior6.8_constant_2_2_2_e_r_0_d_r_0_ema_True_all_batched_threshold_use_sample_training_bf16_1.0_1/filter_voxel_udit_1152_30_16/checkpoints/step=step=3100000.ckpt"


### Point Cloud
MODEL_CONFIG_URI_pointcloud = "s3://dream-shape-output-2/Wave_Geometry_Net_all_Wavelet_General_Encoder_Down_2_Wavelet_General_Decoder_Up_2_original_4_1024_1_0.25_256_bior6.8_constant_2_2_2_e_r_0_d_r_0_ema_True_all_batched_threshold_use_sample_training_bf16_1.0_1/filter_pc_udit_1152_32_16/args.json"
MODEL_CHECKPOINT_URI_pointcloud = "s3://dream-shape-output-2/Wave_Geometry_Net_all_Wavelet_General_Encoder_Down_2_Wavelet_General_Decoder_Up_2_original_4_1024_1_0.25_256_bior6.8_constant_2_2_2_e_r_0_d_r_0_ema_True_all_batched_threshold_use_sample_training_bf16_1.0_1/filter_pc_udit_1152_32_16/checkpoints/step=step=2600000.ckpt"


print(f"Loading model")


#model = Model.from_pretrained(pretrained_model_name_or_path=model_name)
model = Model_internal.from_pretrained(model_config_uri=MODEL_CONFIG_URI_pointcloud,model_checkpoint_uri=MODEL_CHECKPOINT_URI_pointcloud)

def recursively_unwrap_orig_mod(module):
    """
    Recursively unwraps all submodules that have an _orig_mod attribute (from torch.compile).
    """
    for name, child in module.named_children():
        if hasattr(child, "_orig_mod"):
            setattr(module, name, child._orig_mod)
            recursively_unwrap_orig_mod(getattr(module, name))
        else:
            recursively_unwrap_orig_mod(child)


# Unwrap all compiled submodules
recursively_unwrap_orig_mod(model)




model.set_inference_fusion_params(
        scale, diffusion_rescale_timestep
    )

pointcloud = True
voxels = False
sv = False
mv = False

if sv:
    data_onnx = {
        'images': torch.zeros((1, 3, 224, 224)),  # Dummy tensor, for multiview add extra dimension after 1, example 4 image view (1,4,3,224,224)
        'img_idx': torch.tensor([0], device='cuda:0'),
        'low': torch.zeros((1, 1, 46, 46, 46), device='cuda:0'),  # Dummy tensor
        'id': 'dummy'
    }
    data_idx = 0
if mv:
    data_onnx = {
        'images': torch.zeros((1, 4, 3, 224, 224)),  # Dummy tensor, for multiview add extra dimension after 1, example 4 image view (1,4,3,224,224)
        'img_idx': torch.tensor([0], device='cuda:0'),
        'low': torch.zeros((1, 1, 46, 46, 46), device='cuda:0'),  # Dummy tensor
        'id': 'dummy'
    }
    data_idx = 0

if voxels:
    data_onnx = {
        'voxels': torch.zeros((1, 1, 16, 16, 16)),  # Dummy tensor for voxel
        'low': torch.zeros((1, 1, 46, 46, 46), device='cuda:0'),  # Dummy tensor
        'id': 'dummy'
    }
    data_idx = 0

if pointcloud:
    data_onnx = {
        'Pointcloud': torch.zeros((1, 25000, 3)),  # Dummy tensor for point cloud
        'low': torch.zeros((1, 1, 46, 46, 46), device='cuda:0'),  # Dummy tensor
        'id': 'dummy'
    }
    data_idx = 0


torch.onnx.export(
    model,
    (data_onnx, data_idx),
    "model_pointcloud.onnx",
    export_params=True,
    opset_version=19,
    do_constant_folding=True,
    input_names=['data', 'data_idx'],
    output_names=['low_pred', 'highs_pred'],
    verbose = True, 
    dynamo=True)

# Validate the exported model
onnx_model = onnx.load("model_pointcloud.onnx")
onnx.checker.check_model("model_pointcloud.onnx")
print("✅ ONNX model exported and validated successfully")