from pathlib import Path
import open3d as o3d
import os
import sys

sys.path.append(str(Path.cwd().parent))

from pytorch_lightning import seed_everything
from src.model_utils import Model
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



model_name = 'ADSKAILab/WaLa-SV-1B'
scale = 1.8
diffusion_rescale_timestep = 5

MODEL_CONFIG_URI = "s3://dream-shape-output-2/Wave_Geometry_Net_all_Wavelet_General_Encoder_Down_2_Wavelet_General_Decoder_Up_2_original_4_1024_1_0.25_256_bior6.8_constant_2_2_2_e_r_0_d_r_0_ema_True_all_batched_threshold_use_sample_training_bf16_1.0_1/filter_single_image_udit_1152_32_16/args.json"
MODEL_CHECKPOINT_URI = "s3://dream-shape-output-2/Wave_Geometry_Net_all_Wavelet_General_Encoder_Down_2_Wavelet_General_Decoder_Up_2_original_4_1024_1_0.25_256_bior6.8_constant_2_2_2_e_r_0_d_r_0_ema_True_all_batched_threshold_use_sample_training_bf16_1.0_1/filter_single_image_udit_1152_32_16/checkpoints/step=step=3250000.ckpt"


print(f"Loading model")


#model = Model.from_pretrained(pretrained_model_name_or_path=model_name)
model = Model.from_pretrained(
    model_config_uri=MODEL_CONFIG_URI,
    model_checkpoint_uri=MODEL_CHECKPOINT_URI
)


model.set_inference_fusion_params(
        scale, diffusion_rescale_timestep
    )




data_onnx = {
    'images': torch.zeros((1, 3, 224, 224)),  # Dummy tensor
    'img_idx': torch.tensor([0], device='cuda:0'),
    'low': torch.zeros((1, 1, 46, 46, 46), device='cuda:0'),  # Dummy tensor
    'id': 'test'
}
data_idx = 0



torch.onnx.export(
    model,
    (data_onnx, data_idx),
    "model_100.onnx",
    export_params=True,
    opset_version=19,
    do_constant_folding=True,
    input_names=['data', 'data_idx'],
    output_names=['low_pred', 'highs_pred'],
    verbose = True, 
    dynamo=True)

# Validate the exported model
onnx_model = onnx.load("model_100.onnx")
onnx.checker.check_model("model_100.onnx")
print("✅ ONNX model exported and validated successfully")