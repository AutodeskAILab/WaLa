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
import argparse



os.environ["XFORMERS_DISABLED"] = "1"

# Mapping modality to scale and diffusion_rescale_timestep
modality_params = {
    "voxels":      {"scale": 1.5, "steps": 5},
    "pointcloud":  {"scale": 1.3, "steps": 8},
    "sv":      {"scale": 1.8, "steps": 5},
    "sketch":      {"scale": 1.8, "steps": 5},
    "mv":      {"scale": 1.3, "steps": 5},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export WaLa model to ONNX for different modalities.")
    parser.add_argument("--modality", type=str, choices=list(modality_params.keys()), required=True,
                        help="Choose modality: " + ", ".join(modality_params.keys()))
    args = parser.parse_args()

    # Set scale and steps based on modality
    scale = modality_params[args.modality]["scale"]
    diffusion_rescale_timestep = modality_params[args.modality]["steps"]


    
    # Reset all condition flags to False first
    args.use_image_conditions = False
    args.use_pointcloud_conditions = False
    args.use_voxel_conditions = False
    args.use_depth_conditions = False
    
    # Set the correct flag based on modality
    if args.modality in ["sv", "mv", "sketch"]:
        args.use_image_conditions = True
    elif args.modality == "pointcloud":
        args.use_pointcloud_conditions = True
    elif args.modality == "voxels":
        args.use_voxel_conditions = True
    else:
        raise ValueError("Unknown or unsupported modality")

    # Set flags based on argument
    sv = args.modality.startswith("sv")
    mv = args.modality.startswith("mv")
    voxels = args.modality == "voxels"
    pointcloud = args.modality == "pointcloud"
    sketch = args.modality == "sketch"

    # Model selection (example, adjust as needed)
    if sv:
        model_name = 'ADSKAILab/WaLa-SV-1B'
    elif sketch:
        model_name = 'ADSKAILab/WaLa-SK-1B'
    elif mv:
        model_name = 'ADSKAILab/WaLa-RGB4-1B'
    elif voxels:
        model_name = 'ADSKAILab/WaLa-VX16-1B'
    elif pointcloud:
        model_name = 'ADSKAILab/WaLa-PC-1B'
    else:
        raise ValueError("Unknown or not supported modality")

    print(f"Loading model: {model_name}")
    model = Model.from_pretrained(model_name)



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


    recursively_unwrap_orig_mod(model)
    model.set_inference_fusion_params(scale, diffusion_rescale_timestep)

    # Prepare dummy input and dynamic_shapes
    if sv:
        data_onnx = {
            'images': torch.zeros((1, 3, 224, 224)),
            'img_idx': torch.tensor([0], device='cuda:0'),
            'low': torch.zeros((1, 1, 46, 46, 46), device='cuda:0'),
            'id': 'dummy'
        }
        data_idx = 0
        
    elif mv:
        num_views = 4 
        data_onnx = {
            'images': torch.zeros((1, num_views, 3, 224, 224)),
            'img_idx': torch.tensor([0], device='cuda:0'),
            'low': torch.zeros((1, 1, 46, 46, 46), device='cuda:0'),
            'id': 'dummy'
        }
        data_idx = 0
        
    elif voxels:
        data_onnx = {
            'voxels': torch.zeros((1, 1, 16, 16, 16)),
            'low': torch.zeros((1, 1, 46, 46, 46), device='cuda:0'),
            'id': 'dummy'
        }
        data_idx = 0
        
    elif pointcloud:
        data_onnx = {
            'Pointcloud': torch.zeros((1, 25000, 3)),
            'low': torch.zeros((1, 1, 46, 46, 46), device='cuda:0'),
            'id': 'dummy'
        }
        data_idx = 0
        

    torch.onnx.export(
        model,
        (data_onnx, data_idx),
        f"model_{args.modality}.onnx",
        export_params=True,
        opset_version=19,
        do_constant_folding=True,
        input_names=['data', 'data_idx'],
        output_names=['low_pred', 'highs_pred'],
        verbose=True,
        dynamo=True,
    )

    # Validate the exported model
    onnx.checker.check_model(f"model_{args.modality}.onnx")
    print(f"✅ ONNX model for {args.modality} exported and validated successfully")

    
    # Example usages:
    # python Optim/ONNX_Export.py --modality pointcloud
    # python Optim/ONNX_Export.py --modality voxels
    # python Optim/ONNX_Export.py --modality sv
    # python Optim/ONNX_Export.py --modality sketch
    # python Optim/ONNX_Export.py --modality mv
