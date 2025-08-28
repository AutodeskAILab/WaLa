import os
os.environ["XFORMERS_DISABLED"] = "1"

from pathlib import Path
import open3d as o3d
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

import open_clip

###### MVDream Setup
from src.mvdream.ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder
from src.mvdream.camera_utils import get_camera, get_camera_build3d

os.environ['XFORMERS_ENABLED'] = '0'


def setup_camera(
        num_frames = 4,
        testing_views = [0, 6, 10, 26],
    ):
        indices_list = testing_views
        batch_size = num_frames
        
        camera = get_camera_build3d(indices_list)
        camera = camera.repeat(batch_size // num_frames, 1)
        return camera
#####################

# Mapping modality to scale and diffusion_rescale_timestep
modality_params = {
    "voxels":      {"scale": 1.5, "steps": 5},
    "pointcloud":  {"scale": 1.3, "steps": 8},
    "sv":          {"scale": 1.8, "steps": 5},
    "sketch":      {"scale": 1.8, "steps": 5},
    "mv":          {"scale": 1.3, "steps": 5},
    "mvdream":     {"scale": 1.3, "steps": 5},
}


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
                
def prepare_dummy_inputs(modality, model=None):
    """
    Prepare dummy inputs based on the specified modality.
    
    Args:
        modality (str): The modality type (sv, sketch, mv, voxels, pointcloud, mvdream)
        model: The model instance (required for mvdream modality)
    
    Returns:
        tuple: (dummy_inputs, data_idx) for non-mvdream modalities
        tuple: (dummy_inputs,) for mvdream modality
    """
    if modality in ["sv", "sketch"]:
        data_onnx = {
            'images': torch.zeros((1, 3, 224, 224)),
            'img_idx': torch.tensor([0], device='cuda:0'),
            'low': torch.zeros((1, 1, 46, 46, 46), device='cuda:0'),
            'id': 'dummy'
        }
        return data_onnx, 0
        
    elif modality == "mv":
        num_views = 4 
        data_onnx = {
            'images': torch.zeros((1, num_views, 3, 224, 224)),
            'img_idx': torch.tensor([0], device='cuda:0'),
            'low': torch.zeros((1, 1, 46, 46, 46), device='cuda:0'),
            'id': 'dummy'
        }
        return data_onnx, 0
        
    elif modality == "voxels":
        data_onnx = {
            'voxels': torch.zeros((1, 1, 16, 16, 16)),
            'low': torch.zeros((1, 1, 46, 46, 46), device='cuda:0'),
            'id': 'dummy'
        }
        return data_onnx, 0
        
    elif modality == "pointcloud":
        data_onnx = {
            'Pointcloud': torch.zeros((1, 25000, 3)),
            'low': torch.zeros((1, 1, 46, 46, 46), device='cuda:0'),
            'id': 'dummy'
        }
        return data_onnx, 0
        
    elif modality == "mvdream":
        if model is None:
            raise ValueError("Model instance is required for mvdream modality")
            
        prompt_text = 'Dummy input'
        num_of_frames = 6
        testing_views = [3, 6, 10, 26, 49, 50]
        H = 256
        W = 256

        # Pre-process inputs to get tensors
        with torch.no_grad():
            prompt_tokens = open_clip.tokenize(prompt_text).to(model.device)    # [1, 77], int64
            uc_tokens = open_clip.tokenize([""]).to(model.device)               # [1, 77], int64
            camera_embedding = setup_camera(num_of_frames, testing_views).to(model.device)  # [F, cam_dim]
            x_T = torch.randn(num_of_frames, 4, H // 8, W // 8, device=model.device, dtype=torch.float32)

        return (prompt_tokens, uc_tokens, camera_embedding, x_T),
    
    else:
        raise ValueError(f"Unsupported modality: {modality}")


def export_to_onnx(model, modality, dummy_inputs, data_idx=None):
    """
    Export model to ONNX format based on modality.
    
    Args:
        model: The PyTorch model to export
        modality (str): The modality type
        dummy_inputs: Prepared dummy inputs for the model
        data_idx: Data index (only needed for non-mvdream modalities)
    """
    output_path = f"model_{modality}.onnx"
    
    if modality == "mvdream":
        model.eval()
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=False,
            input_names=['prompt_tokenized', 'uc_tokenized', 'camera_embedding', 'x_T'],
            output_names=['images'],
            verbose=True,
            dynamo=True,
        )
    else:
        torch.onnx.export(
            model,
            (dummy_inputs, data_idx),
            output_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=['data', 'data_idx'],
            output_names=['low_pred', 'highs_pred'],
            verbose=True,
            dynamo=True,
        )
    
    # Validate the exported model
    onnx.checker.check_model(output_path)
    print(f"✅ ONNX model for {modality} exported and validated successfully")

 

 
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
    args.use_mv_dream_conditions = False

    # Set the correct flag based on modality
    if args.modality in ["sv", "mv", "sketch"]:
        args.use_image_conditions = True
    elif args.modality == "pointcloud":
        args.use_pointcloud_conditions = True
    elif args.modality == "voxels":
        args.use_voxel_conditions = True
    elif args.modality == "mvdream":
        args.use_mv_dream_conditions = True
    else:
        raise ValueError("Unknown or unsupported modality")

    # Set flags based on argument
    sv = args.modality == "sv"
    mv = args.modality == "mv"
    voxels = args.modality == "voxels"
    pointcloud = args.modality == "pointcloud"
    sketch = args.modality == "sketch"
    mvdream = args.modality == "mvdream"

    # Model selection 
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
    elif mvdream:
        model_name = "ADSKAILab/WaLa-MVDream-DM6"
    else:
        raise ValueError("Unknown or not supported modality")

    print(f"Loading model: {model_name}")

    # --- Conditional loading only for mvdream ---
    if mvdream:
        ## Load MVDream model
        model = load_mvdream_model(
            pretrained_model_name_or_path=model_name,
            device=getattr(args, "device", "cuda:0")
        )
        
        ## Uncompile the model to access the original module
        recursively_unwrap_orig_mod(model)

        ## Prepare dummy inputs and export
        dummy_inputs, = prepare_dummy_inputs(args.modality, model)
        export_to_onnx(model, args.modality, dummy_inputs)
    else:
        ## Load other models
        model = Model.from_pretrained(pretrained_model_name_or_path=model_name)
        model.set_inference_fusion_params(scale, diffusion_rescale_timestep)

        ## Uncompile the model to access the original module
        recursively_unwrap_orig_mod(model)

        ## Prepare dummy inputs and export
        data_onnx, data_idx = prepare_dummy_inputs(args.modality)
        export_to_onnx(model, args.modality, data_onnx, data_idx)
  



    # Example usages:
    # python onnx_export.py --modality pointcloud
    # python onnx_export.py --modality voxels
    # python onnx_export.py --modality sv
    # python onnx_export.py --modality sketch
    # python onnx_export.py --modality mv
    # python onnx_export.py --modality mvdream
