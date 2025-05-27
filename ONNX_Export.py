from pathlib import Path
import open3d as o3d
import os

from pytorch_lightning import seed_everything

from src.dataset_utils import (
    get_singleview_data,
    get_multiview_data,
    get_voxel_data_json,
    get_image_transform_latent_model,
    get_pointcloud_data,
    get_mv_dm_data,
    get_sv_dm_data,
    get_sketch_data
)
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

output_dir = 'examples/Test_Gen'
output_format = 'obj'
target_num_faces = None
scale = 1.8
seed = 42
diffusion_rescale_timestep = 5

print(f"Loading model")


model = Model.from_pretrained(pretrained_model_name_or_path=model_name)
image_transform = get_image_transform_latent_model()
model.set_inference_fusion_params(
        scale, diffusion_rescale_timestep
    )


single_image = Path('examples/single_view/table.png')

data_onnx = get_singleview_data(
        image_file=Path(single_image),
        image_transform=image_transform,
        device=model.device,
        image_over_white=False,
    )
data_idx = 0
save_dir = Path(output_dir) 
base_name = os.path.basename(single_image)
image_name = os.path.splitext(base_name)[0]  
data_onnx['id'] = 'test'     


torch.onnx.export(
    model,
    (data_onnx, data_idx),
    "model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['data', 'data_idx'],
    output_names=['output'],
    verbose = True, # might make it faster?
    dynamo=True)

# Validate the exported model
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX model exported and validated successfully")