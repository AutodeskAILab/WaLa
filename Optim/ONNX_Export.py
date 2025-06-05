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

output_dir = '../examples/Test_Gen'
output_format = 'obj'
target_num_faces = None
scale = 1.8
seed = 42
diffusion_rescale_timestep = 10

print(f"Loading model")


model = Model.from_pretrained(pretrained_model_name_or_path=model_name)
model.set_inference_fusion_params(
        scale, diffusion_rescale_timestep
    )


single_image = Path('../examples/single_view/table.png')

data_onnx = {
    'images': torch.zeros((1, 3, 224, 224)),  # Dummy tensor
    'img_idx': torch.tensor([0], device='cuda:0'),
    'low': torch.zeros((1, 1, 46, 46, 46), device='cuda:0'),  # Dummy tensor
    'id': 'test'
}
data_idx = 0
save_dir = Path(output_dir) 
base_name = os.path.basename(single_image)
image_name = os.path.splitext(base_name)[0]  


torch.onnx.export(
    model,
    (data_onnx, data_idx),
    "model_10.onnx",
    export_params=True,
    opset_version=19,
    do_constant_folding=True,
    input_names=['data', 'data_idx'],
    output_names=['low_pred', 'highs_pred'],
    verbose = True, 
    dynamo=True)

# Validate the exported model
onnx_model = onnx.load("model_10.onnx")
onnx.checker.check_model("model_10.onnx")
print("✅ ONNX model exported and validated successfully")