import numpy as np
import torch
import open_clip
from pathlib import Path
import sys
import requests
from PIL import Image
import time
# Add your project root to sys.path if needed
sys.path.append(str(Path(__file__).resolve().parent))

from TRT_Run import setup_camera
import atexit
import pycuda.driver as cuda

def cleanup():
    try:
        while True:
            cuda.Context.pop()
    except Exception:
        pass  # Context stack is empty

atexit.register(cleanup)

def prepare_inputs(prompt="generate me a cup"):
    num_of_frames = 6
    testing_views = [3, 6, 10, 26, 49, 50]
    H = 256
    W = 256

    with torch.no_grad():
        prompt_tokens = open_clip.tokenize(prompt).cpu().numpy().astype(np.int64)    # [1, 77], int64
        uc_tokens = open_clip.tokenize([""]).cpu().numpy().astype(np.int64)          # [1, 77], int64
        camera_embedding = setup_camera(num_of_frames, testing_views).cpu().numpy().astype(np.float32)  # [F, cam_dim]
        x_T = torch.randn(num_of_frames, 4, H // 8, W // 8).cpu().numpy().astype(np.float32)

    # Convert to lists for JSON serialization
    payload = {
        "inputs": [
            prompt_tokens.tolist(),
            uc_tokens.tolist(),
            camera_embedding.tolist(),
            x_T.tolist()
        ]
    }
    return payload

if __name__ == "__main__":
    start = time.time()
    payload = prepare_inputs("generate me a cup")
    response = requests.post("http://localhost:8000/infer", json=payload)
    response.raise_for_status()
    outputs = response.json()["outputs"]

    # Assume outputs[0] is a batch of images: shape [N, H, W, C] or [N, H, W]
    output_images = outputs[0]
    for i, img_array in enumerate(output_images):
        arr = np.array(img_array, dtype=np.uint8)
        img = Image.fromarray(arr)
        img.show()  # Or img.save(f"output_{i}.png")
    end = time.time()
    print(f"Inference time: {end - start} seconds")
