import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pycuda.driver as cuda
import tensorrt as trt
import threading
import torch
import sys
from pathlib import Path
import traceback
import atexit

# Add the project root to sys.path so 'src' can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from TRT_Run import load_engine, run_trt_inference, setup_camera

###### MVDream Setup
from src.mvdream.ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder
from src.mvdream.camera_utils import get_camera, get_camera_build3d
import open_clip

app = FastAPI()

# ---- Config ----
ENGINE_PATH = "/home/rayhub-user/Optim-WaLa/Optim/model_mvdream.trt"
DEVICE_ID = 0

# ---- CUDA/TRT Global State ----
cuda.init()
device = cuda.Device(DEVICE_ID)
cuda_driver_context = device.make_context()
stream = cuda.Stream()
engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()
context_lock = threading.Lock()  # For thread safety

# ---- Warmup ----
def warmup():
    prompt_text = 'Dummy input'
    num_of_frames = 6
    testing_views = [3, 6, 10, 26, 49, 50]
    H = 256
    W = 256

    with torch.no_grad():
        prompt_tokens = open_clip.tokenize(prompt_text).cpu().numpy().astype(np.int64)    # [1, 77], int64
        uc_tokens = open_clip.tokenize([""]).cpu().numpy().astype(np.int64)               # [1, 77], int64
        camera_embedding = setup_camera(num_of_frames, testing_views).cpu().numpy().astype(np.float32)  # [F, cam_dim]
        x_T = torch.randn(num_of_frames, 4, H // 8, W // 8).cpu().numpy().astype(np.float32)

    dummy_inputs = [prompt_tokens, uc_tokens, camera_embedding, x_T]
    run_trt_inference(engine, dummy_inputs, stream, context)
    print("Warmup complete.")

warmup()


def cleanup():
    try:
        while True:
            cuda_driver_context.pop()
    except Exception:
        pass  # Context stack is empty

atexit.register(cleanup)

# ---- Request/Response Models ----
class InferenceRequest(BaseModel):
    inputs: list  # List of lists or nested lists representing numpy arrays
@app.post("/infer")
def infer(request: InferenceRequest):
    try:
        # Convert input lists to numpy arrays with correct dtype
        np_inputs = [
            np.array(request.inputs[0], dtype=np.int64),    # prompt_tokens
            np.array(request.inputs[1], dtype=np.int64),    # uc_tokens
            np.array(request.inputs[2], dtype=np.float32),  # camera_embedding
            np.array(request.inputs[3], dtype=np.float32),  # x_T
        ]
        with context_lock:
            cuda_driver_context.push()  # Ensure we are on the correct CUDA context
            outputs = run_trt_inference(engine, np_inputs, stream, context)
            cuda_driver_context.pop()  # Pop the context after inference
        outputs_as_lists = [out.tolist() for out in outputs]
        return {"outputs": outputs_as_lists}
    except Exception as e:
        import traceback
        print("Exception during inference:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

