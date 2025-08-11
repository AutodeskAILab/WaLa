import os
import tensorrt as trt
import numpy as np
import time
import torch
from pathlib import Path
import pycuda.driver as cuda
#import pycuda.autoinit
import io
import boto3
import sys
import argparse
from PIL import Image

cuda.init()
device = cuda.Device(0)
cuda_driver_context = device.make_context()

# Set the device for PyTorch. This should happen after the context is created.
if torch.cuda.is_available():
    torch.cuda.set_device(0)

device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Add the parent directory to sys.path so 'src' can be imported
sys.path.append(str(Path.cwd().parent))
from src.dataset_utils import (
    get_singleview_data_trt,
    get_singleview_data,
    get_multiview_data,
    get_voxel_data_json,
    get_image_transform_latent_model,
    get_pointcloud_data,
    get_mv_dm_data,
    get_sv_dm_data,
    get_sketch_data
)
import Optim_Utils

###### MVDream Setup
from src.mvdream.ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder
from src.mvdream.camera_utils import get_camera, get_camera_build3d
import open_clip

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

# Load TensorRT engine
def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(f.read())



def load_engine_from_s3(bucket_name, s3_key):
    """
    Loads a TensorRT engine directly from S3 into memory and checks buffer size.
    """


    s3 = boto3.client('s3')
    # Get expected file size from S3
    obj = s3.head_object(Bucket=bucket_name, Key=s3_key)
    expected_size = obj['ContentLength']

    engine_buffer = io.BytesIO()
    s3.download_fileobj(bucket_name, s3_key, engine_buffer)
    engine_buffer.seek(0)
    actual_size = len(engine_buffer.getvalue())
    print(f"S3 engine size: {expected_size} bytes, Downloaded buffer size: {actual_size} bytes")
    if actual_size != expected_size:
        raise RuntimeError("Downloaded engine size does not match S3 object size!")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(engine_buffer.read())


# Modern TensorRT inference function using tensor API
def run_trt_inference(engine, input_data,stream,context):
    """
    Run inference using TensorRT with modern tensor API
    
    Args:
        engine: TensorRT engine
        input_data: List of input numpy arrays
        
    Returns:
        List of output numpy arrays
    """

  

    # Get input and output tensor names
    input_names = []
    output_names = []
    
    time_input_names = time.time()
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)
    print(f"Time to get input/output names: {time.time() - time_input_names:.4f} seconds")
    # Prepare device memory
    device_memory = []
    
    time_handle = time.time()
    # Handle inputs
    for i, name in enumerate(input_names):
        if i >= len(input_data):
            continue
            
        data = input_data[i]
        
        # Handle dynamic input shapes
        if -1 in engine.get_tensor_shape(name):
            context.set_input_shape(name, data.shape)
        
        # Allocate device memory and copy input data
        mem = cuda.mem_alloc(data.nbytes)
        cuda.memcpy_htod_async(mem, data, stream)
        device_memory.append(mem)
        context.set_tensor_address(name, int(mem))
    print(f"Time to handle inputs: {time.time() - time_handle:.4f} seconds")

    # Prepare outputs
    outputs = []
    output_memory = []
    
    time_allocate = time.time()
    # Allocate output memory
    for name in output_names:
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        
        # Allocate device memory for output
        size = trt.volume(shape) * np.dtype(dtype).itemsize
        mem = cuda.mem_alloc(size)
        device_memory.append(mem)
        output_memory.append(mem)
        
        # Set output tensor address
        context.set_tensor_address(name, int(mem))
        
        # Create host output array
        output = np.empty(shape, dtype=dtype)
        outputs.append(output)
    print(f"Time to allocate output memory: {time.time() - time_allocate:.4f} seconds")

    # Run inference
    time_asyncv3 = time.time()
    context.execute_async_v3(stream_handle=stream.handle)
    print(f"Async inference time: {time.time() - time_asyncv3:.4f} seconds")

    time_copy = time.time()
    # Copy outputs from device to host
    for i, mem in enumerate(output_memory):
        cuda.memcpy_dtoh_async(outputs[i], mem, stream)
    print(f"Time to copy outputs: {time.time() - time_copy:.4f} seconds")

    time_sync = time.time()
    # Synchronize to ensure all operations are complete
    stream.synchronize()
    print(f"Time to synchronize: {time.time() - time_sync:.4f} seconds")

    time_free = time.time()
    # Free device memory
    for mem in device_memory:
        mem.free()
    print(f"Time to free device memory: {time.time() - time_free:.4f} seconds")


    return outputs


# Main experiment function

def run_tensorrt_experiment(
    input_dir, save_dir, engine_path, prompt, modality="singleview", use_s3=False, multiview=False
):
    image_transform = get_image_transform_latent_model()
    if use_s3:
        raise NotImplementedError("S3 loading is disabled in this version.")
    else:
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine file {engine_path} does not exist.")
        engine = load_engine(engine_path)

    results = []
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    time_contx = time.time()
    context = engine.create_execution_context()
    print(f"Time to create execution context: {time.time() - time_contx:.4f} seconds")
    

    # --- Main loop ---
    if modality == "mvdream":
        if not prompt:
            raise ValueError("A text prompt must be provided for the mvdream modality.")
        
        time_default = time.time()

        cuda_driver_context.push()
        #torch.cuda.set_device(0)  # or your device index

        num_of_frames = 6
        testing_views = [3, 6, 10, 26, 49, 50]
        # Instantiate the imported embedder to get text embeddings
        tokens = open_clip.tokenize(prompt)
        H = 256
        W = 256

         # Prepare ALL required inputs for the engine
        with torch.no_grad():
            prompt_tokens = open_clip.tokenize(prompt)                 # [1,77], int64
            uc_tokens = open_clip.tokenize([""])                       # [1,77], int64
            camera_embedding = setup_camera(num_of_frames, testing_views)  # [F, cam_dim], f32
            x_T = torch.randn(num_of_frames, 4, H//8, W//8)            # [F,4,H/8,W/8], f32

            # Convert to numpy with correct dtypes
            prompt_tokens_np = prompt_tokens.cpu().numpy().astype(np.int64)
            uc_tokens_np = uc_tokens.cpu().numpy().astype(np.int64)
            camera_embedding_np = camera_embedding.cpu().numpy().astype(np.float32)
            x_T_np = x_T.cpu().numpy().astype(np.float32)

        # Build input map by NAME. Adjust keys to match your ONNX export.
        # Common names used in your exporter: 'prompt_tokenized','uc_tokenized','camera_embedding','x_T'
        inputs= [
            prompt_tokens_np,
            uc_tokens_np,
            camera_embedding_np,
            x_T_np,
        ]
        prompt_slug = "".join(filter(str.isalnum, prompt)).lower()[:30]
        obj_name = f"mvdream_{prompt_slug}"

        stream = cuda.Stream()
        start_time = time.time()
        outputs = run_trt_inference(engine, inputs, stream, context)
        inference_time = time.time() - start_time

        print(f"TensorRT inference time: {inference_time:.4f} seconds")
        results.append({"object": obj_name, "inference_time": inference_time})
        
        output_images = outputs[0]
        
        for i, img_array in enumerate(output_images):
            img = Image.fromarray(img_array.astype(np.uint8))
            img_path = save_dir / f"{obj_name}_view_{i}.png"
            img.save(img_path)
            print(f"Saved output image to {img_path}")
            
        print('Total Inference time with Writing', time.time() - time_default)
        cuda_driver_context.pop()
    else:
        input_path = Path(input_dir)
        for obj in input_path.iterdir():
            cuda_driver_context.push()
            time_default = time.time()

            if modality == "multiview":
                for obj_folder in sorted(input_path.iterdir()):
                    if not obj_folder.is_dir():
                        continue
                    img_files = sorted(list(obj_folder.glob("*.png")))
                    if not img_files:
                        print(f"No images found in {obj_folder}")
                        continue
                    # Process images in batches of 4
                    for i in range(0, len(img_files), 4):
                        batch_files = img_files[i:i+4]
                        if len(batch_files) < 4:
                            print(f"Skipping batch starting at {batch_files[0]}: less than 4 images.")
                            continue
                        image_views = [int(Path(f).stem) for f in batch_files]
                        data = get_multiview_data(
                            image_files=batch_files,
                            views=image_views,
                            image_transform=image_transform,
                            device='cpu'
                        )
                        obj_name = obj_folder.name + "_" + "_".join([Path(f).stem for f in batch_files])
                        inputs = [
                            data["images"].cpu().numpy() if hasattr(data["images"], "cpu") else np.array(data["images"]),
                            data["low"].cpu().numpy() if hasattr(data["low"], "cpu") else np.array(data["low"]),
                            data["img_idx"].cpu().numpy() if hasattr(data["img_idx"], "cpu") else np.array([data["img_idx"]], dtype=np.int64)
                        ]
            elif modality == "pointcloud":
                if not obj.is_file():
                    cuda_driver_context.pop()
                    continue
                # Assume obj is a .npy file containing pointcloud data
                data = get_pointcloud_data(
                    pointcloud_file=obj,
                    device='cpu'
                )   
                obj_name = obj.stem
                inputs = [
                    data["Pointcloud"].cpu().numpy() if hasattr(data["Pointcloud"], "cpu") else np.array(data["Pointcloud"]),
                    data["low"].cpu().numpy() if hasattr(data["low"], "cpu") else np.array(data["low"]),
                    np.array([0], dtype=np.int64)
                ]
            elif modality == "voxels":
                if not obj.is_file():
                    cuda_driver_context.pop()
                    continue
                # Assume obj is a .npy file containing voxel data
                data = get_voxel_data_json(
                    voxel_file=obj,
                    voxel_resolution=16,
                    device='cpu'
                )
                obj_name = obj.stem
                inputs = [
                    data["voxels"].cpu().numpy() if hasattr(data["voxels"], "cpu") else np.array(data["voxels"]),
                    data["low"].cpu().numpy() if hasattr(data["low"], "cpu") else np.array(data["low"]),
                    np.array([0], dtype=np.int64)
                ]            
            else:  # singleview/sketch
                if not obj.is_file():
                    cuda_driver_context.pop()
                    continue
                data = get_singleview_data(
                    image_file=obj,
                    image_transform=image_transform,
                    image_over_white=False,
                    device='cpu'
                )
                obj_name = obj.stem
                inputs = [
                    data["images"].cpu().numpy() if hasattr(data["images"], "cpu") else np.array(data["images"]),
                    data["low"].cpu().numpy() if hasattr(data["low"], "cpu") else np.array(data["low"]),
                    data["img_idx"].cpu().numpy() if hasattr(data["img_idx"], "cpu") else np.array([data["img_idx"]], dtype=np.int64)
                ]

            stream = cuda.Stream()
            start_time = time.time()
            outputs = run_trt_inference(engine, inputs, stream, context)
            inference_time = time.time() - start_time

            print(f"TensorRT inference time: {inference_time:.4f} seconds")
            results.append({"object": obj_name, "inference_time": inference_time})
            
            # Convert numpy arrays to torch tensors
            low_trt = outputs[0]
            highs_trt = outputs[1:]

            if isinstance(low_trt, np.ndarray):
                low_trt = torch.from_numpy(low_trt)
            # Convert each element in highs_trt to tensor if it's a numpy array
            highs_trt = [
                torch.from_numpy(h) if isinstance(h, np.ndarray) else h
                for h in highs_trt
            ]
            # If highs_trt is not a list or tuple, wrap it in a list
            if not isinstance(highs_trt, (list, tuple)):
                highs_trt = [highs_trt]

            # Print shape or value safely
            first_high = highs_trt[0]
            if hasattr(first_high, "size"):
                print(first_high.size())
            elif hasattr(first_high, "shape"):
                print(first_high.shape)
            else:
                print(first_high)

            # Move all tensors to the correct device
            if hasattr(low_trt, "to"):
                low_trt = low_trt.to('cuda:0')
            highs_trt = [h.to('cuda:0') if hasattr(h, "to") else h for h in highs_trt]
            obj_path = str(save_dir / f"{obj_name}_trt.obj")
            Optim_Utils.save_visualization_obj(
                obj_name,
                obj_path=obj_path,
                samples=(low_trt, highs_trt)
            )
            print('Total Inference time with Writing', time.time() - time_default)

            cuda_driver_context.pop()

    avg_time = sum(item["inference_time"] for item in results) / len(results) if results else 0
    print(f"\nAverage TensorRT inference time: {avg_time:.4f} seconds")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with TensorRT engine.")
    parser.add_argument("--input_dir", type=str, required=False, help="Input directory containing data")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save output objects")
    parser.add_argument("--engine_path", type=str, required=True, help="Path to TensorRT engine file")
    parser.add_argument("--modality", type=str, choices=["singleview", "multiview", "pointcloud", "voxels","mvdream"], required=True, help="Modality type")
    parser.add_argument("--prompt", type=str, default="generate me a cup", help="Text prompt for mvdream modality.")

    args = parser.parse_args()

    # Validate arguments
    if args.modality != 'mvdream' and not args.input_dir:
        parser.error("--input_dir is required for modalities other than 'mvdream'.")


    try:
        # Use keyword arguments for clarity and to avoid positional errors
        results = run_tensorrt_experiment(
            input_dir=args.input_dir,
            save_dir=args.save_dir,
            engine_path=args.engine_path,
            modality=args.modality,
            prompt=args.prompt,
            multiview=(args.modality == "multiview")
        )
    finally:
        cuda_driver_context.pop()
        pass





# Example usages:
# python TRT_Run.py --input_dir ../examples/single_view --save_dir ../examples/Test_Gen --engine_path model_sv.trt --modality singleview
# python TRT_Run.py --input_dir ../examples/multi_view --save_dir ../examples/Test_Gen --engine_path model_mv.trt --modality multiview
# python TRT_Run.py --input_dir ../examples/pointcloud --save_dir ../examples/Test_Gen --engine_path model_pointcloud.trt --modality pointcloud
# python TRT_Run.py --input_dir ../examples/voxel --save_dir ../examples/Test_Gen --engine_path model_voxels.trt --modality voxels