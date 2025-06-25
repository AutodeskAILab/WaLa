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

cuda.init()
device = cuda.Device(0)
cuda_driver_context = device.make_context()


# Add the parent directory to sys.path so 'src' can be imported
sys.path.append(str(Path.cwd().parent))
from src.dataset_utils import get_singleview_data,get_multiview_data, get_image_transform_latent_model
import Optim_Utils

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
    image_dir, save_dir, engine_path, bucket_name='giuliaa-optim', s3_key='TRT/model_base.trt', use_s3=False, multiview=False
):
    image_transform = get_image_transform_latent_model()
    if use_s3:
        engine = load_engine_from_s3(bucket_name, s3_key)
    else:
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine file {engine_path} does not exist.")
        engine = load_engine(engine_path)

    results = []
    image_path = Path(image_dir)
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    time_contx = time.time()
    context = engine.create_execution_context()
    print(f"Time to create execution context: {time.time() - time_contx:.4f} seconds")

    # --- Main loop ---
    for obj in image_path.iterdir():
        time_default = time.time()
        cuda_driver_context.push()

        if multiview:
            if not obj.is_dir():
                cuda_driver_context.pop()
                continue
            # Multiview: load all images in the 'img' subfolder
            img_dir = obj / "img"
            img_files = sorted(list(img_dir.glob("*.png")))[:6]  # or "*.jpg"
            if not img_files:
                print(f"No images found in {img_dir}")
                cuda_driver_context.pop()
                continue
            image_views = [int(Path(f).stem) for f in img_files]
            data = get_multiview_data(
                image_files=img_files,
                views=image_views,
                image_transform=image_transform,
                device='cpu'
            )
            img_name = obj.name
        else:
            # Single view: treat each file as one object
            if not obj.is_file():
                cuda_driver_context.pop()
                continue
            data = get_singleview_data(
                image_file=obj,
                image_transform=image_transform,
                image_over_white=False,
                device='cpu'
            )
            img_name = obj.stem
        # Prepare input data as numpy arrays
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
        results.append({"image": img_name, "inference_time": inference_time})

        low_trt_cpu = torch.from_numpy(outputs[0])
        high_trt_cpu = torch.from_numpy(outputs[1])
        low_trt = low_trt_cpu.to('cuda:0')
        high_trt = [high_trt_cpu.to('cuda:0')]

        obj_path = str(save_dir / f"{img_name}_trt.obj")
        Optim_Utils.save_visualization_obj(img_name, obj_path, (low_trt, high_trt))
        print('Total Inference time with Writing', time.time() - time_default)

        cuda_driver_context.pop()

    avg_time = sum(item["inference_time"] for item in results) / len(results)
    print(f"\nAverage TensorRT inference time: {avg_time:.4f} seconds")
    return results


if __name__ == "__main__":
    try:
        # Set multiview=True to enable multiview functionality
        results = run_tensorrt_experiment(
            '/GSO_Renders',
            '/GSO_Multiview_Objects_6-5.1',
            "model_MV_6-5.1.trt",
            'giuliaa-optim',
            '/TRT/model_MV_6-5.1.trt',
            use_s3=False,
            multiview=True  # <-- Enable multiview support
        )
    finally:
        # pop the one context we pushed at import time
        cuda_driver_context.pop()