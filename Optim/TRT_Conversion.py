import onnx
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import io
import boto3
import os

# --- CONFIGURATION ---
bucket_name = 'giuliaa-optim'
onnx_s3_key = '/ONNX/model_voxel_16.onnx'
onnx_data_s3_key = '/ONNX/model_voxel_16.onnx.data'
local_onnx = 'model_voxel_16.onnx'
local_data = 'model_voxel_16.onnx.data'
trt_engine_path = "model_voxel_16-5.1.trt"
trt_s3_key = '/TRT/model_voxel_16-5.1.trt'
UPLOAD_TO_S3 = True  # <-- Set to True to upload, False to skip


# --- DOWNLOAD ONNX AND DATA FILES FROM S3 IF NOT PRESENT ---
s3 = boto3.client('s3')
if not os.path.exists(local_onnx):
    print("Downloading model.onnx from S3...")
    s3.download_file(bucket_name, onnx_s3_key, local_onnx)
else:
    print("model.onnx already exists locally, skipping download.")

if not os.path.exists(local_data):
    print("Downloading model.onnx.data from S3...")
    s3.download_file(bucket_name, onnx_data_s3_key, local_data)
else:
    print("model.onnx.data already exists locally, skipping download.")

print("ONNX files ready.")

# --- TENSORRT CONVERSION ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, TRT_LOGGER)

# Set optimization level flags
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 10 << 30) # 10 GiB
config.builder_optimization_level = 4
config.set_flag(trt.BuilderFlag.BF16)

#config.set_flag(trt.BuilderFlag.DEBUG)  # Layer-by-layer sync
config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)  # Enforce Precision Constraint
config.set_flag(trt.BuilderFlag.STRICT_NANS)  # Catch NaN propagation
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


# Parse the ONNX model
if not os.path.exists(local_onnx):
    raise FileNotFoundError(f"ONNX model file {local_onnx} does not exist.")

with open(local_onnx, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX model")

# Build the TensorRT engine 
serialized_engine = builder.build_serialized_network(network, config)

with open(trt_engine_path, "wb") as f:
    f.write(serialized_engine)

if UPLOAD_TO_S3:
    with open(trt_engine_path, "rb") as f:
        s3.upload_fileobj(f, bucket_name, trt_s3_key)
    print(f"Uploaded TensorRT engine to s3://{bucket_name}/{trt_s3_key}")
else:
    print("S3 upload skipped (UPLOAD_TO_S3 is False)")
