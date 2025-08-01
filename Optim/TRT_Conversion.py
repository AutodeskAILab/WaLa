import argparse
import onnx
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import io
import os

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine.")
    parser.add_argument("--onnx_path", type=str, default="model_pointcloud.onnx", help="Path to ONNX model file")
    parser.add_argument("--engine_path", type=str, default="model_pointcloud.trt", help="Path to save TensorRT engine")
    args = parser.parse_args()

    local_onnx = args.onnx_path
    trt_engine_path = args.engine_path

    # --- CHECK ONNX AND DATA FILES ---
    if not os.path.exists(local_onnx):
        raise FileNotFoundError(f"ONNX model file {local_onnx} does not exist.")
    else:
        print(f"{local_onnx} found.")

    print("ONNX files ready.")

    # --- TENSORRT CONVERSION ---
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser_trt = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 10 << 30) # 10 GiB
    config.builder_optimization_level = 4
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.set_flag(trt.BuilderFlag.STRICT_NANS)
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    with open(local_onnx, "rb") as f:
        if not parser_trt.parse(f.read()):
            for i in range(parser_trt.num_errors):
                print(parser_trt.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    serialized_engine = builder.build_serialized_network(network, config)
    with open(trt_engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to {trt_engine_path}")

if __name__ == "__main__":
    main()

# Example usages:
# python TRT_Conversion.py --onnx_path model_pointcloud.onnx --engine_path model_pointcloud.trt
# python TRT_Conversion.py --onnx_path model_voxels.onnx --engine_path model_voxels.trt
# python TRT_Conversion.py --onnx_path model_sv.onnx --engine_path model_sv.trt
# python TRT_Conversion.py --onnx_path model_sketch.onnx --engine_path model_sketch.trt
# python TRT_Conversion.py --onnx_path model_mv.onnx --engine_path model_mv.trt