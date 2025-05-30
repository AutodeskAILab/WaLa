import onnx
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit


# Convert ONNX to TensorRT engine
onnx_model_path = "model.onnx"
trt_engine_path = "model.trt"
trt_engine_path_optimized = "model_optimized.trt"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_model_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX model")


config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 10 << 30) # 1 GiB
#config.set_flag(trt.BuilderFlag.FP16)
#config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
#config.set_tactic_sources(trt.TacticSource.CUBLAS | trt.TacticSource.CUDNN | trt.TacticSource.EDGE_MASK_CONVOLUTIONS)
#config.builder_optimization_level = 3

# Build the TensorRT engine (TensorRT 8.x+)
serialized_engine = builder.build_serialized_network(network, config)

with open(trt_engine_path, "wb") as f:
    f.write(serialized_engine)
