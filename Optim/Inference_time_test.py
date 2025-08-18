import time
from pathlib import Path
from TRT_Run import run_trt_inference, load_engine
import subprocess
import sys
import os
import numpy as np
import torch
from PIL import Image
import pycuda.driver as cuda

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


cuda.init()
device = cuda.Device(0)
cuda_driver_context = device.make_context()


PROMPTS = [
    "generate me a cup",
    "a red car",
    "a futuristic chair",
    "a medieval castle",
    "a flying dragon"
]

ENGINE_PATH = "model_mvdream.trt"
SAVE_DIR = "Test"
MODALITY = "mvdream"

def run_tensorrt_experiment_reuse(
    input_dir, save_dir, engine_path, prompt, modality=MODALITY, use_s3=False, multiview=False, engine=None, context=None
):
    image_transform = get_image_transform_latent_model()
    if use_s3:
        raise NotImplementedError("S3 loading is disabled in this version.")
    else:
        if engine is None:
            if not os.path.exists(engine_path):
                raise FileNotFoundError(f"TensorRT engine file {engine_path} does not exist.")
            engine = load_engine(engine_path)
        if context is None:
            context = engine.create_execution_context()

    results = []
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # --- Main loop ---
    if modality == "mvdream":
        if not prompt:
            raise ValueError("A text prompt must be provided for the mvdream modality.")
        
        time_default = time.time()

        cuda_driver_context.push()

        num_of_frames = 6
        testing_views = [3, 6, 10, 26, 49, 50]
        H = 256
        W = 256

        with torch.no_grad():
            prompt_tokens = open_clip.tokenize(prompt)
            uc_tokens = open_clip.tokenize([""])
            camera_embedding = setup_camera(num_of_frames, testing_views)
            x_T = torch.randn(num_of_frames, 4, H//8, W//8)

            prompt_tokens_np = prompt_tokens.cpu().numpy().astype(np.int64)
            uc_tokens_np = uc_tokens.cpu().numpy().astype(np.int64)
            camera_embedding_np = camera_embedding.cpu().numpy().astype(np.float32)
            x_T_np = x_T.cpu().numpy().astype(np.float32)

        inputs = [
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

    avg_time = sum(item["inference_time"] for item in results) / len(results) if results else 0
    print(f"\nAverage TensorRT inference time: {avg_time:.4f} seconds")
    return results


def run_original_experiment(prompt, save_dir, model=None):
    import time
    from PIL import Image
    from pathlib import Path
    t1 = time.time()
    num_of_frames = 6
    testing_views = [3, 6, 10, 26, 49, 50]
    os.environ['XFORMERS_ENABLED'] = '1'

    images_np, image_views = model.inference_step(prompt=prompt, num_frames=num_of_frames, testing_views=testing_views)
    images = [Image.fromarray(image) for image in images_np]
    
    save_dir = Path(save_dir) / Path("depth_maps")
    save_dir.mkdir(parents=True, exist_ok=True)


    for i, img in enumerate(images):
        prompt_slug = "".join(filter(str.isalnum, prompt)).lower()[:30]
        obj_name = f"mvdream_{prompt_slug}"
        img_path = save_dir / f"{obj_name}_view_{i}.png"
        img.save(img_path)
    elapsed = time.time() - t1
    print(f'Prompt: "{prompt}" | Total Inference Time: {elapsed:.4f} s')
    return elapsed


def main():
    save_dir = Path(SAVE_DIR)
    (save_dir / "trt").mkdir(parents=True, exist_ok=True)
    (save_dir / "original").mkdir(parents=True, exist_ok=True)
    trt_results = []
    orig_results = []
    print("Starting inference time test...")
    

    # --- TRT warmup and context reuse ---
    print("Loading TRT engine and creating context...")
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()

    # Warmup round
    print("Running TRT warmup round...")
    warmup_prompt = "warmup prompt"
    run_tensorrt_experiment_reuse(
        input_dir=None,
        save_dir=save_dir / "trt",
        engine_path=ENGINE_PATH,
        prompt=warmup_prompt,
        modality=MODALITY,
        engine=engine,
        context=context
    )

    # Run all TRT inferences
    for prompt in PROMPTS:
        print(f"\nPrompt: {prompt}")
        print("Running TRT inference...")
        t0 = time.time()
        results = run_tensorrt_experiment_reuse(
            input_dir=None,
            save_dir=save_dir / "trt",
            engine_path=ENGINE_PATH,
            prompt=prompt,
            modality=MODALITY,
            engine=engine,
            context=context
        )
        trt_time = results[0]["inference_time"] if results else (time.time() - t0)
        print(f"TRT inference time: {trt_time:.4f} seconds")
        trt_results.append({
            "prompt": prompt,
            "trt_time": trt_time
        })

    # Release TRT resources and CUDA memory
    print("\nReleasing TRT resources and CUDA memory...")
    import gc
    import torch
    torch.cuda.empty_cache()
    gc.collect()
    try:
        cuda_driver_context.pop()
    except Exception:
        pass

    # --- Check if the model is available ---
     # If model is not provided, load it here (do this ONCE outside the loop for fairness)
    
    from src.mvdream_utils import load_mvdream_model
    model = load_mvdream_model(
        pretrained_model_name_or_path="ADSKAILab/WaLa-MVDream-DM6",
        device=("cuda:0")
    )

    # Run all original inferences
    for prompt in PROMPTS:
        print(f"\nPrompt: {prompt}")
        print("Running original inference...")
        orig_time = run_original_experiment(prompt, save_dir / "original", model=model)
        print(f"Original inference time: {orig_time:.4f} seconds")
        orig_results.append({
            "prompt": prompt,
            "original_time": orig_time
        })

    print("\n=== Inference Time Comparison ===")
    for trt, orig in zip(trt_results, orig_results):
        print(f"Prompt: {trt['prompt']}")
        print(f"  TRT:      {trt['trt_time']:.4f} s")
        print(f"  Original: {orig['original_time']:.4f} s")
        print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        cuda_driver_context.pop()