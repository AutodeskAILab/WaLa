from pathlib import Path
import open3d as o3d
import os

from pytorch_lightning import seed_everything

from src.dataset_utils import (
    get_singleview_data,
    get_multiview_data,
    get_voxel_data_json,
)
from src.model_utils import Model
import argparse


def simplify_mesh(obj_path, target_num_faces=1000):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    simplified_mesh = mesh.simplify_quadric_decimation(target_num_faces)
    o3d.io.write_triangle_mesh(obj_path, simplified_mesh)


def add_args(parser):
    input_data_group = parser.add_mutually_exclusive_group()
    input_data_group.add_argument(
        "--images",
        type=str,
        nargs="+",
        help="Path to input image(s). A 3D object will be generated from each image.",
    )
    input_data_group.add_argument(
        "--multi_view_images",
        type=str,
        nargs="+",
        help="Path to input multi_view images. A 3D object will be generated from these images.",
    )
    input_data_group.add_argument(
        "--voxel_files",
        type=str,
        nargs="+",
        help="Path to input voxel files. A 3D object will be generated from each voxel file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="./checkpoint.ckpt",
        choices=["ADSKAILab/Make-A-Shape-single-view-20m",
                "ADSKAILab/Make-A-Shape-multi-view-20m",
                "ADSKAILab/Make-A-Shape-voxel-16res-20m",
                "ADSKAILab/Make-A-Shape-voxel-32res-20m",
                "ADSKAILab/Make-A-Shape-voxel-16res-20m"
        ],
        help="Model name (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use. If cuda is not available, it will use cpu  (default: %(default)s).",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="obj",
        help="Output format (obj, sdf).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="Scale of the generated object (default: %(default)s).",
    )
    parser.add_argument(
        "--diffusion_rescale_timestep",
        type=int,
        default=100,
        help="Diffusion rescale timestep (default: %(default)s).",
    )
    parser.add_argument(
        "--target_num_faces",
        type=int,
        default=None,
        help="Target number of faces for mesh simplification.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility (default: %(default)s).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="examples",
        help="Path to output directory.",
    )


def generate_3d_object(
    model,
    data,
    data_idx,
    scale,
    diffusion_rescale_timestep,
    save_dir="examples",
    output_format="obj",
    target_num_faces=None,
    seed=42,
):
    # Set seed
    seed_everything(seed, workers=True)

    save_dir.mkdir(parents=True, exist_ok=True)
    model.set_inference_fusion_params(scale, diffusion_rescale_timestep)
    output_path = model.test_inference(
        data, data_idx, save_dir=save_dir, output_format=output_format
    )

    if output_format == "obj" and target_num_faces:
        simplify_mesh(output_path, target_num_faces=target_num_faces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    print(f"Loading model")
    model = Model.from_pretrained(pretrained_model_name_or_path=args.model_name)

    if hasattr(model, "image_transform"):
        image_transform = model.image_transform
    else:
        image_transform = None

    if args.images:
        for image_path in args.images:
            print(f"Processing image: {image_path}")
            data = get_singleview_data(
                image_file=Path(image_path),
                image_transform=image_transform,
                device=model.device,
                image_over_white=False,
            )
            data_idx = 0
            save_dir = Path(args.output_dir) / Path(image_path).stem

            model.set_inference_fusion_params(
                args.scale, args.diffusion_rescale_timestep
            )

            generate_3d_object(
                model,
                data,
                data_idx,
                args.scale,
                args.diffusion_rescale_timestep,
                save_dir,
                args.output_format,
                args.target_num_faces,
                args.seed,
            )
    elif args.multi_view_images:
        image_views = [
            int(os.path.basename(Path(image).name).split(".")[0])
            for image in args.multi_view_images
        ]
        data = get_multiview_data(
            image_files=args.multi_view_images,
            views=image_views,
            image_transform=image_transform,
            device=model.device
        )
        data_idx = 0
        save_dir = Path(args.output_dir) / Path(args.multi_view_images[0]).stem
        generate_3d_object(
            model,
            data,
            data_idx,
            args.scale,
            args.diffusion_rescale_timestep,
            save_dir,
            args.output_format,
            args.target_num_faces,
            args.seed,
        )
    elif args.voxel_files:
        for voxel_file in args.voxel_files:
            print(f"Processing voxel file: {voxel_file}")
            data = get_voxel_data_json(
                voxel_file=Path(voxel_file),
                voxel_resolution=16,
                device=model.device,
            )
            data_idx = 0
            save_dir = Path(args.output_dir) / Path(voxel_file).stem
            generate_3d_object(
                model,
                data,
                data_idx,
                args.scale,
                args.diffusion_rescale_timestep,
                save_dir,
                args.output_format,
                args.target_num_faces,
                args.seed,
            )
