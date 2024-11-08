from pathlib import Path
import open3d as o3d
import os

from pytorch_lightning import seed_everything

from src.dataset_utils import (
    get_singleview_data,
    get_multiview_data,
    get_voxel_data_json,
    get_image_transform_latent_model,
    get_pointcloud_data,
    get_dm6_data,
    get_sketch_data
)
from src.model_utils import Model
from src.mvdream_utils import load_mvdream_model
import argparse
from PIL import Image


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
    input_data_group.add_argument(
        "--pointcloud",
        type=str,
        nargs="+",
        help="Path to input poincloud files. A 3D object will be generated from each pointcloud file.",
    )
    input_data_group.add_argument(
        "--dm6",
        type=str,
        nargs="+",
        help="Path to input 6 depth-map images. A 3D object will be generated from these depth-maps.",
    )
    input_data_group.add_argument(
        "--text_to_dm6",
        type=str,
        nargs="+",
        help="String used to generate 6 depth-map image views.",
    )
    input_data_group.add_argument(
        "--text_to_mv",
        type=str,
        nargs="+",
        help="String used to generate 4 multi-view images.",
    )
    input_data_group.add_argument(
        "--sketch",
        type=str,
        nargs="+",
        help="Path to sketch file that will be used to generate a 3D object.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="./checkpoint.ckpt",
        # choices=["ADSKAILab/WaLa-SV-1B",
        #         "ADSKAILab/WaLa-SK-1B",
        #         "ADSKAILab/WaLa-UN-1B",
        #         "ADSKAILab/WaLa-MVDream-DM6",
        #         "ADSKAILab/WaLa-MVDream-RGB4",
        #         "ADSKAILab/WaLa-DM4-1B",
        #         "ADSKAILab/WaLa-DM6-1B",
        #         "ADSKAILab/WaLa-VX16-1B",
        #         "ADSKAILab/WaLa-PC-1B",
        #         "ADSKAILab/WaLa-RGB4-1B"
        # ],
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
    
    if args.text_to_dm6 or args.text_to_mv:
        model = load_mvdream_model(
            pretrained_model_name_or_path = args.model_name, 
            device = args.device
        )
        image_transform = None    
    else:
        model = Model.from_pretrained(pretrained_model_name_or_path=args.model_name)
        image_transform = get_image_transform_latent_model()

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
            device=model.device,
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
    elif args.pointcloud:
        for pointcloud_path in args.pointcloud:
            print(f"Processing pointcloud file: {pointcloud_path}")
            data = get_pointcloud_data(
                pointcloud_file=Path(pointcloud_path), device=model.device
            )

            data_idx = 0
            save_dir = Path(args.output_dir) / Path(pointcloud_path).stem
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

    elif args.dm6:
        dm_views = [
            int(os.path.basename(Path(dm).name).split(".")[0]) for dm in args.dm6
        ]

        data = get_dm6_data(
            image_files=args.dm6,
            views=dm_views,
            image_transform=image_transform,
            device=model.device,
        )

        data_idx = 0
        save_dir = Path(args.output_dir) / Path(args.dm6[0]).stem

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

    elif args.text_to_dm6:
        text_input = str(args.text_to_dm6)

        num_of_frames = 6
        testing_views = [3, 6, 10, 26, 49, 50]

        images_np, image_views = model.inference_step(prompt=text_input, num_frames=num_of_frames, testing_views=testing_views)
        images = [Image.fromarray(image) for image in images_np]
        
        save_dir = Path(args.output_dir) / Path("depth_maps")
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(images):
            output_path = os.path.join(save_dir, f"image_{i}.png")
            img.save(output_path, format = "PNG")


    elif args.text_to_mv:
        text_input = str(args.text_to_mv)

        num_of_frames = 4
        testing_views = [0, 6, 10, 26]

        images_np, image_views = model.inference_step(prompt=text_input, num_frames=num_of_frames, testing_views=testing_views)
        images = [Image.fromarray(image) for image in images_np]
        
        save_dir = Path(args.output_dir) / Path("mv_images")
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(images):
            output_path = os.path.join(save_dir, f"image_{i}.png")
            img.save(output_path, format = "PNG")


    elif args.sketch:
        for sketch_path in args.sketch:
            print(f"Processing sketch: {sketch_path}")

            data = get_sketch_data(
                image_file=Path(sketch_path),
                image_transform=image_transform,
                device=model.device,
                image_over_white=False,
            )

            data_idx = 0
            save_dir = Path(args.output_dir) / Path(sketch_path).stem

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

