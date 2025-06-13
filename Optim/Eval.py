import boto3
import os
import pathlib
from pathlib import Path
import trimesh
import numpy as np
import comet_ml
import torch
import argparse
import shutil
import mesh2sdf_fix 
from lfd import LightFieldDistance
import numpy as np
import math


def sync_s3_folder_to_local(bucket, s3_prefix, local_dir):
    """
    Download all .obj files from an S3 prefix to a local directory if not already present.
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('.obj'):
                continue
            local_path = local_dir / os.path.basename(key)
            if not local_path.exists():
                print(f"Downloading {key} to {local_path}")
                with open(local_path, 'wb') as f:
                    s3.download_fileobj(bucket, key, f)
            else:
                print(f"Already present: {local_path}")



################# Build 3D Functions ################


def compute_iou(sdf_a, sdf_b):
    """
    Returns a float intersection over union between the given signed distance fields.
    Arguments:
        sdf_a: A np.array of any shape.
        sdf_b: A np.array of any shape.
    """
    intersection = np.logical_and(sdf_a < 0, sdf_b < 0)
    union = np.logical_or(sdf_a < 0, sdf_b < 0)
    return float(np.sum(intersection)) / np.sum(union)

def scale_to_unit_cube(mesh, scale_ratio=1.0):
    """
    Returns a copy of the given mesh scaled to a unit cube.
    Arguments:
        mesh:        A trimesh.Trimesh.
        scale_ratio: A scaling factor.
    Returns:
        A trimesh.Trimesh.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)
    vertices *= scale_ratio

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def rotate_x(mesh, degrees=270):
    """
    Returns a copy of the given mesh rotated on the x axis.
    Arguments:
        mesh:       A trimesh.Trimesh.
        degrees:    Rotation angle in degrees.
    """
    rotated = mesh.copy()
    rotated.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(degrees), (1, 0, 0), point=(0, 0, 0))
    )

    return rotated

def rotate_y(mesh, degrees=270):
    """
    Returns a copy of the given mesh rotated on the x axis.
    Arguments:
        mesh:       A trimesh.Trimesh.
        degrees:    Rotation angle in degrees.
    """
    rotated = mesh.copy()
    rotated.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(degrees), (0, 1, 0), point=(0, 0, 0))
    )

    return rotated

def rotate_z(mesh, degrees=270):
    """
    Returns a copy of the given mesh rotated on the x axis.
    Arguments:
        mesh:       A trimesh.Trimesh.
        degrees:    Rotation angle in degrees.
    """
    rotated = mesh.copy()
    rotated.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(degrees), (0, 0, 1), point=(0, 0, 0))
    )

    return rotated

def mesh_to_sdf(mesh, resolution):
    """
    Convert a mesh to a signed distance field.
    Arguments:
        mesh:       A trimesh.Trimesh.
        resolution: The resolution with which to create the SDF.
        **kwargs:   Additional keyword arguments passed to mesh2sdf.compute.
    Returns:
        A np.array.
    """
    return mesh2sdf_fix.compute(
        vertices=mesh.vertices, faces=mesh.faces, size=resolution
    )

#############################################################################


def compute_lfd(mesh_a, mesh_b):
    """
    Returns a Light‐Field Distance between two meshes.
    Arguments may be either trimesh.Trimesh objects or file‐paths (str or Path).
    """
    # Remove Executable_0 if it exists
    exec_path = '/home/rayhub-user/.conda/envs/wala/lib/python3.10/site-packages/lfd/Executable_0'
    if os.path.exists(exec_path):
        try:
            if os.path.isdir(exec_path):
                shutil.rmtree(exec_path)
            else:
                os.remove(exec_path)
        except Exception as e:
            print(f"Warning: Could not remove {exec_path}: {e}")

    # if user passed paths, load them
    if isinstance(mesh_a, (str, pathlib.Path)):
        mesh_a = trimesh.load(str(mesh_a), force='mesh')
    if isinstance(mesh_b, (str, pathlib.Path)):
        mesh_b = trimesh.load(str(mesh_b), force='mesh')

    # Try the LightFieldDistance class approach
    lfd_value: float = LightFieldDistance(verbose=True).get_distance(
        mesh_a.vertices, mesh_a.faces,
        mesh_b.vertices, mesh_b.faces,
        0
    )
    return lfd_value


def compare_local_obj_folders(
    folder_a, folder_b,
    suffix_a=".obj", suffix_b=".obj",
    verbose=True, grid_size=256,
    experiment=None,
    rotate=True,
    s3_bucket_a=None, s3_prefix_a=None,
    s3_bucket_b=None, s3_prefix_b=None
):
    """
    Compare two local folders: for each common base name, compute SDF IoU
    of <base><suffix_a> vs <base><suffix_b>, applying rotations to A if rotate is True.
    If files are missing locally, fetch them from S3 using provided bucket/prefix.
    Logs each IoU value to Comet if experiment is provided.
    """
    # Sync from S3 if requested and folder does not exist or is empty
    if s3_bucket_a and s3_prefix_a:
        if not os.path.exists(folder_a) or not os.listdir(folder_a):
            print(f"Syncing {folder_a} from S3 bucket {s3_bucket_a} with prefix {s3_prefix_a}")
            sync_s3_folder_to_local(s3_bucket_a, s3_prefix_a, folder_a)
    if s3_bucket_b and s3_prefix_b:
        if not os.path.exists(folder_b) or not os.listdir(folder_b):
            print(f"Syncing {folder_b} from S3 bucket {s3_bucket_b} with prefix {s3_prefix_b}")
            sync_s3_folder_to_local(s3_bucket_b, s3_prefix_b, folder_b)

    files_a = [f for f in os.listdir(folder_a) if f.endswith(suffix_a)]
    files_b = [f for f in os.listdir(folder_b) if f.endswith(suffix_b)]
    map_a = {f[:-len(suffix_a)]: f for f in files_a}
    map_b = {f[:-len(suffix_b)]: f for f in files_b}
    common = sorted(set(map_a) & set(map_b))
    if verbose:
        print(f"Found {len(common)} common pairs in\n  A: {folder_a}\n  B: {folder_b}\n")

    scores = []
    for idx, base in enumerate(common, start=1):
        fA = os.path.join(folder_a, map_a[base])
        fB = os.path.join(folder_b, map_b[base])
        if verbose:
            print(f"[{idx}/{len(common)}] {map_a[base]} vs {map_b[base]} → ", end="")
        try:
            mesh_a = trimesh.load(fA, force='mesh')
            mesh_b = trimesh.load(fB, force='mesh')
            if rotate:
                mesh_a = rotate_x(mesh_a, -90)
            mesh_a = scale_to_unit_cube(mesh_a, scale_ratio=0.9)
            mesh_b = scale_to_unit_cube(mesh_b, scale_ratio=0.9)
            sdf_a = mesh_to_sdf(mesh_a, grid_size)
            sdf_b = mesh_to_sdf(mesh_b, grid_size)
            iou = compute_iou(sdf_a, sdf_b)
        except Exception as e:
            print(f"Error comparing {fA} and {fB}: {e}")
            iou = 0.0
        scores.append(iou)
        if verbose:
            print(f"SDF IoU={iou:.4f}")
        if experiment is not None:
            try:
                experiment.log_metric("iou_value", iou, step=idx)
                experiment.log_parameter("iou_pair_name", f"{map_a[base]} vs {map_b[base]}", step=idx)
            except Exception as e:
                if verbose:
                    print(f"Comet logging failed for pair {map_a[base]} vs {map_b[base]}: {e}")

    if not scores:
        print("No matching pairs found.")
        return 0.0
    avg = sum(scores) / len(scores)
    print(f"\nAverage SDF IoU across {len(scores)} pairs = {avg:.4f}")
    return avg

def compare_local_obj_folders_lfd(
    folder_a, folder_b,
    suffix_a=".obj", suffix_b=".obj",
    verbose=True,
    experiment=None,
    rotate=True,
    s3_bucket_a=None, s3_prefix_a=None,
    s3_bucket_b=None, s3_prefix_b=None
):
    """
    Compare two local folders: for each common base name, compute the
    Light‐Field Distance (LFD) of <base><suffix_a> vs <base><suffix_b>.
    Optionally rotates mesh A if rotate is True.
    If files are missing locally, fetch them from S3 using provided bucket/prefix.
    """
    # Sync from S3 if requested and folder does not exist or is empty
    if s3_bucket_a and s3_prefix_a:
        if not os.path.exists(folder_a) or not os.listdir(folder_a):
            print(f"Syncing {folder_a} from S3 bucket {s3_bucket_a} with prefix {s3_prefix_a}")
            sync_s3_folder_to_local(s3_bucket_a, s3_prefix_a, folder_a)
    if s3_bucket_b and s3_prefix_b:
        if not os.path.exists(folder_b) or not os.listdir(folder_b):
            print(f"Syncing {folder_b} from S3 bucket {s3_bucket_b} with prefix {s3_prefix_b}")
            sync_s3_folder_to_local(s3_bucket_b, s3_prefix_b, folder_b)

    files_a = [f for f in os.listdir(folder_a) if f.endswith(suffix_a)]
    files_b = [f for f in os.listdir(folder_b) if f.endswith(suffix_b)]
    map_a = {f[:-len(suffix_a)]: f for f in files_a}
    map_b = {f[:-len(suffix_b)]: f for f in files_b}
    common = sorted(set(map_a) & set(map_b))

    if verbose:
        print(f"Found {len(common)} common pairs in\n  A: {folder_a}\n  B: {folder_b}\n")

    distances = []
    for idx, base in enumerate(common, start=1):
        fA = os.path.join(folder_a, map_a[base])
        fB = os.path.join(folder_b, map_b[base])
        if verbose:
            print(f"[{idx}/{len(common)}] {map_a[base]} vs {map_b[base]} → ", end="")
        try:
            mesh_a = trimesh.load(fA, force='mesh')
            mesh_b = trimesh.load(fB, force='mesh')
            if rotate:
                mesh_a = rotate_x(mesh_a, -90)
            mesh_a = scale_to_unit_cube(mesh_a, scale_ratio=0.9)
            mesh_b = scale_to_unit_cube(mesh_b, scale_ratio=0.9)
            dist = compute_lfd(mesh_a, mesh_b)
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            dist = float('nan')
        distances.append(dist)
        if verbose:
            print(f"LFD={dist}")
        if experiment is not None:
            try:
                experiment.log_metric("lfd_distance", dist, step=idx)
                experiment.log_parameter("lfd_pair_name", f"{map_a[base]} vs {map_b[base]}", step=idx)
            except Exception as e:
                if verbose:
                    print(f"Comet logging failed for pair {map_a[base]} vs {map_b[base]}: {e}")

    if not distances:
        if verbose:
            print("No matching pairs found.")
        return None

    valid = [d for d in distances if not (d is None or (isinstance(d, float) and np.isnan(d)))]
    avg = sum(valid) / len(valid) if valid else float('nan')
    if verbose:
        print(f"\nAverage LFD across {len(valid)} pairs = {avg}")
    return avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D mesh similarity metrics.")
    parser.add_argument("--metric", choices=["lfd", "iou"], default="lfd", help="Metric to compute: 'lfd' or 'iou'")
    parser.add_argument("--folder_a", type=str, default="/Google_Dataset_Objects/", help="Path to first folder")
    parser.add_argument("--folder_b", type=str, default="/Original_Model_Objects/", help="Path to second folder")
    parser.add_argument("--suffix_a", type=str, default=".obj", help="Suffix for files in folder_a")
    parser.add_argument("--suffix_b", type=str, default=".obj", help="Suffix for files in folder_b")
    parser.add_argument("--grid_size", type=int, default=256, help="Grid size for IoU computation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--comet", action="store_true", default=False, help="Enable Comet logging")
    parser.add_argument("--no-rotate", action="store_true", default=False, help="Disable mesh rotation")
    parser.add_argument("--s3_bucket_a", type=str, default=None, help="S3 bucket for folder_a")
    parser.add_argument("--s3_prefix_a", type=str, default=None, help="S3 prefix for folder_a")
    parser.add_argument("--s3_bucket_b", type=str, default=None, help="S3 bucket for folder_b")
    parser.add_argument("--s3_prefix_b", type=str, default=None, help="S3 prefix for folder_b")

    args = parser.parse_args()

    experiment = None
    if args.comet:
        try:
            from comet_ml import start
            experiment = start(
                api_key="mqrUAXjKBRul24uX6pxR3gRHX*eyJiYXNlVXJsIjoiaHR0cHM6Ly9jb21ldC5kZXYuY2xvdWRvcy5hdXRvZGVzay5jb20ifQ",
                project_name="evals",
                workspace="alessandro-giuliano"
            )
            experiment.log_parameters({
                "metric": args.metric,
                "folder_a": args.folder_a,
                "folder_b": args.folder_b,
                "suffix_a": args.suffix_a,
                "suffix_b": args.suffix_b,
                "grid_size": args.grid_size if args.metric == "iou" else None,
                "rotate": not args.no_rotate,
                "s3_bucket_a": args.s3_bucket_a,
                "s3_prefix_a": args.s3_prefix_a,
                "s3_bucket_b": args.s3_bucket_b,
                "s3_prefix_b": args.s3_prefix_b
            })
        except Exception as e:
            print(f"Comet logging not enabled: {e}")

    if args.metric == "lfd":
        avg_lfd = compare_local_obj_folders_lfd(
            args.folder_a,
            args.folder_b,
            suffix_a=args.suffix_a,
            suffix_b=args.suffix_b,
            verbose=args.verbose,
            experiment=experiment,
            rotate=not args.no_rotate,
            s3_bucket_a=args.s3_bucket_a,
            s3_prefix_a=args.s3_prefix_a,
            s3_bucket_b=args.s3_bucket_b,
            s3_prefix_b=args.s3_prefix_b
        )
        print("Average LFD:", avg_lfd)
        if experiment:
            try:
                experiment.log_metric("average_lfd", avg_lfd)
            except Exception as e:
                print(f"Comet logging failed: {e}")

    elif args.metric == "iou":
        avg_iou = compare_local_obj_folders(
            args.folder_a,
            args.folder_b,
            suffix_a=args.suffix_a,
            suffix_b=args.suffix_b,
            verbose=args.verbose,
            grid_size=args.grid_size,
            experiment=experiment,
            rotate=not args.no_rotate,
            s3_bucket_a=args.s3_bucket_a,
            s3_prefix_a=args.s3_prefix_a,
            s3_bucket_b=args.s3_bucket_b,
            s3_prefix_b=args.s3_prefix_b
        )
        print("Average IoU (local):", avg_iou)
        if experiment:
            try:
                experiment.log_metric("average_iou", avg_iou)
            except Exception as e:
                print(f"Comet logging failed: {e}")

################################################################################
# Usage shortcut:
# xvfb-run /home/rayhub-user/.conda/envs/wala/bin/python /home/rayhub-user/Optim-WaLa/Optim/Eval.py  --metric lfd --verbose --comet --no-rotate
# Note needs Edward better fork https://github.com/edward1997104/light-field-distance
# s3://giuliaa-optim//TRT/Google_Dataset_Outputs/15.1/
# s3://giuliaa-optim/
