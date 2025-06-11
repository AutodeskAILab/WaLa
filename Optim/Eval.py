import boto3
import os
from pathlib import Path
import trimesh
import numpy as np
import torch

import mesh2sdf_fix 



import numpy as np

def normalize_mesh(mesh):
    m = mesh.copy()
    verts = m.vertices
    centroid = verts.mean(axis=0)
    verts_centered = verts - centroid
    scale = 1.0 / np.max(np.linalg.norm(verts_centered, axis=1))
    m.apply_translation(-centroid)
    m.apply_scale(scale)
    return m

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

def compare_local_obj_folders(
    folder_a, folder_b,
    suffix_a=".obj", suffix_b=".obj",
    verbose=True, grid_size=128
):
    """
    Compare two local folders: for each common base name, compute SDF IoU
    of <base><suffix_a> vs <base><suffix_b>, applying rotations to A.
    """
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
            # rotate the Google mesh:
            mesh_a = rotate_x(mesh_a, -90)
            mesh_a = rotate_y(mesh_a, -45)

            ## Normalize the mesh
            mesh_a = scale_to_unit_cube(mesh_a, scale_ratio=0.9)
            mesh_b = scale_to_unit_cube(mesh_b, scale_ratio= 0.9)
            # compute SDFs
            sdf_a = mesh_to_sdf(mesh_a, grid_size)
            sdf_b = mesh_to_sdf(mesh_b, grid_size)


            iou = compute_iou(sdf_a, sdf_b)

        except Exception as e:
            print(f"Error comparing {fA} and {fB}: {e}")
            iou = 0.0
        scores.append(iou)
        if verbose:
            print(f"SDF IoU={iou:.4f}")

    if not scores:
        print("No matching pairs found.")
        return 0.0
    avg = sum(scores) / len(scores)
    print(f"\nAverage SDF IoU across {len(scores)} pairs = {avg:.4f}")
    return avg

avg_iou = compare_local_obj_folders(
    "/Google_Dataset_Objects/",
    "/Original_Model_Objects/",
    suffix_a=".obj",
    suffix_b=".obj",
    verbose=True,
    grid_size=128)
print("Average IoU (local):", avg_iou)
