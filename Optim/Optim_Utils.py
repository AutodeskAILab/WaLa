import os
import shutil
import zipfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.diffusion_modules.dwt import DWTInverse3d
import mcubes
import time
import torch
import boto3

def download_s3_folder(bucket_name, s3_prefix, local_dir):
    """
    Download all files under a given S3 prefix to a local directory, preserving structure.
    Eg. download_s3_folder("giuliaa-optim", "/TRT/Google_Dataset_Outputs/10.1/", "/TRT_OBJ/10.1")
    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_prefix (str): Prefix (folder path) in the S3 bucket.
        local_dir (str or Path): Local directory to download files into.
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Skip "folders"
            if key.endswith('/'):
                continue
            # Compute local file path
            rel_path = os.path.relpath(key, s3_prefix)
            local_path = local_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {key} to {local_path}")
            with open(local_path, 'wb') as f:
                s3.download_fileobj(bucket_name, key, f)

def download_s3_file(bucket_name, s3_key, local_path):
    """
    Download a single file from S3 to a local path.
    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): Key (path) of the file in S3.
        local_path (str or Path): Local file path to save.
    """
    s3 = boto3.client('s3')
    local_path = str(local_path)
    with open(local_path, 'wb') as f:
        s3.download_fileobj(bucket_name, s3_key, f)

def upload_folder_to_s3(folder_path, bucket_name, s3_prefix=""):
    """
    Uploads all files in a folder (recursively) to an S3 bucket, preserving folder structure.

    Args:
        folder_path (str or Path): Path to the folder to upload.
        bucket_name (str): Name of the S3 bucket.
        s3_prefix (str): Optional prefix (folder) in the bucket.
    """
    s3 = boto3.client('s3')
    folder_path = Path(folder_path)
    for file_path in folder_path.rglob('*'):
        if file_path.is_file():
            # Compute S3 key to preserve folder structure
            relative_path = file_path.relative_to(folder_path)
            s3_key = f"{s3_prefix}/{relative_path}".lstrip('/') if s3_prefix else str(relative_path)
            with open(file_path, "rb") as f:
                s3.upload_fileobj(f, bucket_name, s3_key)
            print(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")

def upload_files_to_s3(file_paths, bucket_name, s3_prefix=""):
    """
    Uploads a list of files to an S3 bucket.
    # Example usage:
    #files = ["model_10.onnx.data", "model_10.onnx"]
    #upload_files_to_s3(files, "giuliaa-optim", s3_prefix="/ONNX/Steps_10")

    Args:
        file_paths (list): List of file paths (str or Path) to upload.
        bucket_name (str): Name of the S3 bucket.
        s3_prefix (str): Optional prefix (folder) in the bucket.
    """
    s3 = boto3.client('s3')
    for file_path in file_paths:
        file_path = Path(file_path)
        s3_key = f"{s3_prefix}/{file_path.name}" if s3_prefix else file_path.name
        with open(file_path, "rb") as f:
            s3.upload_fileobj(f, bucket_name, s3_key)
        print(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")


def find_and_move_matching_files(source_folder, check_folder, destination_folder, TRT=False):
    """
    Finds files in source_folder with names that match files in check_folder,
    and moves the matching files from source_folder to destination_folder.
    If TRT is True, expects source files to have '_trt' before .obj and matches them to check_folder files without '_trt'.

    Args:
        source_folder (str): Path to the folder containing files to potentially move.
        check_folder (str): Path to the folder containing the reference filenames.
        destination_folder (str): Path to the folder where matching files should be moved.
        TRT (bool): If True, match source files with '_trt.obj' to check files with '.obj'.
    """

    try:
        check_files = set(os.listdir(check_folder))
        print(f"[DEBUG] Files in check_folder ({check_folder}): {len(check_files)} found")

        os.makedirs(destination_folder, exist_ok=True)

        source_files = os.listdir(source_folder)
        print(f"[DEBUG] Files in source_folder ({source_folder}): {len(source_files)} found")

        matches = 0
        for filename in source_files:
            match_name = filename
            if TRT and filename.lower().endswith('_trt.obj'):
                # Remove '_trt' before .obj to get the base name for matching
                match_name = filename[:-8] + '.obj'  # Remove '_trt.obj', add '.obj'
            if match_name in check_files:
                matches += 1
                source_path = os.path.join(source_folder, filename)
                destination_path = os.path.join(destination_folder, filename)
                shutil.move(source_path, destination_path)
                print(f"Moved: {filename} from {source_folder} to {destination_folder}")
            else:
                print(f"[DEBUG] No match for: {filename} (match_name: {match_name})")

        print(f"[DEBUG] Total matches moved: {matches}")

    except FileNotFoundError:
        print("Error: One or more folders not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def check_folder_contents(folder_path):
    """
    Checks the contents of a folder and prints the filenames.

    Args:
        folder_path (str): The path to the folder.
    """
    try:
        filenames = os.listdir(folder_path)
        print(f"Contents of {folder_path}:")
        for filename in filenames:
            print(f"- {filename}")
    except FileNotFoundError:
        print(f"Error: Folder not found: {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")



def compare_file_sizes(folder1, folder2):
    """
    Compares files with the same name in two folders and prints their sizes and summary stats.
    """
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    common_files = files1 & files2

    if not common_files:
        print("No matching files found.")
        return

    total_size1 = 0
    total_size2 = 0

    print(f"{'Filename':40} {'Size in folder1 (bytes)':25} {'Size in folder2 (bytes)':25} {'Difference (bytes)':20}")
    print("-" * 110)
    for filename in sorted(common_files):
        path1 = os.path.join(folder1, filename)
        path2 = os.path.join(folder2, filename)
        size1 = os.path.getsize(path1)
        size2 = os.path.getsize(path2)
        diff = size1 - size2
        total_size1 += size1
        total_size2 += size2
        print(f"{filename:40} {size1:<25} {size2:<25} {diff:<20}")

    total_diff = total_size1 - total_size2
    percent_smaller = (total_diff / total_size1 * 100) if total_size1 else 0

    print("\nSummary:")
    print(f"Total size in folder1: {total_size1} bytes")
    print(f"Total size in folder2: {total_size2} bytes")
    print(f"Total reduction: {total_diff} bytes ({percent_smaller:.2f}% smaller)")


def extract_first_thumbnail(zip_dir, output_dir):
    """
    Extracts the first thumbnail image from each zip file in zip_dir
    and saves it to output_dir with the zip file's stem as the filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    for zip_path in Path(zip_dir).glob('*.zip'):
        with zipfile.ZipFile(zip_path, 'r') as z:
            thumbnail_files = [f for f in z.namelist() if f.startswith('thumbnails/') and not f.endswith('/')]
            if thumbnail_files:
                thumb_file = thumbnail_files[0]
                ext = Path(thumb_file).suffix
                out_name = zip_path.stem + ext
                out_path = Path(output_dir) / out_name
                with z.open(thumb_file) as source, open(out_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
                print(f"Extracted {thumb_file} from {zip_path.name} as {out_name}")



def export_obj_original(vertices: np.ndarray, triangles: np.ndarray, filename: str, flip_normals: bool = False):
    """
    Export a 3D mesh to a Wavefront (.obj) file.

    If `flip_normals` is True, reverses the order of the vertices in each face
    to flip the normals. Default is False.
    MCUBES ORIGINAL
    """

    with open(filename, 'w') as fh:

        for v in vertices:
            fh.write("v {} {} {}\n".format(*v))

        if not flip_normals:
            for f in triangles:
                fh.write("f {} {} {}\n".format(*(f + 1)))
        else:
            for f in triangles:
                fh.write("f {} {} {}\n".format(*(f[::-1] + 1)))


def export_obj_fast(vertices: np.ndarray, triangles: np.ndarray, filename: str, flip_normals: bool = False):
    """
    Export a 3D mesh to a Wavefront (.obj) file, optimized for speed.
    MCUBES MODIFIED
    """
    with open(filename, 'w') as fh:
        # Write vertices
        np.savetxt(fh, vertices, fmt='v %.6f %.6f %.6f')
        # Prepare faces (OBJ is 1-based indexing)
        if not flip_normals:
            np.savetxt(fh, triangles + 1, fmt='f %d %d %d')
        else:
            np.savetxt(fh, triangles[:, ::-1] + 1, fmt='f %d %d %d')

def benchmark_export(func, vertices, triangles, filename, flip_normals=False):
    start = time.time()
    func(vertices, triangles, filename, flip_normals)
    end = time.time()
    return end - start

def save_visualization_obj(image_name, obj_path, samples, experiment = None, idx = None):
        """Save a visualization object."""
        low, highs = samples
        
        args_max_depth = 3
        args_wavelet = 'bior6.8'
        args_padding_mode = 'constant'
        args_resolution = 256

        dwt_inverse_3d = DWTInverse3d(args_max_depth, args_wavelet,args_padding_mode )
        sdf_recon = dwt_inverse_3d((low, highs))

        m_time = time.time()
        vertices, triangles = mcubes.marching_cubes(
            sdf_recon.cpu().detach().numpy()[0, 0], 0.0
        )
        marching_time = time.time() - m_time
        print(f"Marching cubes time: {marching_time:.4f} seconds")

        # Save the mesh as an OBJ file
        vertices = (vertices / args_resolution) * 2.0 - 1.0
        triangles = triangles[:, ::-1]

        e_time = time.time()
        mcubes.export_obj(vertices, triangles, obj_path)  
        export_time = time.time() - e_time

        print(f"Export OBJ time: {export_time:.4f} seconds")
        print("Number of vertices:", len(vertices))
        print("Number of triangles:", len(triangles))


        try:
            experiment.log_metric(
                "mcubes.marching_cubes time", marching_time, step = idx
            )
            experiment.log_metric(
                "export obj time", export_time, step = idx
            )
            experiment.log_metric(
                "Number of Vertices", len(vertices), step = idx
            )
            experiment.log_metric(
                "Number of Triangles", len(triangles), step = idx
            )
        except:
            pass


class Optim_Visualizations():
    
    
    def retrieve_and_plot(experiment, metric_names=None, metric_display_names=None, plotting=True):
        
        if metric_names is None:
            metric_names = [
                "Extract Image",
                "Latent Diffusion Time",
                "Latent Decoding Time",
                "Wavelet Preparation Time",
                "Low to Highs conversion",
                "Inverse DWT time elapsed",    
                "mcubes.marching_cubes time",
                "export obj time",
                "Default Delta",
            ]

   

        if metric_display_names is None:
            metric_display_names = {
                "Extract Image": "Image Extraction Time",
                "Latent Diffusion Time": "Diffusion Process Time",
                "Latent Decoding Time": "Latent Decoding",
                "Wavelet Preparation Time": "Wavelet Prep",
                "Low to Highs conversion": "Low→High Conversion",
                "Inverse DWT time elapsed": "Inverse DWT Time",
                "mcubes.marching_cubes time": "Marching Cubes Time",
                "export obj time": "Object File Writing Time",
                "Default Delta": "Total Runtime"

            }



        all_metric_data = {}

        # Plot the distribution of each metric in a separate figure
        for metric_name in metric_names:
            # Retrieve metric data
            metric_data = experiment.get_metrics(metric_name)
            # Extract values and convert to float
            values = [float(point["metricValue"]) for point in metric_data]
            all_metric_data[metric_name] = values

            display_name = metric_display_names.get(metric_name, metric_name)
            
            if plotting:
                plt.figure(figsize=(10, 6))
                sns.histplot(values, kde=True, bins=20, label=display_name)
                plt.xlabel("Metric Value")
                plt.ylabel("Frequency")
                plt.title(f"Distribution of Metric: {display_name}")
                plt.legend()
                plt.grid()
                plt.show()
        
    

        return all_metric_data, metric_names, metric_display_names





    def get_stats(
        experiment, 
        metric_names, 
        sum_metric_names=None, 
        display=False
    ):
        """
        Retrieve and calculate statistics for each metric in an experiment.
        Returns the metric statistics dictionary and the Default Delta values.
        Optionally prints the statistics if display=True.

        Args:
            experiment: The experiment object.
            metric_names (list): List of all metric names to calculate statistics for.
            sum_metric_names (list, optional): List of metric names to include in the sum (e.g., only timing metrics).
            display (bool): Whether to print statistics.
        """
        metric_statistics = {}
        total_sum = 0
        do_sum = (sum_metric_names is not None)

        # Remove "Default Delta" from metric_names for stats calculation
        metric_names = [name for name in metric_names if name != "Default Delta"]

        if do_sum:
            sum_metric_names = [n for n in sum_metric_names if n != "Default Delta"]
        else:
            # disable summing
            sum_metric_names = []

        for metric_name in metric_names:
            # Retrieve metric data
            metric_data = experiment.get_metrics(metric_name)
            # if there’s no data for this metric, skip it
            if not metric_data:
                continue            # Retrieve metric data
            metric_data = experiment.get_metrics(metric_name)
            # Extract values and convert to float
            values = [float(point["metricValue"]) for point in metric_data]
            
            # Calculate statistics
            metric_stats = {
                "mean": np.mean(values),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values),
                "sum": np.sum(values)
            }
            metric_statistics[metric_name] = metric_stats

            # Only add to total_sum if summing is enabled
            if do_sum and metric_name in sum_metric_names:
                total_sum += metric_stats["sum"]

        # Retrieve the "Default Delta" metric
        default_delta_data = experiment.get_metrics("Default Delta")
        default_delta_values = [float(point["metricValue"]) for point in default_delta_data]
        default_delta_sum = np.sum(default_delta_values)

        if display:
            print("Metric Statistics:")
            for m, stats in metric_statistics.items():
                print(f"\n{m}: mean={stats['mean']:.4f}, "
                      f"median={stats['median']:.4f}, "
                      f"min={stats['min']:.4f}, "
                      f"max={stats['max']:.4f}, "
                      f"sum={stats['sum']:.4f}")

            # only show the aggregate if we actually summed
            if do_sum:
                print(f"\nTotal Sum of Selected Metrics: {total_sum:.4f}")
                print(f"Default Delta Sum: {default_delta_sum:.4f}")
                if np.isclose(total_sum, default_delta_sum, atol=20):
                    pct = (total_sum / default_delta_sum) * 100
                    print(f"The total sum is {pct:.2f}% of Default Delta.")
                else:
                    print("The total sum is NOT close to Default Delta.")

        return metric_statistics, default_delta_values





    def plot_metric_pie(
    metric_statistics, 
    metric_display_names, 
    sum_metric_names=None, 
    title='Contribution of Each Metric to Total Sum (%)'
    ):
        """
        Plots a pie chart showing the contribution of each selected metric to the total sum.
        Only metrics in sum_metric_names are included in the pie chart.

        Args:
            metric_statistics (dict): Dictionary of metric statistics.
            metric_display_names (dict): Mapping of metric names to display names.
            sum_metric_names (list, optional): List of metric names to include in the pie (e.g., only timing metrics).
            title (str): Title for the plot.
        """
        plt.rcParams['font.family'] = 'DejaVu Sans'

        labels = []
        sizes = []

        # If sum_metric_names is not provided, use all metrics
        if sum_metric_names is None:
            sum_metric_names = list(metric_statistics.keys())

        for metric_name in sum_metric_names:
            if metric_name in metric_statistics:
                display_name = metric_display_names.get(metric_name, metric_name)
                labels.append(display_name)
                sizes.append(metric_statistics[metric_name]["sum"])

        # Calculate percentages
        total = sum(sizes)
        if total == 0:
            print("Warning: Total sum of selected metrics is zero. Pie chart will not be meaningful.")
            return

        percentages = [size / total * 100 for size in sizes]

        # Sort by size (descending)
        sorted_indices = np.argsort(sizes)[::-1]
        labels = [labels[i] for i in sorted_indices]
        sizes = [sizes[i] for i in sorted_indices]
        percentages = [percentages[i] for i in sorted_indices]

        # Use an inverted (reversed) colormap for light slices
        cmap = plt.get_cmap('Blues_r')
        colors = [cmap(0.3 + 0.7 * i / (len(labels)-1)) for i in range(len(labels))]

        # Explode the largest slice
        explode = [0.06 if i == 0 else 0 for i in range(len(labels))]

        fig, ax = plt.subplots(figsize=(11, 8))
        fig.patch.set_facecolor('#222831')  # Dark background
        ax.set_facecolor('#222831')

        # Pie chart with white edge, no shadow
        wedges, texts, autotexts = ax.pie(
            sizes,
            colors=colors,
            startangle=140,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0.5 else '',
            textprops={'fontsize': 14, 'fontweight': 'bold', 'color': 'white'},
            explode=explode,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'antialiased': True}
        )

        # Add a legend with full labels and percentages
        legend_labels = [f"{label}: {percent:.1f}%" for label, percent in zip(labels, percentages)]
        legend = ax.legend(
            wedges, legend_labels, title="Metrics", loc="center left",
            bbox_to_anchor=(1.05, 0.5), fontsize=13, title_fontsize=15, frameon=False, labelcolor='white'
        )
        plt.setp(legend.get_title(), fontweight='bold', color='white')

        plt.title(title, fontsize=20, fontweight='bold', pad=20, color='white')
        plt.axis('equal')
        plt.tight_layout(pad=4)
        plt.show()




    def plot_metric_heatmap(
        metric_statistics,
        metric_display_names,
        default_delta_values,
        show_metric_names=None,
        include_min=True,
        include_max=True,
        include_median=True,
        title="Metric Statistics Heatmap"
    ):
        """
        Plots a heatmap of selected metric statistics.
        Args:
            metric_statistics (dict): Dictionary of metric statistics.
            metric_display_names (dict): Mapping of metric names to display names.
            default_delta_values (list): Values for the "Default Delta" metric.
            show_metric_names (list, optional): List of metric names to include in the heatmap. If None, show all.
            include_min (bool): Whether to include the min value.
            include_max (bool): Whether to include the max value.
            include_median (bool): Whether to include the median value.
            title (str): Title for the plot.
        """
        # Prepare columns based on options
        columns = ["Metric", "Mean"]
        if include_median:
            columns.append("Median")
        if include_min:
            columns.append("Min")
        if include_max:
            columns.append("Max")

        data = {col: [] for col in columns}

        # If show_metric_names is None, use all metrics
        if show_metric_names is None:
            show_metric_names = list(metric_statistics.keys())

        for metric_name in show_metric_names:
            if metric_name in metric_statistics:
                stats = metric_statistics[metric_name]
                display_name = metric_display_names.get(metric_name, metric_name)
                data["Metric"].append(display_name)
                data["Mean"].append(stats["mean"])
                if include_median:
                    data["Median"].append(stats["median"])
                if include_min:
                    data["Min"].append(stats["min"])
                if include_max:
                    data["Max"].append(stats["max"])

        # Add Default Delta statistics as "Total Runtime"
        default_delta_stats = {
            "mean": np.mean(default_delta_values),
            "median": np.median(default_delta_values),
            "min": np.min(default_delta_values),
            "max": np.max(default_delta_values)
        }
        data["Metric"].append("Total Runtime")
        data["Mean"].append(default_delta_stats["mean"])
        if include_median:
            data["Median"].append(default_delta_stats["median"])
        if include_min:
            data["Min"].append(default_delta_stats["min"])
        if include_max:
            data["Max"].append(default_delta_stats["max"])

        df = pd.DataFrame(data)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.set_index("Metric"), annot=True, fmt=".4f", cmap="Blues")
        plt.title(title)
        plt.show()




    def compare_metric_distribution(
        experiment1, experiment2, 
        metric_name, 
        label1="Experiment 1", 
        label2="Experiment 2", 
        display_name=None
    ):
        # Retrieve metric data for both experiments
        values1 = [float(point["metricValue"]) for point in experiment1.get_metrics(metric_name)]
        values2 = [float(point["metricValue"]) for point in experiment2.get_metrics(metric_name)]
        
        # Calculate mean and std
        mean1, std1 = np.mean(values1), np.std(values1)
        mean2, std2 = np.mean(values2), np.std(values2)
        
        # Print mean and std
        print(f"{label1}: mean = {mean1:.4f}, std = {std1:.4f}")
        print(f"{label2}: mean = {mean2:.4f}, std = {std2:.4f}")
        
        plt.figure(figsize=(10, 6))
        # KDE plots
        sns.kdeplot(values1, color='blue', label=label1, fill=True, alpha=0.3)
        sns.kdeplot(values2, color='orange', label=label2, fill=True, alpha=0.3)
        # Plot means
        plt.axvline(mean1, color='blue', linestyle='--', label=f"{label1} Mean")
        plt.axvline(mean2, color='orange', linestyle='--', label=f"{label2} Mean")
        
        # Annotate mean and std on the plot
        #plt.text(mean1, plt.ylim()[1]*0.9, f"μ={mean1:.2f}\nσ={std1:.2f}", color='blue', ha='center')
        #plt.text(mean2, plt.ylim()[1]*0.8, f"μ={mean2:.2f}\nσ={std2:.2f}", color='orange', ha='center')
        
        plt.xlabel("Metric Value")
        plt.ylabel("Density")
        plt.title(f"Distribution of Metric: {display_name or metric_name}")
        plt.legend()
        plt.grid()
        plt.show()

        # Print percentage change in means
        if mean1 != 0:
            pct_change = ((mean2 - mean1) / mean1) * 100
            print(f"Percentage change in mean: {pct_change:.2f}%")
        else:
            print("Cannot compute percentage change: mean of first experiment is zero.")
