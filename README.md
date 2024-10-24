# WaLa
![Static Badge](https://img.shields.io/badge/ICLR-2025-blue) [![arXiv](https://img.shields.io/badge/arXiv-2401.11067-b31b1b.svg)](https://arxiv.org/abs/2401.11067) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1W5zPXw9xWNpLTlU5rnq7g3jtIA2BX6aC?usp=sharing)
[![Huggingface space](https://img.shields.io/badge/🤗-Huggingface-yello.svg)](https://huggingface.co/models?search=ADSKAILab/WaLa)

This is the official codebase for the ICLR paper "**WAVELET LATENT DIFFUSION (WALA): BILLION-
PARAMETER 3D GENERATIVE MODEL WITH COM-PACT WAVELET ENCODINGS**"


### [Project](https://www.research.autodesk.com/publications/generative-ai-make-a-shape/) [Page](https://), [ICLR paper](https://), [Models](https://huggingface.co/models?search=ADSKAILab/WaLa), [Colab demo](https://colab.research.google.com/drive/1W5zPXw9xWNpLTlU5rnq7g3jtIA2BX6aC?usp=sharing)

### Tasks
- [ ] Single-view to 3D inference code
- [ ] Multi-view to 3D inference code
- [ ] 16³ resolution Voxel to 3D inference code
- [ ] Google Colab demo
- [ ] Point cloud to 3D inference code
- [ ] Unconditional 3D generation inference code

## Getting Started

### Installation
- Python >= 3.10
- Install CUDA if available
- Install PyTorch according to your platform: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
- Install other dependencies by `pip install -r requirements.txt`

For example, on AWS EC2 instances with PyTorch Deep learning AMI, you can setup the environment as follows:
```
conda create -n wala python==3.10
conda activate wala
pip install -r requirements.txt
```

### Inference

### Single-view to 3D

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/ADSKAILab/WaLa-SV-1B)

The input data for this method is a single-view image of a 3D object.

```sh
python run.py --model_name ADSKAILab/WaLa-SV-1B --images examples/single_view/table.png --output_dir examples --output_format obj
```

### Multi-view to 3D
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/ADSKAILab/WaLa-RGB4-1B)

For multi-view input, the model utilizes multiple images of the same object captured from different camera angles. These images should be named according to the index of the camera view parameters as described in [Data Formats](#data-formats)

```sh
python run.py --model_name ADSKAILab/WaLa-RGB4-1B --multi_view_images examples/multi_view/000.png examples/multi_view/006.png examples/multi_view/010.png examples/multi_view/026.png --output_dir examples --output_format obj
```


### Voxel to 3D (16³ Resolution )
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/ADSKAILab/WaLa-VX16-1B)

This model uses a voxelized representation of the object with a resolution of 16³. The voxel file is a JSON containing the following keys: `resolution`, `occupancy`, and `color`

```sh
python run.py --model_name ADSKAILab/WaLa-VX16-1B --voxel_files examples/voxel/horse_16.json --output_dir examples --output_format obj
```


### Data Formats

- **Single-View Input:** A single image file (e.g., `.png`, `.jpg`) depicting the 3D object.
- **Multi-View Input:** A set of image files taken from different camera angles. The filenames correspond to specific camera parameters. Below is a table that maps the index of each image to its corresponding camera rotation and elevation:

  | **Index** | **Rotation (degrees)** | **Elevation (degrees)** |
  |-----------|------------------------|-------------------------|
  | 0         | 57.37                  | 13.48                   |
  | 1         | 36.86                  | 6.18                    |
  | 2         | 11.25                  | 21.62                   |
  | 3         | 57.27                  | 25.34                   |
  | 4         | 100.07                 | 9.10                    |
  | 5         | 116.91                 | 21.32                   |
  | 6         | 140.94                 | 12.92                   |
  | 7         | 99.88                  | 3.57                    |
  | 8         | 5.06                   | 11.38                   |
  | 9         | 217.88                 | 6.72                    |
  | 10        | 230.76                 | 13.27                   |
  | 11        | 180.99                 | 23.99                   |
  | 12        | 100.59                 | -6.37                   |
  | 13        | 65.29                  | -2.70                   |
  | 14        | 145.70                 | 6.61                    |
  | 15        | 271.98                 | 0.15                    |
  | 16        | 284.36                 | 5.84                    |
  | 17        | 220.28                 | 0.07                    |
  | 18        | 145.86                 | -1.18                   |
  | 19        | 59.08                  | -13.59                  |
  | 20        | 7.35                   | 0.51                    |
  | 21        | 7.06                   | -7.82                   |
  | 22        | 146.05                 | -15.43                  |
  | 23        | 182.55                 | -5.17                   |
  | 24        | 341.95                 | 3.29                    |
  | 25        | 353.64                 | 9.75                    |
  | 26        | 319.81                 | 16.44                   |
  | 27        | 233.76                 | -8.56                   |
  | 28        | 334.96                 | -2.65                   |
  | 29        | 207.67                 | -16.79                  |
  | 30        | 79.72                  | -21.20                  |
  | 31        | 169.69                 | -26.77                  |
  | 32        | 237.16                 | -27.06                  |
  | 33        | 231.72                 | 25.91                   |
  | 34        | 284.84                 | 23.44                   |
  | 35        | 311.22                 | -14.09                  |
  | 36        | 285.15                 | -7.42                   |
  | 37        | 257.11                 | -14.38                  |
  | 38        | 319.14                 | -23.75                  |
  | 39        | 355.62                 | -9.06                   |
  | 40        | 0.00                   | 60.00                   |
  | 41        | 40.00                  | 60.00                   |
  | 42        | 80.00                  | 60.00                   |
  | 43        | 120.00                 | 60.00                   |
  | 44        | 160.00                 | 60.00                   |
  | 45        | 200.00                 | 60.00                   |
  | 46        | 240.00                 | 60.00                   |
  | 47        | 280.00                 | 60.00                   |
  | 48        | 320.00                 | 60.00                   |
  | 49        | 360.00                 | 60.00                   |
  | 50        | 0.00                   | -60.00                  |
  | 51        | 90.00                  | -60.00                  |
  | 52        | 180.00                 | -60.00                  |
  | 53        | 270.00                 | -60.00                  |
  | 54        | 360.00                 | -60.00                  |

- **Voxel Input:** A JSON file containing a voxelized representation of the object. The JSON includes:
  - **resolution:** The grid size of the voxel space (e.g., 16 or 32).
  - **occupancy:** The indices of occupied voxels.
  - **color:** The RGB values for each occupied voxel.


### Google Colab Demo

To quickly try out the Make-A-Shape model without setting up your local environment, check out the [Google Colab Demo](https://colab.research.google.com/drive/1XIoeanLjXIDdLow6qxY7cAZ6YZpqY40d?usp=sharing).


## Citation 

**BibTeX:**

