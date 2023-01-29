# sensus-loci
3D object detection from infrastructure

# Installation

Option 1. Install [MMDetection3D](https://mmdetection3d.readthedocs.io/en/latest/getting_started.html)

- Requires python 3.8

Option 2. Init gitmodules and source mmdet_install.sh from sensus-loci root directory

- Problems with compiling wheels for: ```mim install mmcv-full``` are due to pytorch shipping nvcc in the latest release. Fix it by indicating its true path 
```export PATH=/usr/local/cuda/bin:$PATH```. Bug [#2684](https://github.com/microsoft/DeepSpeed/issues/2684)




# Explanation

## NuScenes Dataset

390k lidar sweeps: 40k in samples and 350k in sweeps