# Get started



3D object detection from infrastructure for autonomous driving




## Installation

Clone this repository and its submodules:

```
git clone https://github.com/ramajoballester/sensus-loci.git
cd sensus-loci
git submodule init
git submodule update
```

And follow the mmdetection3d installation instructions from [official website](https://mmdetection3d.readthedocs.io/en/latest/get_started.html).

After installing mmdetection3d, install sensus-loci:

```
pip install -e .
```

## ROS 2 support

To install the ros_sensus ROS2 package, go to the ros directory ```./sensus/ros``` and follow the instructions for [ROS2 installation](ros_readme.md).


## Known issues

- Downgrade numba to 0.55 to avoid cuda drivers version mismatch. Seems solved.
- Problems with **compiling** wheels for: ```mim install mmcv-full``` are due to pytorch shipping nvcc in the latest release. Fix it by indicating its true path 
```export PATH=/usr/local/cuda/bin:$PATH```. Bug [#2684](https://github.com/microsoft/DeepSpeed/issues/2684)
