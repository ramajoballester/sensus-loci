# Get started

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Framajoballester%2Fsensus-loci%2Fmain%2Fpyproject.toml)
![Read the Docs](https://img.shields.io/readthedocs/sensus-loci)
![GitHub](https://img.shields.io/github/license/ramajoballester/sensus-loci)


3D object detection from infrastructure for autonomous driving. Check the [documentation](https://sensus-loci.readthedocs.io/en/latest/) for more information.


![Real-time visualization of the 3D object detection with ROS](/docs/images/ros_example.png)


## Installation

Clone this repository and its submodules:

```
git clone https://github.com/ramajoballester/sensus-loci.git
cd sensus-loci
git submodule update --init --recursive
```

And follow the mmdetection3d installation instructions from [official website](https://mmdetection3d.readthedocs.io/en/latest/get_started.html).

After installing mmdetection3d, install sensus-loci:

```
pip install -e .
```

For full documentation building support, install the full version of the package:

```
pip install -e .[full]
```

It is recommended to add symbolic links to each dataset folder in data/ directory inside ```sensus-loci``` and ```mmdetection3d``` to get the exact directory tree as in [mmdetection3d docs](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/supported_tasks/lidar_det3d.html#data-preparation). For example:

``` 
ln -s ~/path/to/dataset/ data/dataset
```


## Branches description

- main
- dev: just behind main


## Install DAIR-V2X version

Install the v0.17.1 version of mmdetection3d:
```bash
conda create -n sensus-dair python=3.7.*
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install mmcv-full==1.3.8
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
git clone https://github.com/open-mmlab/mmdetection3d.git --branch v0.17.1 --single-branch
```

Install the modified pypcd package:
```bash
git clone https://github.com/klintan/pypcd.git
cd pypcd
python setup.py install
```

Add DAIR-V2X to the python path (no installation is provided):
```bash
export PYTHONPATH=/path/to/dair-v2x:$PYTHONPATH
```

Convert DAIR to kitti format, infrastructure set:
```bash
python tools/dataset_converter/dair2kitti.py --source-root ~/datasets/DAIR/cooperative-vehicle-infrastructure/infrastructure-side/ --target-root ~/datasets/DAIR/cooperative-vehicle-infrastructure-kittiformat/infrastructure-side/ --split-path data/split_datas/cooperative-split-data.json --label-type lidar --sensor-view infrastructure --no-classmerge --temp-root ~/datasets/.tmp_file
```

and vehicle set:
```bash
python tools/dataset_converter/dair2kitti.py --source-root ~/datasets/DAIR/cooperative-vehicle-infrastructure/vehicle-side/ --target-root ~/datasets/DAIR/cooperative-vehicle-infrastructure-kittiformat/vehicle-side/ --split-path data/split_datas/cooperative-split-data.json --label-type lidar --sensor-view vehicle --no-classmerge --temp-root ~/datasets/.tmp_file
```

Generate infos with mmdetection3d (can be done with latest version). Infrastructure set:
```bash
python tools/create_data.py kitti --root-path ~/datasets/DAIR/cooperative-vehicle-infrastructure-kittiformat/infrastructure-side/ --out-dir ~/datasets/DAIR/cooperative-vehicle-infrastructure-kittiformat/infrastructure-side/ --extra-tag dair
```

and vehicle set:
```bash
python tools/create_data.py kitti --root-path ~/datasets/DAIR/cooperative-vehicle-infrastructure-kittiformat/vehicle-side/ --out-dir ~/datasets/DAIR/cooperative-vehicle-infrastructure-kittiformat/vehicle-side/ --extra-tag dair
```

Cooperative infrastructure set:
+----------------+--------+
| category       | number |
+----------------+--------+
| Pedestrian     | 19444  |
| Cyclist        | 9304   |
| Car            | 94936  |
| Van            | 11644  |
| Truck          | 4601   |
| Person_sitting | 0      |
| Tram           | 0      |
| Misc           | 0      |
+----------------+--------+

Cooperative-vehicle set:
+----------------+--------+
| category       | number |
+----------------+--------+
| Pedestrian     | 6207   |
| Cyclist        | 6284   |
| Car            | 64634  |
| Van            | 7785   |
| Truck          | 5689   |
| Person_sitting | 0      |
| Tram           | 0      |
| Misc           | 0      |
+----------------+--------+







# Install Open3D build with Jupyter support

Rebuild with all the packages installed in system

1. Install `cmake` (from [kitware repository](https://apt.kitware.com/)) and `gcc` (latest version with conda?)
2. Install npm, yarn and nodejs:

```
sudo apt install npm
sudo npm install -g yarn
sudo npm install -g n
sudo n stable
```

3. Download the [Open3D](https://github.com/isl-org/Open3D) source code:

```
git clone https://github.com/isl-org/Open3D.git
cd Open3D
```

4. Install system dependencies from `Open3D/util/install_deps_ubuntu.sh `

Maybe install OSMesa for headless support and GLFW (?):

```
sudo apt-get install libosmesa6-dev
sudo apt-get install libglfw3 ?
```

It might be required to reboot.

5. Activate conda and install dependencies:

```
pip install -r python/requirements_build.txt
pip install -r python/requirements_jupyter_build.txt
```

6. Build Open3D

```
mkdir build && cd build
```

Build with Jupyter support:
```
cmake -DBUILD_JUPYTER_EXTENSION=ON ..
```

With headless support:
```
cmake -DENABLE_HEADLESS_RENDERING=ON \
    -DBUILD_GUI=OFF \
    -DBUILD_WEBRTC=OFF \
    -DUSE_SYSTEM_GLEW=OFF \
    -DUSE_SYSTEM_GLFW=OFF \
    ..
```

7. Install Open3D

```
make install-pip-package -j$(nproc)
```


# ROS 2 support

To install the ros_sensus ROS2 package, go to the ros directory ```./sensus/ros``` and follow the instructions for [ROS2 installation](ros_readme.md).


# Known issues

- Numpy has to be downgraded to 1.22 #2821  (1.23 also works)
- Numba == 0.57.*


# TODO

- Create infos for DAIR-V2X classes. Currently, they are KITTI classes.