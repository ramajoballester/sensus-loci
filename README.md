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
git submodule init
git submodule update
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

## Open3D build with Jupyter support

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


## ROS 2 support

To install the ros_sensus ROS2 package, go to the ros directory ```./sensus/ros``` and follow the instructions for [ROS2 installation](ros_readme.md).
