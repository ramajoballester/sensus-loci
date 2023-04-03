## Execution (legacy)

``` python sensus/ros/inference.py mmdetection3d/demo/data/kitti/kitti_000008.bin mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth ```


``` python sensus/ros/inference.py mmdetection3d/data/nuscenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin mmdetection3d/configs/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py mmdetection3d/checkpoints/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth ```


# Package creation ROS2

1. Create workspace dir with src folder
2. Create package:

``` ros2 pkg create --build-type ament_python ros_sensus ```

Or with complete command line arguments:

``` ros2 pkg create --build-type ament_python ros_sensus 
--dependencies rclpy std_msgs --maintainer-name "Your Name" 
--maintainer-email "Your Email" --license "GPL-3.0-only" 
--description "Your Description"
```

3. Place your code into the src/pkg_name/pkg_name folder. This folder is a python package
4. Adjust package.xml and setup.py for personal information. Can be done when creating package by command line arguments
5. Add ros dependencies to package.xml:

```
<exec_depend>rclpy</exec_depend>
<exec_depend>std_msgs</exec_depend>
```

6. Add python dependencies to setup.py (optional) like any other python package
7. Add entry point to setup.py:

```
entry_points={
    'console_scripts': [
        'talker = ros_sensus.python_script_name:main',
    ],
}
```

8. Check dependencies with:

``` rosdep install -i --from-path src --rosdistro humble -y ```

9. Build package with:

``` colcon build --packages-select ros_sensus ```

10. Source workspace:

``` source install/setup.bash ```

11. Launch another terminal and run your package:

``` ros2 run ros_sensus python_script_name ```


# TODO

- [ ] Transform each bbox3d to bbox3d array for visualization
- [ ] Adapt point cloud to KITTI characteristics to improve accuracy

# Notes

- Install [vision_msgs package manually for source compiled ROS2](https://github.com/ros-perception/vision_msgs)