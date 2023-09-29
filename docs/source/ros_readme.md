# ROS2 Installation

From ```perception_manager/sensus/ros```:

```
colcon build
source ./install/setup.bash
```

## Execution

Perception manager reads PointCloud2 messages from ```pointcloud_topic``` defined in ```src/ros_sensus/config/ros/perception_manager_config.json``` and publishes perception_interfaces/msg/Detection messages to ```detection_topic```:

```
ros2 run ros_sensus perception_manager
```

Perception visualizer reads from ```detection_topic``` and publishes ```vision_msgs/msg/BoundingBox3DArray``` messages to ```visualization_topic```, so they can be visualized in RViz2:

```
ros2 run ros_sensus perception_visualizer
```

## Visualization

From ```perception_manager/sensus/ros```:

```
rviz2 -d ./src/ros_sensus/config/rviz/perception_visualizer.rviz
```


## Notes

- Install [vision_msgs package manually for source compiled ROS2](https://github.com/ros-perception/vision_msgs)

### Conda-ROS compatibility
- Suspect that ```source ros2_humble/install/setup.bash``` is sourcing local workspace 
somehow
- Adding 
```
[build_scripts]
executable = /usr/bin/env python3
```
to ```setup.cfg``` in local workspace solved the conda-ROS compatibility issue
- Activating conda env before ```colcon build``` does not change behaviour


### Experiments

- a1: +6
- a2: +8
- a3: +10
- a4: +4 (acceptable)
- a5: +2 (slightly worse than a4)
- a6: +4 (+3 was terrible)
- a7: +5 (awful)
- a8: +4.5 (bit worse than a4)
- a9: +3.5 (worse than a4)
- a10: +4.25

- nus_1: circle_nms voxel 0.075 +4 (best)
- nus_2: rotate nms voxel 0.075 +4 (bit worse than nus_1)
- nus_3: circle_nms voxel 0.075 +0 (no detections)
- nus_4: circle_nms voxel 0.075 +3 (worse than nus_1)
- nus_5: circle_nms voxel 0.075 +5 (same as nus_4)

- perception_manager_viz


## TODO

- [ ] Transform each bbox3d to bbox3d array for visualization
- [ ] Adapt point cloud to KITTI characteristics to improve accuracy