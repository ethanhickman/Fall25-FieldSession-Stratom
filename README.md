# Fall25-FieldSession-Stratom
Working repository for fall 2025 Stratom field session project. 


#### Setup
1. First build the docker by running `make build-image`. (This step only needs to be run when first cloning the repo or after changes are made to the Dockerfile)



#### Running Container
To run a container with an interactive bash shell, run `make run-container` or `make run-container-x11`. the `run-container-x11` target only works if you have xhost installed on your machine and uses x11 forwarding to display graphical output from the container. This target is required to be able to see camera output or run graphical applications.

Subsequent shells in the same container can be run with `make attach-shell`. The standard workflow is `make run-container-x11` followed by `make attach-shell` as needed for more shells.

Files created while inside the container will be configured to have root ownership. Running `make change-ownership` should change ownership to your user/group to allow for editing.



#### ROS2 Information
By default, the .bashrc for the root user should source the ros2 humble installation so this step should not be needed so long as you are in a root shell. However, if needed, run source /opt/ros/humble/setup.bash. Again, the bashrc file for the root user should run this automatically.

The ros package used by this project is the dunnage_detection package. new nodes can be added to the package at `ros2_ws/src/dunnage_detection/dunnage_detection/`. Once the node is created it needs to be added as a console script entry point in `ros2_ws/src/dunnage_detection/setup.py`. Then, the package must be build by going into the `ros2_ws` directory and running `colcon build --symlink-install`. Finally, the nodes can be run with `ros2 run dunnage_detection <node_name>`.

The realsense camera has existing ros drivers that can be run in a seperate shell to publish camera data to various topics. This can be run with `ros2 launch realsense2_camera rs_align_depth_launch.py`.

Additionally, there are make targets for running each node that handle sourcing and building as needed. For example there are `make run-camera-driver` and `make run-camera-viewer` targets.



#### Data Collection Method
Bag snapshots of dunnage consisting single messages from various topics created by the realsense camera driver will be collected with the `rs_bag_capture` node. The bag will be stored in a timestamped directory along with a yaml file containing measurements associated with the snapshot and any output from the `bag_reader` node. The measurements in the yaml file will be as follows:

- dunnage_voffset: The height of the dunnage relative to the realsense camera in meters. This will likely always be a negative measurement as the camera will be higher than the dunnage
- dunnage_hoffset: The horizontal position of the dunnage relative to the realsense camera in meters. If the dunnage is more to the left in the image, it will be negative while if the dunnage is more on the right, it will be positive.
- dunnage_doffset: The depth distance the dunnage is relative to the realsense camera in meters. This should always be positive as the dunnage should always be somewhere in front of the camera.
- dunnage_angle: The angle of the dunnage relative to the realsense camera in degrees

*NOTE:* The dunnage position will be the position of the top right corner of the face of the center piece of dunnage. The reference point of the camera will be the front of the color camera lens.