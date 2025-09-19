# Fall25-FieldSession-Stratom
Working repository for our field session project. 


#### Setup
1. First build the docker by running `make build-image`. (This step only needs to be run when first cloning the repo or after changes are made to the Dockerfile)
2. Then run the container with an interactive shell with `make run-container`
3. Inside the container shell, run `source /opt/ros/humble/setup.bash`
4. Change directories into the workspace and setup the build directories: `cd ros2_ws && colcon build` (This only needs to be run once per container; if you see build, install, and log directories in the ros2_ws, this command does not need to be run)

