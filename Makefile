IMAGE_NAME := ros2-humble-image
CONTAINER_NAME := ros2-humble-container
SOURCE_SETUP = . ./ros2_ws/install/setup.sh

# builds docker images with legacy build system due to needing cuda at buildtime for darknet
build-image:
	DOCKER_BUILDKIT=0 docker build -t $(IMAGE_NAME) .

# runs docker container based off of image with necessary configuration options
run-container:
	docker run --rm -it --ipc=host --runtime nvidia --gpus all --network host -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix --privileged --group-add video --name $(CONTAINER_NAME) -v "$(shell pwd)":/app -w /app $(IMAGE_NAME) /bin/bash

# runs container with xhost which is needed to see gui apps in container
run-container-x11:
	xhost +local:docker
	$(MAKE) run-container

# attaches additional shells to an already running container
attach-shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# files created in docker container are owned by root due to docker configuration
# this target will give ownership to local user when run outside of container
get-ownership:
	sudo chown -R $$(id -u):$$(id -g) .

clean:
	docker rm -f $(CONTAINER_NAME)
	docker rmi $$(docker images -q $(IMAGE_NAME))

check-ros-env:
	@if [ "$$ROS_VERSION" != "2" ]; then \
		echo "Error: not in ros2 environment!"; \
		exit 1; \
	fi

build-ros-nodes: check-ros-env
	@(cd ros2_ws && colcon build --symlink-install)

# starts the realsense camera driver
run-camera-driver: check-ros-env
	@ros2 launch realsense2_camera rs_align_depth_launch.py

# runs a given ros node in dunnage detection package with value of NODE
run-node: check-ros-env build-ros-nodes
	@if [ -z "$(NODE)" ]; then \
		echo "Error: NODE not set. Use make run-node NODE=<node_name>"; \
		exit 1; \
	fi
	@$(SOURCE_SETUP) && ros2 run dunnage_detection $(NODE)


.PHONY: build-image run-container clean get-ownership attach-shell run-container-x11 check-ros-env build-ros-nodes run-camera-driver run-node
