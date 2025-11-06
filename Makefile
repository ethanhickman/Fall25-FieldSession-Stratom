IMAGE_NAME := ros2-humble-image
CONTAINER_NAME := ros2-humble-container
SOURCE_SETUP = . ./ros2_ws/install/setup.sh

build-image:
	DOCKER_BUILDKIT=0 docker build -t $(IMAGE_NAME) .

run-container:
	docker run --rm -it --ipc=host --runtime nvidia --gpus all --network host -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix --privileged --group-add video --name $(CONTAINER_NAME) -v "$(shell pwd)":/app -w /app $(IMAGE_NAME) /bin/bash

run-container-x11:
	xhost +local:docker
	$(MAKE) run-container

attach-shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

change-ownership:
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

run-camera-driver: check-ros-env
	@ros2 launch realsense2_camera rs_align_depth_launch.py

run-camera-viewer: check-ros-env build-ros-nodes
	@$(SOURCE_SETUP) && ros2 run dunnage_detection rs_camera_viewer

run-bag-snapshot: check-ros-env build-ros-nodes
	@$(SOURCE_SETUP) && ros2 run dunnage_detection rs_bag_snapshot

run-bag-reader: check-ros-env build-ros-nodes
	@$(SOURCE_SETUP) && ros2 run dunnage_detection bag_reader

train-yolov9:
	@cd yolo_ws/yolov9 && python3 train.py --batch 16 --epochs 50 --img 640 --device 0 --min-items 0 --close-mosaic 15 --data ../data.yaml --weights ../weights/gelan-m.pt --cfg models/detect/gelan-m.yaml --hyp data/hyps/hyp.scratch-high.yaml

.PHONY: build-image run-container clean change-ownership attach-shell run-container-x11 check-ros-env build-ros-nodes run-camera-driver run-camera-viewer run-bag-snapshot run-bag-reader
