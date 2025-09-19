IMAGE_NAME := ros2-humble-image
CONTAINER_NAME := ros2-humble-container

build-image:
	docker build -t $(IMAGE_NAME) .

run-container:
	docker run --rm -it --privileged -v /dev/bus/usb:/dev/bus/usb --group-add video --name $(CONTAINER_NAME) -v "$(shell pwd)":/app -w /app $(IMAGE_NAME) /bin/bash

run-container-x11:
	xhost +local:docker
	docker run --rm -it -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix --privileged -v /dev/bus/usb:/dev/bus/usb --group-add video  --name $(CONTAINER_NAME) -v "$(shell pwd)":/app -w /app $(IMAGE_NAME) /bin/bash

attach-shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

change-ownership:
	sudo chown -R $(id -u):$(id -g) .

clean:
	docker rm -f $(CONTAINER_NAME)
	docker rmi $$(docker images -q $(IMAGE_NAME))

.PHONY: build-image run-container clean change-ownership attach-shell run-container-x11
