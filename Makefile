IMAGE_NAME := ros2-humble-image
CONTAINER_NAME := ros2-humble-container

build-image:
	docker build --platform linux/amd64 -t $(IMAGE_NAME) --build-arg USER_ID=$(shell id -u) --build-arg GROUP_ID=$(shell id -g) .

run-container:
	docker run --platform linux/amd64 --rm -it --name $(CONTAINER_NAME) -v "$(shell pwd)":/app -w /app $(IMAGE_NAME) /bin/bash

clean:
	docker rm -f $(CONTAINER_NAME)
	docker rmi $$(docker images -q $(IMAGE_NAME))

.PHONY: build-image run-container clean
