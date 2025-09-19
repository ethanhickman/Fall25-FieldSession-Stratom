IMAGE_NAME := ros2-humble-image
CONTAINER_NAME := ros2-humble-container

build-image:
	docker build -t $(IMAGE_NAME) .

run-container:
	docker run --rm -it --name $(CONTAINER_NAME) -v "$(shell pwd)":/app -w /app $(IMAGE_NAME) /bin/bash

clean:
	docker rm -f $(CONTAINER_NAME)
	docker rmi $$(docker images -q $(IMAGE_NAME))

.PHONY: build-image run-container clean
