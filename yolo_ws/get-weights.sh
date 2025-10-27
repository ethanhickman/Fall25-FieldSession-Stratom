#!/usr/bin/env bash

YOLOWS_DIR=$(dirname `realpath $0`)
WEIGHTS_DIR=$YOLOWS_DIR/weights

mkdir -p $WEIGHTS_DIR

curl -L -# -o $WEIGHTS_DIR/gelan-m.pt https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-m.pt
curl -L -# -o $WEIGHTS_DIR/yolov9-m.pt https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m.pt
