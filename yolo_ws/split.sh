#!/usr/bin/env bash

YOLOWS_DIR=$(dirname `realpath $0`)
DATASET_DIR=$YOLOWS_DIR/dataset_raw

SEED=1

VALID_RATIO=0.1
TEST_RATIO=0.1

for split in train valid test; do
	mkdir -p "$YOLOWS_DIR/$split/images"
	mkdir -p "$YOLOWS_DIR/$split/labels"
done


images_base=($(ls "$DATASET_DIR"/images/*.png | xargs -n 1 basename | sed 's/\.png$//'))
images_base_shuf=($(printf "%s\n" "${images_base[*]}" | shuf --random-source=<(yes $SEED)))

dataset_size=${#images_base_shuf[@]}
valid_count=$(echo "($dataset_size * 0.1) / 1" | bc)
test_count=$(echo "($dataset_size * 0.1) / 1" | bc)
train_count=$(($dataset_size - valid_count - test_count))

for ((i=0; i<dataset_size; i++)); do
	name="${images_base_shuf[i]}"
	png_path="$DATASET_DIR/images/${images_base_shuf[i]}.png"
	txt_path="$DATASET_DIR/labels/${images_base_shuf[i]}.txt"
	if (( i < valid_count )); then
		set_name="valid"
	elif (( i < valid_count + test_count )); then
		set_name="test"
	else
		set_name="train"
	fi
	cp "$png_path" "$YOLOWS_DIR/$set_name/images/"
	cp "$txt_path" "$YOLOWS_DIR/$set_name/labels/"
done
