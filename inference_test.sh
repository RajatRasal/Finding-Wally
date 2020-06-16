#!/bin/bash

dt=$(date '+%d_%m_%Y__%H_%M_%S');
image_numbers=$1

for i in $image_numbers
do
  echo Image: $i
  python inference_with_dataset.py \
    -f ./data/data.csv \
    -g 7000 \
    -i $i \
    -r 2000 \
    -t 0.5 \
    -m ./saved_model \
    -l $dt 2> /tmp/null
done

