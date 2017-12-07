# yolov2

yolo v2 is an object detection algorithm using deep learning.

This repository contains learning and testing in keras and tensorflow.

<div align="center">
<img src="https://user-images.githubusercontent.com/27678705/33706213-b569d890-db76-11e7-9dfa-1aae4bfd1dc7.png" alt="result">
</div>

### ToDo

- refactor trainning code
- tracking function(SORT and Deep SORT) based object detection of yolov2
- demo gif

## Prerequirement

- python3.6
- opencv for python2 involving movie function
- dataset(image and annotation using XML format)
- keras(tensorflow)

## Usage

### train

If you want to learn with your dataset you need to edit config.json.

Please prepare the annotation in XML format with reference to yolov 2 formula.

` python train.py -c config.json`

### test

` python predict.py -c config.json -w path/to/pretrained_weights -i path/to/image -o path/to/output_dir`

### tracking

***todo***

Tracking is performed by SORT algorithm using Kalman filter.

## DEMO

***todo***
