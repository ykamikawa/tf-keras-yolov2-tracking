import argparse
import json
import os
import time

import cv2
import numpy as np
from tqdm import tqdm

from frontend import YOLO
from preprocessing import parse_annotation
from utils import draw_boxes


def predict(args):
    config_path = args.conf
    weights_path = args.weights
    image_path = args.input

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)
    # build model
    yolo = YOLO(
        architecture=config["model"]["architecture"],
        input_size=config["model"]["input_size"],
        labels=config["model"]["labels"],
        max_box_per_image=config["model"]["max_box_per_image"],
        anchors=config["model"]["anchors"],
    )

    # load pretrained model
    print(weights_path)
    yolo.load_weights(weights_path)

    # predict bounding box
    if image_path[-4:] == ".mp4":
        input_file = os.path.basename(args.input)
        video_out = (
            args.output
            + config["model"]["architecture"].replace(" ", "")
            + "_"
            + input_file[:-4]
            + ".mp4"
        )

        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(
            video_out, cv2.VideoWriter_fourcc(*"MPEG"), 30.0, (frame_w, frame_h)
        )

        # start time
        start = time.time()
        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            # boxes is list of box. normalize to 0~1 with input shape
            # box.x: xmin, box.y: ymin, box.w: box width, box.h: box height
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config["model"]["labels"])

            video_writer.write(np.uint8(image))
        # elapse time
        elapse_time = time.time()

        print(nb_frames, " frame are found")
        print("time: {}".format(elapse_time) + "[sec]")

        video_reader.release()
        video_writer.release()
    else:
        # start rime
        start = time.time()
        # load and input image
        image = cv2.imread(image_path)
        boxes = yolo.predict(image)

        # elapse time
        elapse_time = time.time() - start
        print(len(boxes), "boxes are found")
        print("time: {}".format(elapse_time) + "[sec]")

        # draw outputs
        image = draw_boxes(image, boxes, config["model"]["labels"])

        input_file = os.path.basename(args.input)
        cv2.imwrite(
            args.output
            + config["model"]["architecture"].replace(" ", "")
            + "_"
            + input_file[:-4]
            + ".png",
            image,
        )


if __name__ == "__main__":
    # option args
    argparser = argparse.ArgumentParser(
        description="Train and validate YOLO_v2 model on any dataset"
    )
    # predition configration json path
    argparser.add_argument("-c", "--conf", help="path to configuration file")
    # pretrained weight path
    argparser.add_argument("-w", "--weights", help="path to pretrained weights")
    # input image or movie path
    argparser.add_argument(
        "-i", "--input", help="path to an image or an video (mp4 format)"
    )
    # output path
    argparser.add_argument(
        "-o",
        "--output",
        default="./outputs/",
        help="path to output dirctory (mp4 format)",
    )
    # GPU id
    argparser.add_argument("-g", "--PCI_BUS_ID", default="0", help="pci bus id")
    args = argparser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.PCI_BUS_ID

    predict(args)
