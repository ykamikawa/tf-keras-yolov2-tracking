# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
from preprocessing import parse_annotation
from frontend import YOLO
import json


def main(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                                config['train']['train_image_folder'],
                                                config['model']['labels'])

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'],
                                                    config['valid']['valid_image_folder'],
                                                    config['model']['labels'])
    else:
        train_valid_split = int(0.8*len(train_imgs))
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    # detected labels
    overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

    print ('Seen labels:\t', train_labels)
    print ('Given labels:\t', config['model']['labels'])
    print ('Overlap labels:\t', overlap_labels)

    if len(overlap_labels) < len(config['model']['labels']):
        print ('Some labels have no images! Please revise the list of labels in the config.json file!')
        return

    # construct models
    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'],
                labels              = config['model']['labels'],
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    # load pretrained models
    if os.path.exists(config['train']['pretrained_weights']):
        print ("Loading pre-trained weights in", config['train']['pretrained_weights'])
        yolo.load_weights(config['train']['pretrained_weights'])

    # start trianing
    yolo.train(train_imgs         = train_imgs,
               valid_imgs         = valid_imgs,
               train_times        = config['train']['train_times'],
               valid_times        = config['valid']['valid_times'],
               nb_epoch           = config['train']['nb_epoch'],
               learning_rate      = config['train']['learning_rate'],
               batch_size         = config['train']['batch_size'],
               warmup_bs          = config['train']['warmup_batches'],
               object_scale       = config['train']['object_scale'],
               no_object_scale    = config['train']['no_object_scale'],
               coord_scale        = config['train']['coord_scale'],
               class_scale        = config['train']['class_scale'],
               saved_weights_name = config['train']['saved_weights_name'] + "_" + config['model']['architecture'],
               debug              = config['train']['debug'])

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Train and validate YOLO_v2 model on any dataset')
    argparser.add_argument(
        '-c',
        '--conf',
        type=str,
        help='path to configuration file')
    argparser.add_argument(
        '-g',
        '--gpu_num',
        default="1",
        type=str,
        help='gpu num')
    args = argparser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    main(args)
