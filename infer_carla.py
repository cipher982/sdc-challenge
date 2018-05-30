#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: David Rose <david010@gmail.com>
# Date:   2018-05-14
#-------------------------------------------------------------------------------


import argparse
import random
import time
import math
import sys
import cv2
import os

import tensorflow        as tf
import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd

from fcnvgg import FCNVGG
from utils  import *
from glob   import glob
from tqdm   import tqdm

# Imports from demo.py in Udacity Workspace
import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io  import BytesIO, StringIO

def sample_generator(video, image_size, batch_size):
    #print("Starting sample_generator. . .")
    for offset in range(0, len(video), batch_size):
        #rint("\n\n\n\n\n\n\noffset:{0}".format(offset))
        #print("len(video):{0}".format(len(video)))
        #print("batch_size:{0}".format(batch_size))
        files = video[offset:offset+batch_size]
        images = []
        for frame in files:
            image = cv2.resize(frame, image_size)
            images.append(image.astype(np.float32))
            #names.append(os.path.basename(image_file))
        #print("images shape:{0}".format(np.shape(images)))
        yield np.array(images)

def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")

parser = argparse.ArgumentParser(description='Generate data based on a model')
parser.add_argument('--name', default='test',
                    help='project name')
parser.add_argument('--checkpoint', type=int, default=-1,
                    help='checkpoint to restore; -1 is the most recent')
parser.add_argument('--video-file', default='test',
                    help='video file to analyze')
parser.add_argument('--output-dir', default='test-output',
                    help='directory for the resulting images')
parser.add_argument('--batch-size', type=int, default=10,
                    help='batch size')
parser.add_argument('--data-source', default='carla2',
                    help='data source')

args = parser
#args.name        = '/home/runs/t5'
args.name        = 'runs/t5'
args.checkpoint  = -1
args.video_file  = 'test_video.mp4'
args.output_dir  = 'test_output'
args.batch_size  = 20
args.data_source = 'carla2'

state = tf.train.get_checkpoint_state(args.name)
if state is None:
    print('[!] No network state found in ' + args.name)
    sys.exit(1)

try:
    checkpoint_file = state.all_model_checkpoint_paths[args.checkpoint]
    #print("Loaded checkpoint: {0}".format(checkpoint_file))
except IndexError:
    print('[!] Cannot find checkpoint ' + str(args.checkpoint_file))
    sys.exit(1)

metagraph_file = checkpoint_file + '.meta'

if not os.path.exists(metagraph_file):
    print('[!] Cannot find metagraph ' + metagraph_file)
    sys.exit(1)
else:
    #print('Loaded metagraph:  {0}'.format(metagraph_file))
    2+2


try:
    source       = load_data_source(args.data_source)
    label_colors = source.label_colors
except (ImportError, AttributeError, RuntimeError) as e:
    print('[!] Unable to load data source:', str(e))
    sys.exit(1)

#print(0)
file = sys.argv[-1]
video = skvideo.io.vread(file)
#video = np.load("woo2.npy")
#print("\n\n\nSize of video is:{0}\n\n\n".format(np.shape(video)))

global imgs
#print(1)
with tf.Session() as sess:
    #print('[i] Creating the model...')
    net = FCNVGG(sess)
    net.build_from_metagraph(metagraph_file, checkpoint_file)

    #---------------------------------------------------------------------------
    # Process the images
    #---------------------------------------------------------------------------
    generator = sample_generator(video, source.image_size, args.batch_size)


    n_sample_batches = int(math.ceil(len(video)/args.batch_size))
    #description = '[i] Processing video'
    frames = []
    #for x in tqdm(generator, total=n_sample_batches, desc=description, unit='batches'):
    for x in generator:
        #print("\nSize of x is:{0}".format(np.shape(x)))
        feed = {net.image_input:  x,
                net.keep_prob:    1}
        img_labels = sess.run(net.classes, feed_dict=feed)

        #print("\ntype(x):{0} np.shape(x):{1}\n\n\n".format(type(img_labels), np.shape(img_labels)))
        for i in range(len(img_labels)):
            #print("\n\nimg_labeled:{0}".format(np.shape(img_labels[i])))
            print(img_labels[i])
            #print("\n\n")
            labeled_resized = cv2.resize(img_labels[i], (800, 600), interpolation=cv2.INTER_NEAREST)
            print("---------")
            print(labeled_resized)
            road_mask, vehicle_mask = draw_binary_label(labeled_resized, label_colors)
            #print("road_mask:{0}".format(np.shape(road_mask)))
            frames.append([road_mask, vehicle_mask])

answer_key = {}
frame_ix = 1
for frame in frames:
	# Start frames at 1
	road_mask    = frame[0].astype('uint8')
	vehicle_mask = frame[1].astype('uint8')
	#print("\nroad_maskType:{0}, shape:{1}".format(type(road_mask), np.shape(road_mask)))
	answer_key[frame_ix] = [encode(road_mask), encode(vehicle_mask)]
	frame_ix += 1

#print(json.dumps(answer_key))



