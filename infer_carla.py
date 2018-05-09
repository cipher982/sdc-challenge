import argparse
import math
import sys
import cv2
import os

import tensorflow        as tf
import numpy             as np
import matplotlib.pyplot as plt

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
    for offset in range(0, len(video), batch_size):
        files = video[offset:offset+batch_size]
        images = []
        for frame in video:
            image = cv2.resize(frame, image_size)
            images.append(image.astype(np.float32))
            #names.append(os.path.basename(image_file))
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
parser.add_argument('--data-source', default='carla',
                    help='data source')

args = parser
args.name        = 'runs\\t3'
args.checkpoint  = -1
args.video_file  = 'test_video.mp4'
args.output_dir  = 'test_output'
args.batch_size  = 31
args.data_source = 'carla'


state = tf.train.get_checkpoint_state(args.name)
if state is None:
    print('[!] No network state found in ' + args.name)
    sys.exit(1)

try:
    checkpoint_file = state.all_model_checkpoint_paths[args.checkpoint]
    print("Loaded checkpoint: {0}".format(checkpoint_file))
except IndexError:
    print('[!] Cannot find checkpoint ' + str(args.checkpoint_file))
    sys.exit(1)

metagraph_file = checkpoint_file + '.meta'

if not os.path.exists(metagraph_file):
    print('[!] Cannot find metagraph ' + metagraph_file)
    sys.exit(1)
else:
    print('Loaded metagraph:  {0}'.format(metagraph_file))


try:
    source       = load_data_source(args.data_source)
    label_colors = source.label_colors
except (ImportError, AttributeError, RuntimeError) as e:
    print('[!] Unable to load data source:', str(e))
    sys.exit(1)

video = skvideo.io.vread(args.video_file)

with tf.Session() as sess:
    print('[i] Creating the model...')
    net = FCNVGG(sess)
    net.build_from_metagraph(metagraph_file, checkpoint_file)

    #---------------------------------------------------------------------------
    # Process the images
    #---------------------------------------------------------------------------
    generator = sample_generator(video, source.image_size, args.batch_size)
    #generator = sample_generator(video, image_size=args.image_size)

    n_sample_batches = int(math.ceil(len(video)/args.batch_size))
    description = '[i] Processing video'

    for x in tqdm(generator, total=n_sample_batches, desc=description, unit='batches'):
        feed = {net.image_input:  x,
                net.keep_prob:    1}
        img_labels = sess.run(net.classes, feed_dict=feed)

        #binary_car_result = np.where()
        imgs = draw_labels_batch(x, img_labels, label_colors, False)
