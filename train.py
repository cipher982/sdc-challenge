#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>, David Rose <david010@gmail.com>
# Date:   2018-05-14
#-------------------------------------------------------------------------------

import argparse
import math
import sys
import os
import time

import tensorflow        as tf
import numpy             as np
import matplotlib.pyplot as plt

from fcnvgg import FCNVGG
from utils  import *
from tqdm   import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="1"

#-------------------------------------------------------------------------------
# Parse the commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train the FCN')
parser.add_argument('--name', default='runs/t10',
                    help='project name')
parser.add_argument('--data-source', default='carla2',
                    help='data source')
parser.add_argument('--data-dir', default='D:\\data\\lyft',
                    help='data directory')
parser.add_argument('--vgg-dir', default='vgg_graph',
                    help='directory for the VGG-16 model')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=5,
                    help='batch size')
parser.add_argument('--tensorboard-dir', default="runs/t10_tb",
                    help='name of the tensorboard data directory')
parser.add_argument('--checkpoint-interval', type=int, default=25,
                    help='checkpoint interval')
args = parser.parse_args()

print('[i] Project name:         ', args.name)
print('[i] Data source:          ', args.data_source)
print('[i] Data directory:       ', args.data_dir)
print('[i] VGG directory:        ', args.vgg_dir)
print('[i] # epochs:             ', args.epochs)
print('[i] Batch size:           ', args.batch_size)
print('[i] Tensorboard directory:', args.tensorboard_dir)
print('[i] Checkpoint interval:  ', args.checkpoint_interval)

args.name = args.name
print(args.name)
try:
    print('[i] Creating directory {0}...'.format(args.name))
    os.makedirs(args.name)
except OSError:
    print("DIR exists error!")
    #os.system("rm -rf {0}".format(args.name))
    os.makedirs(args.name)

#-------------------------------------------------------------------------------
# Configure the data source
#-------------------------------------------------------------------------------
print('[i] Configuring data source...')
"""try:
    source = load_data_source(args.data_source)
    source.load_data(args.data_dir, 0.1)
    print('[i] # training samples:   ', source.num_training)
    print('[i] # validation samples: ', source.num_validation)
    print('[i] # classes:            ', source.num_classes)
    print('[i] Image size:           ', source.image_size)
    train_generator = source.train_generator
    valid_generator = source.valid_generator
    label_colors    = source.label_colors
except (ImportError, AttributeError, RuntimeError) as e:
    print('[!] Unable to load data source:', str(e))
    sys.exit(1)
"""
source = load_data_source(args.data_source)
source.load_data(args.data_dir, 0.1)
print('[i] # training samples:   ', source.num_training)
print('[i] # validation samples: ', source.num_validation)
print('[i] # classes:            ', source.num_classes)
print('[i] Image size:           ', source.image_size)
train_generator = source.train_generator
valid_generator = source.valid_generator
label_colors    = source.label_colors

#-------------------------------------------------------------------------------
# Create the network
#-------------------------------------------------------------------------------
with tf.Session() as sess:
    print('[i] Creating the model...')
    net = FCNVGG(sess)
    net.build_from_vgg(args.vgg_dir, source.num_classes, progress_hook='tqdm')

    labels = tf.placeholder(tf.float32,
                            shape=[None, None, None, source.num_classes])

    optimizer, loss = net.get_optimizer(labels)

    summary_writer  = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
    saver           = tf.train.Saver(max_to_keep=10)

    label_mapper    = tf.argmax(labels, axis=3)
    n_train_batches = int(math.ceil(source.num_training/args.batch_size))

    initialize_uninitialized_variables(sess)
    print('[i] Training...')

    #---------------------------------------------------------------------------
    # Summaries
    #---------------------------------------------------------------------------
    validation_loss = tf.placeholder(tf.float32)
    validation_loss_summary_op = tf.summary.scalar('validation_loss',
                                                   validation_loss)

    training_loss = tf.placeholder(tf.float32)
    training_loss_summary_op = tf.summary.scalar('training_loss',
                                                 training_loss)

    validation_img    = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    validation_img_gt = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    validation_img_summary_op = tf.summary.image('validation_img',
                                                 validation_img)
    validation_img_gt_summary_op = tf.summary.image('validation_img_gt',
                                                    validation_img_gt)
    validation_img_summary_ops = [validation_img_summary_op,
                                  validation_img_gt_summary_op]


    for e in range(args.epochs):
        #-----------------------------------------------------------------------
        # Train
        #-----------------------------------------------------------------------
        generator = train_generator(args.batch_size)
        description = '[i] Epoch {:>2}/{}'.format(e+1, args.epochs)
        training_loss_total = 0
        for x, y in tqdm(generator, total=n_train_batches,
                         desc=description, unit='batches'):
            feed = {net.image_input:  x,
                    labels:           y,
                    net.keep_prob:    0.5}
            loss_batch, _ = sess.run([loss, optimizer], feed_dict=feed)
            training_loss_total += loss_batch * x.shape[0]
        training_loss_total /= source.num_training

        #-----------------------------------------------------------------------
        # Validate
        #-----------------------------------------------------------------------
        generator = valid_generator(args.batch_size)
        validation_loss_total = 0
        imgs          = None
        img_labels    = None
        img_labels_gt = None
        for x, y in generator:
            feed = {net.image_input:  x,
                    labels:           y,
                    net.keep_prob:    1}
            loss_batch, img_classes, y_mapped = sess.run([loss,
                                                          net.classes,
                                                          label_mapper],
                                                         feed_dict=feed)
            validation_loss_total += loss_batch * x.shape[0]

            if imgs is None:
                imgs          = x[:3, :, :, :]
                img_labels    = img_classes[:3, :, :]
                img_labels_gt = y_mapped[:3, :, :]

        validation_loss_total /= source.num_validation

        #-----------------------------------------------------------------------
        # Write loss summary
        #-----------------------------------------------------------------------
        print("Training Loss : {0}    Validation Loss :{1}".format(training_loss_total, validation_loss_total))
        feed = {validation_loss: validation_loss_total,
                training_loss:   training_loss_total}
        loss_summary = sess.run([validation_loss_summary_op,
                                 training_loss_summary_op],
                                feed_dict=feed)

        summary_writer.add_summary(loss_summary[0], e)
        summary_writer.add_summary(loss_summary[1], e)

        #-----------------------------------------------------------------------
        # Write image summary every 5 epochs
        #-----------------------------------------------------------------------
        if e % 1 == 0:
            imgs_inferred = draw_labels_batch(imgs, img_labels, label_colors)
            imgs_gt       = draw_labels_batch(imgs, img_labels_gt, label_colors)

            feed = {validation_img:    imgs_inferred,
                    validation_img_gt: imgs_gt}
            validation_img_summaries = sess.run(validation_img_summary_ops,
                                                feed_dict=feed)
            summary_writer.add_summary(validation_img_summaries[0], e)
            summary_writer.add_summary(validation_img_summaries[1], e)

        #-----------------------------------------------------------------------
        # Save a checktpoint
        #-----------------------------------------------------------------------
        if (e+1) % args.checkpoint_interval == 0:
            checkpoint = '{}/e{}.ckpt'.format(args.name, e+1)
            saver.save(sess, checkpoint)
            print('Checkpoint saved:', checkpoint)

    checkpoint = '{}/final.ckpt'.format(args.name)
    saver.save(sess, checkpoint)
    print('Checkpoint saved:', checkpoint)