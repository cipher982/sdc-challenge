#-------------------------------------------------------------------------------
# Author: David Rose <David010@gmail.com>
# Date:   2018-05-01
#-------------------------------------------------------------------------------

import random
import cv2
import os
import time

import numpy as np

from collections import namedtuple
from glob import glob

#-------------------------------------------------------------------------------
# Labels
#-------------------------------------------------------------------------------
Label = namedtuple('Label', ['name', 'color'])

def rgb2bgr(tpl):
    return (tpl[2], tpl[1], tpl[0])

label_defs = [
    Label('None'          ,(0, 0,  0)),
    Label('Buildings'     ,(0, 0,  1)),
    Label('Fences'        ,(0, 0,  2)),
    Label('Other'         ,(0, 0,  3)),
    Label('Pedestrians'   ,(0, 0,  4)),
    Label('Poles'         ,(0, 0,  5)),
    Label('RoadLines'     ,(0, 0,  6)),
    Label('Roads'         ,(0, 0,  7)),
    Label('Sidewalks'     ,(0, 0,  8)),
    Label('Vegetation'    ,(0, 0,  9)),
    Label('Vehicles'      ,(0, 0, 10)),
    Label('Walls'         ,(0, 0, 11)),
    Label('TrafficSigns'  ,(0, 0, 12))
    ]

label_defs = [
    Label('None'          ,(0, 0,  0)),
    Label('Buildings'     ,(0, 0,  1)),
    Label('Fences'        ,(0, 0,  2)),
    Label('Other'         ,(0, 0,  3)),
    Label('Pedestrians'   ,(0, 0,  4)),
    Label('Poles'         ,(0, 0,  5)),
    Label('RoadLines'     ,(0, 0,  6)),
    Label('Roads'         ,(0, 0,  7)),
    Label('Sidewalks'     ,(0, 0,  8)),
    Label('Vegetation'    ,(0, 0,  9)),
    Label('Vehicles'      ,(0, 0, 10)),
    Label('Walls'         ,(0, 0, 11)),
    Label('TrafficSigns'  ,(0, 0, 12))
    ]

#-------------------------------------------------------------------------------
def build_file_list(images_root, labels_root):
    image_root_len    = len(images_root)
    image_files       = images_root
    file_list         = []
    for f in image_files:
        whichImage = f.rsplit('\\', 1)[-1]
        f_label    = labels_root + '\\' + whichImage
        file_list.append((f, f_label))
    return file_list

#-------------------------------------------------------------------------------
class CarlaSource:
    #---------------------------------------------------------------------------
    def __init__(self):
        self.image_size      = (416, 320)
        self.num_classes     = len(label_defs)

        self.label_colors    = {i: np.array(l.color) for i, l \
                                                     in enumerate(label_defs)}

        self.num_training    = None
        self.num_validation  = None
        self.train_generator = None
        self.valid_generator = None

    #---------------------------------------------------------------------------
    def load_data(self, data_dir, valid_fraction):
        """
        Load the data and make the generators
        :param data_dir:       the directory where the dataset's file are stored
        :param valid_fraction: what franction of the dataset should be used
                               as a validation sample
        """
        images = data_dir + '\\Train\\CameraSeg\\*.png'
        labels = data_dir + '\\Train\\CameraRGB'
        #print(images)

        image_paths = glob(images)
        label_paths = labels
        #self.label_paths = label_paths

        image_paths = build_file_list(image_paths, label_paths)

        num_images = len(image_paths)
        if num_images == 0:
            raise RuntimeError('No data files found in ' + data_dir)

        random.shuffle(image_paths)
        valid_images = image_paths[:int(valid_fraction*num_images)]
        train_images = image_paths[int(valid_fraction*num_images):]

        self.num_classes     = 13
        self.num_training    = len(train_images)
        self.num_validation  = len(valid_images)
        self.train_generator = self.batch_generator(train_images)
        self.valid_generator = self.batch_generator(valid_images)       

    #---------------------------------------------------------------------------
    def batch_generator(self, image_paths):
        def gen_batch(batch_size, names=False):
            random.shuffle(image_paths)
            for offset in range(0, len(image_paths), batch_size):
                files = image_paths[offset:offset+batch_size]

                images = []
                labels = []
                names_images = []
                names_labels = []
                for i, f in enumerate(files):
                    #print("f:{0}  shape:{1}".format(f,np.shape(f)))
                    #print("f[0]:{0}  f[1]:{1}".format(f[0],f[1]))
                    image_file = f[0]
                    label_file = f[1]

                    #print("loading image. . .")
                    image = cv2.imread(image_file)
                    label = cv2.imread(label_file)

                    #print("Resizing image. . .")
                    image = cv2.resize(image, self.image_size)
                    label = cv2.resize(label, self.image_size)

                    label_bg   = np.zeros([image.shape[0], image.shape[1]], dtype=bool)
                    label_list = []
                    for ldef in label_defs[1:]:
                        label_current  = np.all(label == ldef.color, axis=2)
                        label_bg      |= label_current
                        label_list.append(label_current)
                        if ldef.name == 'Vehicles':
                            print("======================== {0}".format(f))
                            print("label_current:{0}".format(sum(label_current)))
                            print("ldef:{0}  color:{1}".format(ldef,ldef.color))

                    label_bg   = ~label_bg
                    label_all  = np.dstack([label_bg, *label_list])
                    label_all  = label_all.astype(np.float32)

                    images.append(image.astype(np.float32))
                    labels.append(label_all)

                    if names:
                        names_images.append(image_file)
                        names_labels.append(label_file)

                if names:
                    yield np.array(images), np.array(labels), \
                          names_images, names_labels
                else:
                    yield np.array(images), np.array(labels)
        return gen_batch

#-------------------------------------------------------------------------------
def get_source():
    return CarlaSource()