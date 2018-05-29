#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   14.06.2017
#-------------------------------------------------------------------------------

import cv2

import tensorflow as tf
import numpy as np

#-------------------------------------------------------------------------------
def draw_labels(img, labels, label_colors, convert=False):
    """
    Draw the labels on top of the input image
    :param img:          the image being classified
    :param labels:       the output of the neural network
    :param label_colors: the label color map defined in the source
    :param convert:      should the output be converted to RGB
    """
    labels_colored = np.zeros_like(img)
    for label in label_colors:
        #label_mask = labels == label
        #print("\n\nlabel:{0}\n\n".format(label))
        #print("avg:{0}".format(np.max(labels_colored)))
        #print(labels_colored)

        if label   ==  7: # Roads
            road_mask = labels == label
        elif label == 10: # Vehicles
            vehicle_mask = labels == label

    #img = cv2.addWeighted(img, 1, labels_colored, 0.8, 0)
    img = cv2.addWeighted(img, 1, labels_colored, 1, 0)

    if not convert:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def draw_binary_label(tf_img, label_colors, resize=True):
    """
    Convert the TF output to labeled and encoded JSON data for Lyft
    :param tf_img: the image being classified
    :param label_colors: RGB values of labels (source specific)
    """
    for label in label_colors:
        #print("color:{0}".format(color))
        if label   ==  7: # Roads
            road_mask = tf_img == label
        elif label == 10: # Vehicles
            vehicle_mask = tf_img == label



    return road_mask, vehicle_mask
    
#-------------------------------------------------------------------------------
def draw_labels_batch(imgs, labels, label_colors, convert=True):
    """
    Perform `draw_labels` on all the images in the batch
    """
    imgs_labeled = np.zeros_like(imgs)
    for i in range(imgs.shape[0]):
        imgs_labeled[i, :, :, :] = draw_labels(imgs[i,:, :, :],
                                               labels[i, :, :],
                                               label_colors,
                                               convert)
    return imgs_labeled

#-------------------------------------------------------------------------------
def initialize_uninitialized_variables(sess):
    """
    Only initialize the weights that have not yet been initialized by other
    means, such as importing a metagraph and a checkpoint. It's useful when
    extending an existing model.
    """
    uninit_vars    = []
    uninit_tensors = []
    for var in tf.global_variables():
        uninit_vars.append(var)
        uninit_tensors.append(tf.is_variable_initialized(var))
    uninit_bools = sess.run(uninit_tensors)
    uninit = zip(uninit_bools, uninit_vars)
    uninit = [var for init, var in uninit if not init]
    sess.run(tf.variables_initializer(uninit))

#-------------------------------------------------------------------------------
def load_data_source(data_source):
    """
    Load a data source given it's name
    """
    source_module = __import__('source_'+data_source)
    get_source    = getattr(source_module, 'get_source')
    return get_source()
