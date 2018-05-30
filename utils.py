#-------------------------------------------------------------------------------
# Author: David Rose <david010@gmail.com>
# Date:   2018.05.29
#-------------------------------------------------------------------------------

import cv2

import tensorflow as tf
import numpy      as np

#-------------------------------------------------------------------------------

def preprocess_labels(label_image):
    # Create a new single channel label image to modify
    labels_new = np.copy(label_image[:,:,0])
    # Identify lane marking pixels (label is 6)
    lane_marking_pixels = (label_image[:,:,0] == 6).nonzero()
    # Set lane marking pixels to road (label is 7)
    labels_new[lane_marking_pixels] = 7
    # Identify all vehicle pixels
    vehicle_pixels = (label_image[:,:,0] == 10).nonzero()
    # Isolate vehicle pixels associated with the hood (y-position > 496)
    hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
    hood_pixels = (vehicle_pixels[0][hood_indices], \
                   vehicle_pixels[1][hood_indices])
    # Set hood pixel labels to 0
    labels_new[hood_pixels] = 0
    # Return the preprocessed label image 
    return labels_new

def draw_binary_label(base_img, tf_img, label_colors, resize=True):
    """
    Convert the TF output to labeled and encoded JSON data for Lyft
    :param tf_img: the image being classified
    :param label_colors: RGB values of labels (source specific)
    """
    for label in label_colors:
        print("label:{0}".format(label))
        if label   ==  7: # Roads
            print("label:7, roads. . .")
            road_mask = tf_img == label
        elif label == 10: # Vehicles
            print("label:10, vehicles. . .")
            vehicle_mask = tf_img == label
            print(vehicle_mask)

    colored = cv2.addWeighted(img, 1, labels_colored, 1, 0)

    return road_mask, vehicle_mask, colored
    
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
