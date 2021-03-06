#!/usr/bin/env python3

# Based on https://github.com/metno/cc-classifier/blob/master/dataset.py
# 


import re
import cv2
import numpy as np
import os
from sklearn.utils import shuffle
import tensorflow as tf
import sys
import math

import augment

class DataSet(object):

    def __init__(self, images, labels):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels        
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return shuffle(self._images[start:end], self._labels[start:end])


def load_train_sets_from_dirs(datadir, classes, image_size, validation_size):
    import pathlib

    images = []
    labels = []
   
    label_counts = {0:0, 1:0}

    for cl in classes : 
        for p in pathlib.Path(datadir + '/' + str(classes[cl])).iterdir():
            if not p.is_file():
                continue
            path_s = str(p)
            if path_s.endswith(".jpg") is False and path_s.endswith(".jpeg") is False:
                continue

            print(path_s)
            try:
                image = cv2.imread(path_s)
            except cv2.error as e:
                print(e)
                continue

            if image is None:
                print("image %s is none" % path_s)
                continue
            # Already resized
            # image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
        

            label_counts[int(cl)] = label_counts[int(cl)] + 1
            index = classes.index(int(cl))
            image = image.astype(np.float32)
            # convert from [0:255] => [0.0:1.0]
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
   
    images = np.array(images)
    labels = np.array(labels)

    print(label_counts)
    return images, labels, label_counts    
    

def read_train_sets(imagedir, image_size, classes, validation_size):
    class DataSets(object):
        pass
    data_sets = DataSets()


    #images, labels, label_counts = load_training_data(labelsfile, imagedir, image_size, classes)
    images, labels, label_counts = load_train_sets_from_dirs(imagedir, classes, image_size, validation_size=0.20)
    print("SIZE: %d" % (sys.getsizeof(images) / (1024*1024)))
    
    do_aug = False
    if do_aug:
        print("Augmenting data ..")
        aug_images, aug_labels = augment.augment_data2(images, labels, label_counts)
	
        images = np.concatenate([images, aug_images])
        labels = np.concatenate([labels, aug_labels])


    images, labels = shuffle(images, labels)
    
    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]



    train_images = images[validation_size:]
    train_labels = labels[validation_size:]

    print("train_images1: %d, train_labels1: %d" %(len(train_images), len(train_labels)))

    
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.valid = DataSet(validation_images, validation_labels)

    return data_sets

# Test
if __name__ == "__main__":
    classes = [0, 1]
    load_train_sets_from_dirs('/home/espenm/data/scenes/train', classes, 128, validation_size=0.20)
