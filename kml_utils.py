# -*- coding: utf-8 -*-
from __future__ import print_function

import random
import os
import cv2
import numpy as np

IMG_SIZE = 128

# checking existence of the class, train and test lists
def exist_list(list_dir):
    exists = os.path.exists(os.path.join('.', list_dir, 'class.lst')) \
             and os.path.exists(os.path.join('.', list_dir, 'train.lst')) \
             and os.path.exists(os.path.join('.', list_dir, 'test.lst'))
    return exists

# create and return the class, train and test lists
def create_list(data_dir, list_dir, slash):
    classes = os.listdir(os.path.join('.', data_dir))
    data_list = []
    for i, cls in enumerate(classes):
        files = os.listdir(os.path.join('.', data_dir, cls))
        for f in files:
            data_list.append(os.path.join('.', data_dir, cls, f))

    split_index = int(len(data_list) * slash)
    random.shuffle(data_list)
    train_list = data_list[split_index:]
    test_list = data_list[:split_index]
    try:
        os.mkdir(list_dir)
    except OSError:
        print('Directory ./{0} already exists.'.format(list_dir))
    f = open(os.path.join('.', list_dir, 'class.lst'), 'w')
    f.write('\n'.join(classes))
    f.close()
    f = open(os.path.join('.', list_dir, 'train.lst'), 'w')
    f.write('\n'.join(train_list))
    f.close()
    f = open(os.path.join('.', list_dir, 'test.lst'), 'w')
    f.write('\n'.join(test_list))
    f.close()
    return classes, train_list, test_list

# load the class, train and test lists
def load_lists(list_dir):
    f = open(os.path.join('.', list_dir, 'class.lst'), 'r')
    classes = f.read().split()
    f.close()
    f = open(os.path.join('.', list_dir, 'train.lst'), 'r')
    train_list = f.read().split()
    f.close()
    f = open(os.path.join('.', list_dir, 'test.lst'), 'r')
    test_list = f.read().split()
    f.close()
    return classes, train_list, test_list

# load images and add labels
def load_images(classes, data_list):
    images = []
    labels = []
    num_classes = len(classes)
    for data in data_list:
        img = cv2.imread(data)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img/255.0
        images.append(img)
        lbl = np.zeros(num_classes)
        lbl[classes.index(os.path.basename(os.path.dirname(data)))] = 1
        labels.append(lbl)
    return images, labels

