#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:31:38 2019

@author: xavier.qiu
"""

from __future__ import print_function
import glob
import os
import numpy as np
import cv2
from tqdm import tqdm

n_train_classes = 64
n_test_classes = 20
n_val_classes = 16
n_examples, width, height, channels = 350, 100, 100, 3

root_path = './data/mini-imagenet/data'
train_path = os.path.join(root_path, 'train')
val_path = os.path.join(root_path, 'val')
test_path = os.path.join(root_path, 'test')

train_dirs = [f for f in glob.glob(os.path.join(train_path, '*')) if os.path.isdir(f)]
test_dirs = [f for f in glob.glob(os.path.join(test_path, '*')) if os.path.isdir(f)]

# %%

read_and_resize = lambda x: cv2.resize(cv2.imread(x, 1), (width, height))
from collections import defaultdict


def sample_dataset(name='train', m=defaultdict(list)):
    data_path = os.path.join(root_path, name)
    dirs = [f for f in glob.glob(os.path.join(data_path, '*')) if os.path.isdir(f)]
    n_classes = len(dirs)
    dataset = np.zeros((n_classes, n_examples, width, height, channels), dtype=np.uint8)
    for i, d in tqdm(enumerate(dirs)):

        fs = glob.glob(os.path.join(d, '*.JPEG'))
        fs = sorted(fs, key=os.path.getsize)
        fs = np.asarray(fs)

        fs = fs[-n_examples:]
        for j, f in enumerate(fs):
            temp_image = cv2.imread(f, 1)
            # print(temp_image.shape)
            m[d].append(temp_image.shape[:2])
            dataset[i, j] = read_and_resize(f)
    return dataset, m, d


for n in ['train', 'val', 'test']:

    dd = sample_dataset(name=n)

    from collections import Counter
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()
    xx = 1000
    yy = 1000
    i = 0
    for k in dd[1].keys():
        l = dd[1][k]
        ll = np.array(l)
        x, y = np.amin(ll, axis=0)
        xx = min(x, xx)
        yy = min(y, yy)
    print(xx)
    print(yy)
