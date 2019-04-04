#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:06:57 2019

@author: xavier.qiu
"""

from __future__ import print_function
import glob
import os
import numpy as np
import cv2

root_path = './data/mini-imagenet/data'
train_path = os.path.join(root_path, 'train')
test_path = os.path.join(root_path, 'test')

train_dirs = [f for f in glob.glob(os.path.join(train_path, '*')) if os.path.isdir(f)]
test_dirs = [f for f in glob.glob(os.path.join(test_path, '*')) if os.path.isdir(f)]

# %%

# randomly pick a class from all the class
selected_class = np.random.choice(train_dirs, 1)[0]
sample_images = glob.glob(selected_class + "/*.JPEG")
sample_images = np.random.choice(sample_images, 5)
sample_images

import cv2

sample_images_array = []
for f in sample_images:
    sample_images_array.append(cv2.imread(f))

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

plt.figure(figsize=(16, 16))
plt.subplot(221)

plt.imshow(sample_images_array[0])
plt.grid()
plt.title("")
plt.subplot(222)

plt.imshow(sample_images_array[1])
plt.grid()
plt.title("")

plt.subplot(223)

plt.imshow(sample_images_array[2])
plt.grid()
plt.title("")

plt.subplot(224)

plt.imshow(sample_images_array[3])
plt.grid()
plt.title("")

# %%
