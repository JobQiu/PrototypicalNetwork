from __future__ import print_function
import glob
import os
import numpy as np
import cv2

n_train_classes = 64
n_test_classes = 20
n_val_classes = 16
n_examples, width, height, channels = 350, 84, 84, 3

root_path = './data/mini-imagenet/data'
train_path = os.path.join(root_path, 'train')
val_path = os.path.join(root_path, 'val')
test_path = os.path.join(root_path, 'test')

train_dirs = [f for f in glob.glob(os.path.join(train_path, '*')) if os.path.isdir(f)]
test_dirs = [f for f in glob.glob(os.path.join(test_path, '*')) if os.path.isdir(f)]
val_dirs = [f for f in glob.glob(os.path.join(val_path, '*')) if os.path.isdir(f)]

assert len(train_dirs) == n_train_classes
assert len(test_dirs) == n_test_classes

read_and_resize = lambda x: cv2.resize(cv2.imread(x, 1), (width, height))


def sample_dataset(dataset, dirs, name='train'):
    for i, d in enumerate(dirs):
        fs = np.asarray(glob.glob(os.path.join(d, '*.JPEG')))
        fs = fs[np.random.permutation(len(fs))][:n_examples]
        for j, f in enumerate(fs):
            dataset[i, j] = read_and_resize(f)
        print('{}: {} of {}'.format(name, i + 1, len(dirs)))
    return dataset


val_dataset = np.zeros((n_val_classes, n_examples, width, height, channels), dtype=np.uint8)
val_dataset = sample_dataset(val_dataset, val_dirs, name='val')
val_dataset_0 = val_dataset[:n_val_classes // 2]
val_dataset_1 = val_dataset[n_val_classes // 2:]

np.save('mini-imagenet-val_0.npy', val_dataset_0)
np.save('mini-imagenet-val_1.npy', val_dataset_0)

"""

test_dataset = np.zeros((n_test_classes, n_examples, width, height, channels), dtype=np.uint8)
test_dataset = sample_dataset(test_dataset, test_dirs, name='test')
test_dataset_0 = test_dataset[:n_test_classes // 2]
test_dataset_1 = test_dataset[n_test_classes // 2:]

np.save('mini-imagenet-test_0.npy', test_dataset_0)
np.save('mini-imagenet-test_1.npy', test_dataset_1)

del test_dataset

train_dataset = np.zeros((n_train_classes, n_examples, width, height, channels), dtype=np.uint8)
train_dataset = sample_dataset(train_dataset, train_dirs)

for i in range(8):
    temp = train_dataset[i * 8:(i + 1) * 8]
    np.save('mini-imagenet-train_{}.npy'.format(i), temp)
del train_dataset
"""
