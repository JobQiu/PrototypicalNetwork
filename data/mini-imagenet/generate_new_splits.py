import os


def calculate_non_empty_dir_num(type='train'):
    dirs = next(os.walk('data/' + type))[1]

    empty_dirs = [x for x in dirs if not os.listdir('data/' + type + "/" + x)]
    empty_dirs = set(empty_dirs)

    # print(empty_dirs)
    # print(len(empty_dirs) / len(dirs))

    # print(len(dirs) - len(empty_dirs))
    return len(dirs) - len(empty_dirs)


train_size = calculate_non_empty_dir_num()

test_size = calculate_non_empty_dir_num('test')

print("there are {} train classes".format(train_size))
print("there are {} test classes".format(test_size))

# %%
type = 'train'

dirs = next(os.walk('data/' + type))[1]

empty_dirs = [x for x in dirs if not os.listdir('data/' + type + "/" + x)]
empty_dirs = set(empty_dirs)

non_empty_train_dirs = [x for x in dirs if x not in empty_dirs]

type = 'test'

dirs = next(os.walk('data/' + type))[1]

empty_dirs = [x for x in dirs if not os.listdir('data/' + type + "/" + x)]
empty_dirs = set(empty_dirs)

non_empty_test_dirs = [x for x in dirs if x not in empty_dirs]

type = 'val'

dirs = next(os.walk('data/' + type))[1]

empty_dirs = [x for x in dirs if not os.listdir('data/' + type + "/" + x)]
empty_dirs = set(empty_dirs)

non_empty_val_dirs = [x for x in dirs if x not in empty_dirs]

valid_train_str = ""
for e in non_empty_train_dirs:
    valid_train_str += "\"" + e + "\"" + " "
print(valid_train_str)

valid_test_str = ""
for e in non_empty_test_dirs:
    valid_test_str += "\"" + e + "\"" + " "
print(valid_test_str)

valid_val_str = ""
for e in non_empty_val_dirs:
    valid_val_str += "\"" + e + "\"" + " "
print(valid_val_str)

# %%

import pandas as pd

train_split = pd.read_csv("splits/train.csv")
test_split = pd.read_csv("splits/test.csv")
val_split = pd.read_csv("splits/val.csv")

from collections import Counter

c1 = Counter(train_split['label'])  # 64
c2 = Counter(test_split['label'])
c3 = Counter(val_split['label'])

c = set(list(c1.keys()) + list(c2.keys()) + list(c3.keys()))

# %%

import numpy as np

np.random.seed(42)

os.remove('valid_splits/train.csv')

with open("valid_splits/train.csv", "a") as f:
    f.write("filename,label\n")
    for _class in non_empty_train_dirs:
        images = next(os.walk('data/train/' + _class))[2]
        filenames = np.random.choice(images, 600)
        for file in filenames:
            f.write("{},{}\n".format(file, _class))

# %%

os.remove('valid_splits/test.csv')
with open("valid_splits/test.csv", "a") as f:
    f.write("filename,label\n")
    for _class in non_empty_test_dirs:
        images = next(os.walk('data/test/' + _class))[2]
        filenames = np.random.choice(images, 600)
        for file in filenames:
            f.write("{},{}\n".format(file, _class))

os.remove('valid_splits/val.csv')
with open("valid_splits/val.csv", "a") as f:
    f.write("filename,label\n")
    for _class in non_empty_val_dirs:
        images = next(os.walk('data/val/' + _class))[2]
        filenames = np.random.choice(images, 600)
        for file in filenames:
            f.write("{},{}\n".format(file, _class))
# %%
