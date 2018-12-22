# -*- coding: utf-8 -*-
import tensorflow as tf
import cv2
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt

path = r'/Users/junkchen/Downloads/data/train_data'
filename_label = pd.read_csv(r'/Users/junkchen/Downloads/data/train.csv')
file_names = filename_label['filename']
labels = filename_label['label']
print(file_names[:3])

image = cv2.imread(os.path.join(path, file_names[1]))
plt.imshow(image)

IMAGE_SIZE = 128
# 得到的结果是一个 ndarray
image1 = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
# 得到的结果是一个张量
image1 = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))
