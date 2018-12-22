# -*- coding: utf-8 -*-
# 1、导包
import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
import keras
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from keras.utils import to_categorical
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 2、
IMAGE_SIZE = 128
path = r'E:\PrivateDocuments\object_recognition_data\train_data'
filename_label = pd.read_csv(r'E:\PrivateDocuments\object_recognition_data\train.csv')
filesname = filename_label['filename']
labels = filename_label['label']


def load_image(path, filesname, height, width, channels):
    images = []
    for image_name in filesname:
        image = cv2.imread(os.path.join(path, image_name))
        image = cv2.resize(image, (height, width))
        images.append(image / 255.0)
    images = np.array(images)
    images = images.reshape([-1, height, width, channels])
    return images


# 会很慢，不建议这样操作
# dataset = load_image(path, filesname, IMAGE_SIZE, IMAGE_SIZE, 3)


def make_train_and_val_set(dataset, labels, test_size):
    train_set, val_set, train_label, val_label = train_test_split(dataset,
                                                                  labels,
                                                                  test_size=test_size,
                                                                  random_state=5)
    return train_set, val_set, train_label, val_label


train_set_name, val_set_name, train_label, val_label = make_train_and_val_set(filesname, labels, 0.2)

BATCH_SIZE = 64


def data_generator(image_path, filesname, labels, batch_size):
    # batch_size
    batchs = (len(labels) + batch_size - 1) // batch_size
    X = []
    Y = []

    while(True):
        for i in range(batchs):
            y = labels[i * batch_size:(i + 1) * batch_size]
            y0, y1 = label2vec(y)
            x_names = filesname[i * batch_size:(i + 1) * batch_size]
            x = load_image(image_path, x_names, IMAGE_SIZE, IMAGE_SIZE, 3)

            X.append(x)
            Y.append(y1)

            X = np.array(X)
            Y = np.array(Y)

            X = X.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 3])
            Y = Y.reshape([-1, 102])

            yield(X, Y)
            X = []
            Y = []


#
encoder = LabelEncoder()
encoder.fit(labels)


def label2vec(labels):
    # 数字的label[1, 2, 3, 4]
    labels1 = encoder.transform(labels)
    one_hot_labels1 = to_categorical(labels1, num_classes=102)
    return labels1, one_hot_labels1


train_label0, train_label1 = label2vec(train_label)
val_label0, val_label1 = label2vec(val_label)


def build_model():
    model = keras.Sequential()
    model.add(Conv2D(32, kernel_size=3, padding='same', activation=tf.nn.relu, input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3],
                     data_format='channels_last',))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, padding='same', activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=3, padding='same', activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(256, kernel_size=3, padding='same', activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1024, activation=tf.nn.relu))
    model.add(Dense(1024, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(102, activation=tf.nn.softmax))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    model.summary()
    return model


model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# model.fit(x=train_set, y=train_label, batch_size=100, epochs=20, verbose=1, validation_split=0.1)
val_set = load_image(path, val_set_name, IMAGE_SIZE, IMAGE_SIZE, 3)
history = model.fit_generator(data_generator(path, train_set_name, train_label, BATCH_SIZE),
                              (len(train_label) + BATCH_SIZE - 1)//BATCH_SIZE,
                              epochs=16, validation_data=(val_set, val_label1),
                              callbacks=[early_stop])

history.history.keys()
model.save('demo/model.h5')
model.save('demo/model_adam.h5')
model.save('demo/model_adadelta.h5')
# model = keras.models.load_model('demo/model.h5')


def acc_topk(y_true, y_pred, k):
    """Get top 5"""
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k)


preds = model.predict(val_set)
with tf.Session() as sess:
    print(sess.run(acc_topk(val_label1, preds, k=5)))


# 测试集
test_path = r'E:\PrivateDocuments\object_recognition_data\test_data'


def file_names(path):
    names = []
    for file in os.listdir(path):
        if not file.startswith("._"):
            names.append(file)
    return names


test_set_names = file_names(test_path)
test_set = load_image(test_path, file_names(test_path), IMAGE_SIZE, IMAGE_SIZE, 3)

test_predicts = model.predict(test_set)
type(test_predicts)
len(test_predicts[0])
test_predicts.shape
print(test_predicts[0][0], test_predicts[0][1])

print(val_label0[0], val_label1[0])
print(test_predicts[0])


def vec2label(label_vec):
    """
    one hot coding 转回 字符串 label
    :param label_vec:
    :return:
    """
    label = encoder.inverse_transform(label_vec)
    return label


sess = tf.Session()


# def get_top_k_label(preds, k=1):
#     """
#
#     :param preds:
#     :param k:
#     :return:
#     """
#     startTime = time.clock()
#     top_k = tf.nn.top_k(preds, k).indices
#     print("coast time1: %f, %s" % (time.clock() - startTime, top_k))
#     # with tf.Session() as sess:
#     top_k = sess.run(top_k)
#     print("coast time2: %f, %s" % (time.clock() - startTime, top_k))
#     top_k_label = vec2label(top_k)
#     # endTime = time.clock()
#     print("coast time3: %f" % (time.clock()-startTime))
#     return top_k_label


def get_top_k_label(preds, k=1):
    """

    :param preds:[][]
    :param k:
    :return:
    """
    start_time = time.clock()
    top_ks = []
    for i in range(len(preds[:])):
        top_ks.append(tf.nn.top_k(preds[i], k).indices)
    print("cost time1: %f" % (time.clock() - start_time))
    # print("top_ks: %s" % top_ks)
    with tf.Session() as sess:
        top_ks = sess.run(top_ks)
        print("cost time2: %f" % (time.clock() - start_time))
        # print("top_ks: %s" % top_ks)
    top_k_labels = []
    for i in range(len(top_ks)):
        top_k_labels.append(vec2label(top_ks[i]))
    print("cost time3: %f" % (time.clock()-start_time))
    return top_k_labels


def get_all_top_k_label(preds, k=1):
    """
    :param preds:
    :param k:
    :return:
    """
    all_result_label = []
    length = len(preds[:])
    for i in range(length):
        top_k_labels = get_top_k_label(preds[i], k)
        all_result_label.append(top_k_labels)
    return all_result_label


# result_labels = get_top_k_label(test_predicts[0], 5)
get_top_k_label(test_predicts[:5], 5)
result_labels = get_top_k_label(test_predicts, 5)
# result_labels = get_all_top_k_label(test_predicts, 5)
print(result_labels)
range(len(test_predicts[:]))

index = test_set_names
# "image_name",
columns = ["predict1", "predict2", "predict3", "predict4", "predict5"]
print(test_set_names[:2])
len(index)
len(result_labels)
result_labels_df = pd.DataFrame(data=result_labels, index=index, columns=columns)
result_labels_df.to_csv("demo/predict_result_sgd.csv")
result_labels_df.to_csv("demo/predict_result_adam.csv")
result_labels_df.to_csv("demo/predict_result_adadelta.csv")
