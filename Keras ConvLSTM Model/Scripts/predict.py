import pandas as pd
import numpy as np
import datetime
import os
import sys
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import h5py

from keras.models import Model, load_model
from keras.layers import Input, Dense, MaxPooling2D, MaxPooling3D, Dropout, BatchNormalization, Flatten, Conv2D, Conv3D, AveragePooling3D, LSTM, Reshape
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard


from keras.layers import AveragePooling2D
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D

import tensorflow as tf
import cv2


# def custom_loss():

#     # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
#     def loss(y_true, y_pred):
#         return K.mean(K.square(y_pred - y_true), axis=-1)

#     # Return a function
#     return loss
# def loss(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true), axis=-1)


val_input = np.load('val_sequences_400.npy')
val_input = np.expand_dims(val_input, axis=-1)
val_label = np.load('val_labels_400.npy')
val_label = np.squeeze(np.expand_dims(val_label, axis=-1), axis=1)


model = load_model("D:\\examples_d\\CNNs\\baadals\\CONVLSTM\\loss.48.3830-104.h5", custom_objects={'loss': loss})
print ("1")

pred = model.predict(val_input[10:11, :, :, :, :])
pred = pred[0]
pred = np.where(pred == 1, 255, 0)
pred = pred.astype(np.uint8)
cv2.imshow("pred", pred)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print('-')
# print(val_label[0])
# fig = plt.figure()
# data = pred[0, :, :, 0]
# im = plt.imshow(data)
# plt.subplot(2, 1, 1)
# plt.title('Prediction')

# data2 = val_label[0, :, :, 0]
# im2 = plt.imshow(data2)
# plt.subplot(2, 1, 2)
# plt.title('Ground Truth')
# plt.show()

val_label = val_label.astype(np.uint8)
# pred = pred.astype(np.uint8)

# cv2.imshow("pred", pred[0])
cv2.imshow("gt", val_label[10])

cv2.waitKey(0)
cv2.destroyAllWindows()
