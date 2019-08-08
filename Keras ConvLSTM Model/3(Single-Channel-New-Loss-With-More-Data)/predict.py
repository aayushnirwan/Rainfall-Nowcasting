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
import new_loss

val_input = np.load('val_sequences_400_all1.npy')
# val_input = val_input/255.0
val_input = np.where(val_input>0,1,0)
# val_input = np.expand_dims(val_input, axis=-1)
# val_label1 = np.load('val_labels_400.npy')
val_label = np.load('val_labels_400_all1.npy')
# val_label = val_label/255.0
val_label = np.where(val_label>0,255,0)
print (val_input.shape)
# print (val_input[0][9].shape)
print (val_label.shape)
# print (val_input[0:1,:,:,:,:].shape)
# val_label1 = np.squeeze(np.expand_dims(val_label1, axis=-1), axis=1)
# print (val_label1.shape)
# train_input = np.load('train_sequences_400_new.npy')
# train_input = train_input[0:1,:,:,:,:]
# train_input = train_input/255.0
# train_input = np.where(train_input>0,255,0)
# train_input = train_input[0][9]
# train_input = np.squeeze(train_input,axis=-1)
# print (train_input.shape)


# train_label = np.load('train_labels_400_new.npy')
# train_label = train_label[0]
# train_label = np.squeeze(train_label,axis=-1)
# train_label = train_label/255.0
# train_label = np.where(train_label>0,255,0)
# print (train_label.shape)

import cv2
# cv2.imwrite('gt.jpeg',train_label)
# cv2.imwrite('a.jpeg',train_input)

model = load_model("loss.0.5597-013.h5", custom_objects={'focal_tversky': new_loss.focal_tversky})
# model = load_model("loss.0.0031-024.h5")
print ("Model Loaded")
# pred = model.predict(val_input[0:1, ...])
# pred = pred[0]
# pred = np.squeeze(pred,axis=-1)
# pred = np.where(pred==1,255,0)
# print (pred.shape)
# cv2.imwrite("pred.jpeg",pred)
# print (pred[0])
# print('-')
# print(val_label[0])
# print (pred.shape)
# fig = plt.figure()
# data = val_label[0, :, :, 0]
#
# im = plt.imshow(data)
# plt.show()
# val_label = val_label/255
# val_label1=val_label1[10]
# val_input=val_input[0]
# print (val_label1.shape)
# pred = pred[0]
# print (pred.shape)
# # print (np.unique(pred,return_counts=True))
# #
# val_label=val_label[0]
# print (val_label.shape)
# print (np.unique(pred[0],return_counts=True))
# pred = pred[0]
# print (pred)
# c = 0
# for x in pred:
#     for y in x:
#         if y[0]==0:
#             c+=1
# print (c)
#
# c = 0
# for x in val_label:
#     for y in x:
#         if y[0]==0:
#             c+=1
# print (c)
# pred = pred[0]
# import cv2
# pred=np.where(np.argmax(pred,axis=2)==0,255,0)
# pred = np.squeeze(pred,axis=-1)
# print (pred.shape)
# print (np.unique(pred,return_counts=True))
# pred = np.where(pred==1,255,0)
# print (pred.shape)
# print (np.unique(pred,return_counts=True))

# cv2.imshow('pred',pred)
#

# #
# fig = plt.figure()
# data = val_label[0, :, :, 0]
# x = val_label[2]
# val_label = val_label[0]
# val_label = val_label/255.0
# val_label = np.where(val_label==1,255,0)
# val_label = np.squeeze(val_label,axis=-1)
# print (val_label.shape)
# print(np.unique(val_label,return_counts=True))
# val_label = np.where(np.argmax(val_label,axis=2)==0,255,0)
# print (val_label.shape)
# im = plt.imshow(val_label)
# plt.show()
from tqdm import tqdm

# print (pred.shape)
import cv2
# x=np.where(np.argmax(pred,axis=2)==0,255,0)
# print(np.unique(x,return_counts=True))
# print(np.unique(x,return_counts=True))
# x = np.squeeze(x,axis=-1)
# print (x.shape)
# cv2.imwrite("pred2.jpeg",pred)
# cv2.imwrite("gt2.jpeg",val_label)
# cv2.imwtite("gt3.jpeg",x)
for i in tqdm(range(370)):
    pred = model.predict(val_input[i:i+1,:,:,:,:])
    pred = pred[0]
    pred = np.squeeze(pred,axis=-1)

    # print (np.unique(pred,return_counts=True))

    # pred1 = np.where(pred>0,255,0)
    # print (np.unique(pred1,return_counts=True))

    pred = np.where(pred==1,255,0)
    # print (np.unique(pred2,return_counts=True))
    cv2.imwrite("results/{}.jpeg".format(i),pred)
    #
    x = val_label[i]
    x = np.squeeze(x,axis=-1)
    x = np.where(x>0,255,0)
    # print (np.unique(x,return_counts=True))
    cv2.imwrite("results/{} (1).jpeg".format(i),x)
    #
    # for j in range(10):
    #     y = val_input[i][j]
    #     y = np.squeeze(y,axis=-1)
    #     y = np.where(y>0,255,0)
    #     cv2.imwrite("results/{} ({}).jpeg".format(i,j),y)



    x=np.where(val_input[i][9]>0,255,0)
    cv2.imwrite("results/{} (0).jpeg".format(i),x)

# print (val_label.shape)
