import pandas as pd
import numpy as np
import datetime
import os
import sys
from datetime import datetime
from datetime import timedelta
# import matplotlib.pyplot as plt
import h5py

from keras.models import Model
from keras.layers import Input, Dense, MaxPooling2D, MaxPooling3D, Dropout, BatchNormalization, Flatten, Conv2D, Conv3D, AveragePooling3D, LSTM, Reshape
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard


from keras.layers import AveragePooling2D
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D

import tensorflow as tf
import new_loss


def fn_get_model_convLSTM_2():

    model = Sequential()

    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7),
                         input_shape=(None, 400, 400, 1),
                         return_sequences=True,
                         go_backwards=True,
                         padding='same',
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.2
                         ))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=16, kernel_size=(7, 7),
                         return_sequences=True,
                         go_backwards=True,
                         padding='same',
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.2
                         ))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=8, kernel_size=(7, 7),
                         return_sequences=False,
                         go_backwards=True,
                         padding='same',
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.3, recurrent_dropout=0.2
                         ))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=16, kernel_size=(1, 1),
                     activation='relu',
                     padding='same',
                     data_format='channels_last'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                     activation='sigmoid',
                     padding='same',
                     data_format='channels_last'))

    print(model.summary())

    return model


train_input = np.load('train_sequences_400_new.npy')
train_label = np.load('train_labels_400_new.npy')
val_input = np.load('val_sequences_400_new.npy')
val_label = np.load('val_labels_400_new.npy')
# train_input = np.expand_dims(train_input, axis=-1)
# train_label = np.squeeze(np.expand_dims(train_label, axis=-1), axis=1)
# val_input = np.expand_dims(val_input, axis=-1)
# val_label = np.squeeze(np.expand_dims(val_label, axis=-1), axis=1)


model = fn_get_model_convLSTM_2()

# def custom_loss():
#
#     # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
#     def loss(y_true,y_pred):
#         return K.mean(K.square(y_pred - y_true), axis=-1)
#
#     # Return a function
#     return loss


# def loss(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true), axis=-1)


# Compile the model
model.compile(loss=new_loss.focal_tversky, optimizer='adam', metrics=['accuracy'])
# model.compile(loss=new_loss.focal_tversky, optimizer='adam', metrics=['accuracy'])

photo = TensorBoard(log_dir='logs')
checkpoint_val = ModelCheckpoint('models/val_loss.{val_loss:.4f}-{epoch:03d}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
checkpoint_loss = ModelCheckpoint('models/loss.{loss:.4f}-{epoch:03d}.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.fit(x=train_input, y=train_label, batch_size=1, epochs=300, verbose=1, callbacks=[checkpoint_loss, checkpoint_val, photo], validation_data=(val_input, val_label), shuffle=True)

# model.fit(x=train_input[0:2,:,:,:,:], y=train_label[0:2,:,:,:], batch_size=1, epochs=300, verbose=1, validation_data=(val_input[0:1,:,:,:,:], val_label[0:1,:,:,:]), shuffle=True)
