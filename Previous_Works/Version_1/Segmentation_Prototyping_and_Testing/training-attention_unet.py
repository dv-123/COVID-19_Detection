from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, \
    add, multiply, Conv2DTranspose
from keras.layers import concatenate, core, Dropout
from keras.models import Model
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.core import Lambda
import keras.backend as K
import keras as k

import numpy as np
import os
import random

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import time

# may work better with x-ray images but performs bad on CT-images
# or
# the data on which the network is trained can be very less as compaired to the
# size of the network

seed = 99
np.random.seed = seed

NAME = "Attention-U-D2-Net_CNN-{}".format(int(time.time()))

data_format = 'channels_last'

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


def attention_up_and_concate(down_layer, layer, data_format = data_format):
    if data_format == 'channel_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    up = UpSampling2D(size = (2,2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel//4, data_format = data_format)

    if data_format == 'channel_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate =  my_concat([up, layer])

    return concate


def attention_block_2d(x, g, inter_channel, data_format=data_format):
    theta_x = Conv2D(inter_channel, [1,1], strides=[1,1], data_format=data_format)(x)

    phi_g = Conv2D(inter_channel, [1,1], strides=[1,1], data_format=data_format)(g)

    f = Activation('relu')(add([theta_x,phi_g]))

    psi_f = Conv2D(1, [1,1], strides=[1,1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    att_x = multiply([x,rate])

    return att_x


################################## image preprocessing ###########################################

print("For Train Dataset !")

#DATA_TRAIN = "Dataset_1/Train/"
DATA_TRAIN = "Dataset_2/Train/"

Train_X_path = DATA_TRAIN + "ct_scans/"
Train_Y_path = DATA_TRAIN + "infection_mask/"

x_ids_train = next(os.walk(Train_X_path))[1]
y_ids_train = next(os.walk(Train_Y_path))[1]

X_path_train = []
Y_path_train = []

for id in x_ids_train:
    path_x = Train_X_path + id +'/'
    path_y = Train_Y_path + id +'/'
    X_path_train.append(path_x)
    Y_path_train.append(path_y)

s_tr = 0
for n in X_path_train:
    img_ids = next(os.walk(n))[2]
    for i in img_ids:
        s_tr = s_tr+1


# important par of inniciating
X_train = np.zeros((s_tr, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((s_tr, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

count_x = 0
print("Processing for X train data !")
for n,path in tqdm(enumerate(X_path_train), total=len(X_path_train)):
    img_ids = next(os.walk(path))[2]
    for img in img_ids:
        image = imread(path + img)
        image = resize(image, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
        X_train[count_x] = image
        count_x = count_x+1


count_y = 0
print("Processing for Y train data !")
for n,path in tqdm(enumerate(Y_path_train), total=len(Y_path_train)):
    img_ids = next(os.walk(path))[2]
    for img in img_ids:
        image = imread(path + img)
        image = np.expand_dims(resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
        Y_train[count_y] = image
        count_y = count_y+1


############################## now making the attention U-net ####################################

inputs = Input((IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))
x = inputs
depth = 4
features = 64
num_class = 1
skips = []

for i in range(depth):
    x = Conv2D(features, (3,3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3,3), activation='relu', padding='same', data_format=data_format)(x)
    skips.append(x)
    x = MaxPooling2D((2,2), data_format=data_format)(x)
    features = features*2

x = Conv2D(features, (3,3), activation='relu', padding='same', data_format=data_format)(x)
x = Dropout(0.2)(x)
x = Conv2D(features, (3,3), activation='relu', padding='same', data_format=data_format)(x)

for i in reversed(range(depth)):
    features = features//2
    x = attention_up_and_concate(x, skips[i], data_format=data_format)
    x = Conv2D(features, (3,3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3,3), activation='relu', padding='same', data_format=data_format)(x)

output = Conv2D(num_class, (1,1), activation='sigmoid', padding='same', data_format=data_format)(x)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

###################################################################################################

callbacks = [
    k.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    k.callbacks.TensorBoard(log_dir='logs\{}'.format(NAME)),
    k.callbacks.ModelCheckpoint('model_D2_for_Covid-19_Attention_U_net.h5', verbose=1, save_best_only=True)]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=10, epochs=25, verbose=1, callbacks=callbacks)



# to run the tensorboard command
# tensorboard --logdir=logs/
# now run the command and the oprn the browser
# you will get a local host address copy that address
# in the browser address bar type following:
# pat the copied address
# and press enter
