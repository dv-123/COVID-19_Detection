import tensorflow as tf
from tensorflow.keras.regularizers import l2
import numpy as np
import os
import random

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import time

seed = 99
np.random.seed = seed

NAME = "Attention_Nested-U_D2-Net_CNN-{}".format(int(time.time()))

data_format = 'channels_last'

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

deep_supervision = False
dropout_rate = 0.5
num_class = 1

def attention_block_2d(x, g, inter_channel, data_format=data_format):
    theta_x = tf.keras.layers.Conv2D(inter_channel, [1,1], strides=[1,1], data_format=data_format)(x)

    phi_g = tf.keras.layers.Conv2D(inter_channel, [1,1], strides=[1,1], data_format=data_format)(g)

    f = tf.keras.layers.Activation('relu')(tf.keras.layers.add([theta_x,phi_g]))

    psi_f = tf.keras.layers.Conv2D(1, [1,1], strides=[1,1], data_format=data_format)(f)

    rate = tf.keras.layers.Activation('sigmoid')(psi_f)

    att_x = tf.keras.layers.multiply([x,rate])

    return att_x

def attention_up_and_concate(down_layer, layer, data_format = data_format):
    if data_format == 'channel_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    up = tf.keras.layers.UpSampling2D(size = (2,2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel//4, data_format = data_format)

    if data_format == 'channel_first':
        my_concat = tf.keras.layers.Lambda(lambda x: tf.keras.layers.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = tf.keras.layers.Lambda(lambda x: tf.keras.layers.concatenate([x[0], x[1]], axis=3))

    concate =  my_concat([up, layer])

    return concate

### defining a satndard 2D convolution unit ###
def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    act = 'relu'

    x = tf.keras.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation = act, name = 'conv'+stage+'_1',
                        kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = l2(1e-4))(input_tensor)
    x = tf.keras.layers.Dropout(dropout_rate, name = 'dp'+stage+'_1')(x)
    x = tf.keras.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation = act, name = 'conv'+stage+'_2',
                        kernel_initializer = 'he_normal', padding = 'same', kernel_regularizer = l2(1e-4))(x)
    x = tf.keras.layers.Dropout(dropout_rate, name = 'dp'+stage+'_2')(x)

    return x

#############################Image_Preprocessing################################################

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

########################## making the nested attention U-net ###################################

nb_filter = [32,64,128,256,512]
act = 'relu'

inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

### 1
conv1_1 = standard_unit(s, stage = '11', nb_filter = nb_filter[0])
pool1 = tf.keras.layers.MaxPooling2D((2,2), strides = (2,2), name = 'pool1')(conv1_1)
### 2
conv2_1 = standard_unit(pool1, stage = '21', nb_filter = nb_filter[1])
pool2 = tf.keras.layers.MaxPooling2D((2,2), strides = (2,2), name = 'pool2')(conv2_1)

#up1_2 = tf.keras.layers.Conv2DTranspose(nb_filter[0], (2,2), strides = (2,2), name = 'up12', padding = 'same')(conv2_1)
#conv1_2 = tf.keras.layers.concatenate([up1_2, conv1_1], name = 'merge12') # check if axis is necessary
conv1_2 = attention_up_and_concate(conv2_1, conv1_1, data_format=data_format)
conv1_2 = standard_unit(conv1_2, stage = '12', nb_filter=nb_filter[0])
### 3
conv3_1 = standard_unit(pool2, stage = '31', nb_filter = nb_filter[2])
pool3 = tf.keras.layers.MaxPooling2D((2,2), name = 'pool3')(conv3_1)

#up2_2 = tf.keras.layers.Conv2DTranspose(nb_filter[1], (2,2), strides = (2,2), name = 'up22', padding = 'same')(conv3_1)
#conv2_2 = tf.keras.layers.concatenate([up2_2, conv2_1], name = 'merge22') # check for axis
conv2_2 = attention_up_and_concate(conv3_1, conv2_1, data_format=data_format)
conv2_2 = standard_unit(conv2_2, stage = '22', nb_filter = nb_filter[1])

#up1_3 = tf.keras.layers.Conv2DTranspose(nb_filter[0], (2,2), strides = (2,2), name = 'up13', padding = 'same')(conv2_2)
#conv1_3 = tf.keras.layers.concatenate([up1_3, conv1_1, conv1_2], name = 'merge13') # check for axis
conv1_3 = attention_up_and_concate(conv2_2, conv1_1, data_format=data_format)
conv1_3 = standard_unit(conv1_3, stage = '13', nb_filter = nb_filter[0])
### 4
conv4_1 = standard_unit(pool3, stage = '41', nb_filter = nb_filter[3])
pool4 = tf.keras.layers.MaxPooling2D((2,2), strides = (2,2), name='pool4')(conv4_1)

#up3_2 = tf.keras.layers.Conv2DTranspose(nb_filter[2], (2,2), strides = (2,2), name = 'up32', padding = 'same')(conv4_1)
#conv3_2 = tf.keras.layers.concatenate([up3_2, conv3_1], name = 'merge32') #check fo axis
conv3_2 = attention_up_and_concate(conv4_1, conv3_1, data_format=data_format)
conv3_2 = standard_unit(conv3_2, stage = '32', nb_filter = nb_filter[2])

#up2_3 = tf.keras.layers.Conv2DTranspose(nb_filter[1], (2,2), strides = (2,2), name = 'up23', padding = 'same')(conv3_2)
#conv2_3 = tf.keras.layers.concatenate([up2_3, conv2_1, conv2_2], name = 'merge23') # check for axis
conv2_3 = attention_up_and_concate(conv3_2, conv2_1, data_format=data_format)
conv2_3 = standard_unit(conv2_3, stage = '23', nb_filter = nb_filter[1])

#up1_4 = tf.keras.layers.Conv2DTranspose(nb_filter[0], (2,2), strides = (2,2), name = 'up14', padding = 'same')(conv2_3)
#conv1_4 = tf.keras.layers.concatenate([up1_4, conv1_1, conv1_2, conv1_3], name = 'merge14') # check for axis
conv1_4 = attention_up_and_concate(conv2_3, conv1_1, data_format=data_format)
conv1_2 = standard_unit(conv1_4, stage = '14', nb_filter = nb_filter[0])
### 5
conv5_1 = standard_unit(pool4, stage='51', nb_filter = nb_filter[4])

#up4_2 = tf.keras.layers.Conv2DTranspose(nb_filter[3], (2,2), strides = (2,2), name = 'up42', padding = 'same')(conv5_1)
#conv4_2 = tf.keras.layers.concatenate([up4_2, conv4_1], name='merge42') # check for axis
conv4_2 = attention_up_and_concate(conv5_1, conv4_1, data_format=data_format)
conv4_2 = standard_unit(conv4_2, stage = '42', nb_filter = nb_filter[3])

#up3_3 = tf.keras.layers.Conv2DTranspose(nb_filter[2], (2,2), strides = (2,2), name = 'up33', padding = 'same')(conv4_2)
#conv3_3 = tf.keras.layers.concatenate([up3_3, conv3_1, conv3_2], name = 'merge33') # check for axis
conv3_3 = attention_up_and_concate(conv4_2, conv3_1, data_format=data_format)
conv3_3 = standard_unit(conv3_3, stage = '33', nb_filter = nb_filter[2])

#up2_4 = tf.keras.layers.Conv2DTranspose(nb_filter[1], (2,2), strides = (2,2), name = 'up24', padding = 'same')(conv3_3)
#conv2_4 = tf.keras.layers.concatenate([up2_4, conv2_1, conv2_2, conv2_3], name = 'merge24') # check for axis
conv2_4 = attention_up_and_concate(conv3_3, conv2_1, data_format=data_format)
conv2_4 = standard_unit(conv2_4, stage = '24', nb_filter = nb_filter[1])

#up1_5 = tf.keras.layers.Conv2DTranspose(nb_filter[0], (2,2), strides = (2,2), name = 'up15', padding = 'same')(conv2_4)
#conv1_5 = tf.keras.layers.concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name = 'merge15') # check for axis
conv1_5 = attention_up_and_concate(conv2_4, conv1_1, data_format=data_format)
conv1_5 = standard_unit(conv1_5, stage = '15', nb_filter = nb_filter[0])
### end

nested_output_1 = tf.keras.layers.Conv2D(num_class, (1,1), activation = 'sigmoid', name = 'output_1', kernel_initializer = 'he_normal',
                                    padding = 'same', kernel_regularizer = l2(1e-4))(conv1_2)
nested_output_2 = tf.keras.layers.Conv2D(num_class, (1,1), activation = 'sigmoid', name = 'output_2', kernel_initializer = 'he_normal',
                                    padding = 'same', kernel_regularizer = l2(1e-4))(conv1_3)
nested_output_3 = tf.keras.layers.Conv2D(num_class, (1,1), activation = 'sigmoid', name = 'output_3', kernel_initializer = 'he_normal',
                                    padding = 'same', kernel_regularizer = l2(1e-4))(conv1_4)
nested_output_4 = tf.keras.layers.Conv2D(num_class, (1,1), activation = 'sigmoid', name = 'output_4', kernel_initializer = 'he_normal',
                                    padding = 'same', kernel_regularizer = l2(1e-4))(conv1_5)


if deep_supervision:
    model = tf.keras.Model(inputs = [inputs], outputs = [nested_output_1,
                                                        nested_output_2,
                                                        nested_output_3,
                                                        nested_output_4])
else:
    model = tf.keras.Model(inputs = [inputs], outputs = [nested_output_4])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

################################################################################################################3

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs\{}'.format(NAME)),
    tf.keras.callbacks.ModelCheckpoint('model_D2_for_Covid-19_Nested_attention_U_net.h5', verbose=1, save_best_only=True)]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=10, epochs=25, callbacks=callbacks)



# to run the tensorboard command
# tensorboard --logdir=logs/
# now run the command and the oprn the browser
# you will get a local host address copy that address
# in the browser address bar type following:
# pat the copied address
# and press enter
