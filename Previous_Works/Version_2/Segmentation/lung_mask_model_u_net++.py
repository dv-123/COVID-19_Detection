import tensorflow as tf
import numpy as np
import os
import random
import cv2

from tqdm import tqdm
from sklearn.utils import shuffle
from tensorflow.keras.regularizers import l2

from skimage.io import imread, imshow
from skimage.transform import resize
import skimage
import matplotlib.pyplot as plt
import time

seed = 80
np.random.seed = seed

NAME = "lung_mask_++-{}".format(int(time.time()))

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


deep_supervision = False
dropout_rate = 0.5
num_class = 1

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


##########################################Train_Dataset############################################

print("For Train Dataset !")

DATA_TRAIN = "Dataset/Train/"

Train_X_path = DATA_TRAIN + "Images/"
Train_Y_path = DATA_TRAIN + "Masks/Lung_mask/"

x_ids_train = next(os.walk(Train_X_path))[2]
y_ids_train = next(os.walk(Train_Y_path))[2]

X_path_train = []
Y_path_train = []

for id in x_ids_train:
    path_x = Train_X_path + id
    X_path_train.append(path_x)

for id in y_ids_train:
    path_y = Train_Y_path + id
    Y_path_train.append(path_y)

s_tr = len(Y_path_train) #both the lengths are same

# important par of inniciating
X_train = np.zeros((s_tr, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((s_tr, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

count_x = 0
print("Processing for X train data !")
for n,path in tqdm(enumerate(X_path_train), total=len(X_path_train)):
    image = imread(path)
    # Applying CLAHE:
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    # Applying min-max Contrast streching
    #min = np.min(image)
    #max = np.max(image)
    #image_new = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
    #for i in range(image.shape[0]):
    #    for j in range(image.shape[1]):
    #        image_new[i,j] = 255*(image[i,j]-min)/(max-min)
    #image = image_new
    image = skimage.color.gray2rgb(image)
    image = resize(image, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
    X_train[count_x] = image
    count_x = count_x+1

count_y = 0
print("Processing for Y train data !")
for n,path in tqdm(enumerate(Y_path_train), total=len(Y_path_train)):
    image = imread(path)
    image = np.expand_dims(resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
    Y_train[count_y] = image
    count_y = count_y+1

X_train_new,Y_train_new = shuffle(X_train,Y_train, random_state=2)
################################################Model############################################################

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

up1_2 = tf.keras.layers.Conv2DTranspose(nb_filter[0], (2,2), strides = (2,2), name = 'up12', padding = 'same')(conv2_1)
conv1_2 = tf.keras.layers.concatenate([up1_2, conv1_1], name = 'merge12') # check if axis is necessary
conv1_2 = standard_unit(conv1_2, stage = '12', nb_filter=nb_filter[0])
### 3
conv3_1 = standard_unit(pool2, stage = '31', nb_filter = nb_filter[2])
pool3 = tf.keras.layers.MaxPooling2D((2,2), name = 'pool3')(conv3_1)

up2_2 = tf.keras.layers.Conv2DTranspose(nb_filter[1], (2,2), strides = (2,2), name = 'up22', padding = 'same')(conv3_1)
conv2_2 = tf.keras.layers.concatenate([up2_2, conv2_1], name = 'merge22') # check for axis
conv2_2 = standard_unit(conv2_2, stage = '22', nb_filter = nb_filter[1])

up1_3 = tf.keras.layers.Conv2DTranspose(nb_filter[0], (2,2), strides = (2,2), name = 'up13', padding = 'same')(conv2_2)
conv1_3 = tf.keras.layers.concatenate([up1_3, conv1_1, conv1_2], name = 'merge13') # check for axis
conv1_3 = standard_unit(conv1_3, stage = '13', nb_filter = nb_filter[0])
### 4
conv4_1 = standard_unit(pool3, stage = '41', nb_filter = nb_filter[3])
pool4 = tf.keras.layers.MaxPooling2D((2,2), strides = (2,2), name='pool4')(conv4_1)

up3_2 = tf.keras.layers.Conv2DTranspose(nb_filter[2], (2,2), strides = (2,2), name = 'up32', padding = 'same')(conv4_1)
conv3_2 = tf.keras.layers.concatenate([up3_2, conv3_1], name = 'merge32') #check fo axis
conv3_2 = standard_unit(conv3_2, stage = '32', nb_filter = nb_filter[2])

up2_3 = tf.keras.layers.Conv2DTranspose(nb_filter[1], (2,2), strides = (2,2), name = 'up23', padding = 'same')(conv3_2)
conv2_3 = tf.keras.layers.concatenate([up2_3, conv2_1, conv2_2], name = 'merge23') # check for axis
conv2_3 = standard_unit(conv2_3, stage = '23', nb_filter = nb_filter[1])

up1_4 = tf.keras.layers.Conv2DTranspose(nb_filter[0], (2,2), strides = (2,2), name = 'up14', padding = 'same')(conv2_3)
conv1_4 = tf.keras.layers.concatenate([up1_4, conv1_1, conv1_2, conv1_3], name = 'merge14') # check for axis
conv1_4 = standard_unit(conv1_4, stage = '14', nb_filter = nb_filter[0])
### 5
conv5_1 = standard_unit(pool4, stage='51', nb_filter = nb_filter[4])

up4_2 = tf.keras.layers.Conv2DTranspose(nb_filter[3], (2,2), strides = (2,2), name = 'up42', padding = 'same')(conv5_1)
conv4_2 = tf.keras.layers.concatenate([up4_2, conv4_1], name='merge42') # check for axis
conv4_2 = standard_unit(conv4_2, stage = '42', nb_filter = nb_filter[3])

up3_3 = tf.keras.layers.Conv2DTranspose(nb_filter[2], (2,2), strides = (2,2), name = 'up33', padding = 'same')(conv4_2)
conv3_3 = tf.keras.layers.concatenate([up3_3, conv3_1, conv3_2], name = 'merge33') # check for axis
conv3_3 = standard_unit(conv3_3, stage = '33', nb_filter = nb_filter[2])

up2_4 = tf.keras.layers.Conv2DTranspose(nb_filter[1], (2,2), strides = (2,2), name = 'up24', padding = 'same')(conv3_3)
conv2_4 = tf.keras.layers.concatenate([up2_4, conv2_1, conv2_2, conv2_3], name = 'merge24') # check for axis
conv2_4 = standard_unit(conv2_4, stage = '24', nb_filter = nb_filter[1])

up1_5 = tf.keras.layers.Conv2DTranspose(nb_filter[0], (2,2), strides = (2,2), name = 'up15', padding = 'same')(conv2_4)
conv1_5 = tf.keras.layers.concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name = 'merge15') # check for axis
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

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs\{}'.format(NAME)),
    tf.keras.callbacks.ModelCheckpoint('lung_mask_nested_unet.h5', verbose=1, save_best_only=True)]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=10, epochs=25, callbacks=callbacks)



# to run the tensorboard command
# tensorboard --logdir=logs/
# now run the command and the oprn the browser
# you will get a local host address copy that address
# in the browser address bar type following:
# pat the copied address
# and press enter
