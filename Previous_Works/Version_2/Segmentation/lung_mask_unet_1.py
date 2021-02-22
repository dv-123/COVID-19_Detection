import tensorflow as tf
import numpy as np
import os
import random
import cv2

from tqdm import tqdm
from sklearn.utils import shuffle

from skimage.io import imread, imshow
from skimage.transform import resize
import skimage
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

seed = 80
#np.random.seed = seed

NAME = "lung_mask_1-{}".format(int(time.time()))

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

##########################################Train_Dataset############################################


print("For Train Dataset !")

DATA_TRAIN = "Dataset_1/Train/"

Train_X_path = DATA_TRAIN + "Images/"
Train_Y_path = DATA_TRAIN + "Masks/Lungs/"

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

X_train_1, X_test, y_train_1, y_test = train_test_split(X_train_new,Y_train_new, test_size = 0.1, random_state = 2)

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)


#image_datagen.fit(X_train_new, augment=True, seed=seed)
#mask_datagen.fit(Y_train_new, augment=True, seed=seed)
#image_generator = image_datagen.flow_from_directory(
#    "Dataset_1/Train/Img/",
#    class_mode=None,
#    target_size=(128,128),
#    seed=seed)
#mask_generator = mask_datagen.flow_from_directory(
#    "Dataset_1/Train/Msk",
#    class_mode=None,
#    target_size=(128,128),
#    seed=seed)
image_generator = image_datagen.flow(X_train_1, batch_size=32, seed = seed)
mask_generator = mask_datagen.flow(y_train_1, batch_size=32, seed = seed)
#datagen = ImageDataGenerator(rotation_range = 90, width_shift_range = 0.2, height_shift_range = 0.2,
#                            shear_range = 0.2, zoom_range=0.2, horizontal_flip = True, fill_mode='nearest')

#train_generator = datagen.flow(X_train_1,y_train_1, batch_size=32)
train_generator = (pair for pair in zip(image_generator, mask_generator))
#train_generator = zip(image_generator, mask_generator)

################################################Model############################################################
#Build the model
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs) #for x devide x by 255 and x corresponds to inputs
# this keras.lambds is same as the regular lambda function in regular python.
# where in regular python instead of defining as regular function by def we can use the
# lambda function to unwrap the function within that line.

# Contraction Path
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
# so we have to start with some weights in the network which are the tuned, and kernel_initializer
# is used to define the distribution of initialized weights.
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive Path
u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# since we will be working with a binary classification i.e. either a cell or not a cell so we will be using
# binary_crossentropy as our loss function.
model.summary()

# code for model fitting, input data, earlystoping, checkpoints and callbacks.

# ModelCheckpoint

callbacks = [
#    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs\{}'.format(NAME)),
    tf.keras.callbacks.ModelCheckpoint('model_X_ray_lung_mask_1_2.h5', verbose=1, save_best_only=True)]

results = model.fit_generator(train_generator, steps_per_epoch=2000, epochs=101, verbose=1, validation_data = (X_test, y_test), callbacks=callbacks)
#results = model.fit_generator(
#    train_generator,
#    steps_per_epoch=2000,
#    epochs=50, callbacks = callbacks)

# to run the tensorboard command
# tensorboard --logdir=logs/
# now run the command and the oprn the browser
# you will get a local host address copy that address
# in the browser address bar type following:
# pat the copied address
# and press ente
