import tensorflow as tf
import numpy as np
import os
import random

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import time

seed = 80
np.random.seed = seed

NAME = "U-D2-Net_CNN-{}".format(int(time.time()))

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

##########################################Train_Dataset############################################
print("For Train Dataset !")

DATA_TRAIN = "Dataset_1/Train/"
#DATA_TRAIN = "Dataset_2/Train/"

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

###########################################Test_Dataset##############################################
print("For Test Dataset !")

DATA_TEST = "Dataset_2/Test/"

Test_X_path = DATA_TEST + "ct_scans/"
Test_Y_path = DATA_TEST + "infection_mask/"

x_ids_test = next(os.walk(Test_X_path))[1]
y_ids_test = next(os.walk(Test_Y_path))[1]

X_path_test = [];
Y_path_test = [];

for id in x_ids_test:
    path_x = Test_X_path + id +'/'
    path_y = Test_Y_path + id +'/'
    X_path_test.append(path_x)
    Y_path_test.append(path_y)

s_ts = 0
for n in X_path_test:
    img_ids = next(os.walk(n))[2]
    for i in img_ids:
        s_ts = s_ts+1

# important par of inniciating
X_test = np.zeros((s_ts, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((s_ts, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

count_x = 0
print("Processing for X test data !")
for n,path in tqdm(enumerate(X_path_test), total=len(X_path_test)):
    img_ids = next(os.walk(path))[2]
    for img in img_ids:
        image = imread(path + img)
        image = resize(image, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
        X_test[count_x] = image
        count_x = count_x+1

count_y = 0
print("Processing for Y test data !")
for n,path in tqdm(enumerate(Y_path_test), total=len(Y_path_test)):
    img_ids = next(os.walk(path))[2]
    for img in img_ids:
        image = imread(path + img)
        image = np.expand_dims(resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
        Y_test[count_y] = image
        count_y = count_y+1

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
    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs\{}'.format(NAME)),
    tf.keras.callbacks.ModelCheckpoint('model_D2_for_Covid-19_U_net.h5', verbose=1, save_best_only=True)]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)


# testing the model
idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose = 1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose = 1)
preds_test = model.predict(X_test, verbose = 1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# perform a sanity check on some random validation samples
ix = random.randint(0,len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

# to run the tensorboard command
# tensorboard --logdir=logs/
# now run the command and the oprn the browser
# you will get a local host address copy that address
# in the browser address bar type following:
# pat the copied address
# and press enter
