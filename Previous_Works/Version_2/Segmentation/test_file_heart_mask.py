import tensorflow as tf

import numpy as np
import os
import cv2

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import skimage
import matplotlib.pyplot as plt


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

##################################### image preprocessing ###########################################


###########################################Test_Dataset##############################################

print("For Test Dataset !")

DATA_TEST = "Dataset/Test/"

Test_X_path = DATA_TEST + "Images/"
Test_Y_path = DATA_TEST + "Masks/Heart/"

x_ids_test = next(os.walk(Test_X_path))[2]
y_ids_test = next(os.walk(Test_Y_path))[2]

X_path_test = []
Y_path_test = []

for id in x_ids_test:
    path_x = Test_X_path + id
    X_path_test.append(path_x)

for id in y_ids_test:
    path_y = Test_Y_path + id
    Y_path_test.append(path_y)

s_tr = len(Y_path_test) #both the lengths are same

# important par of inniciating
X_test = np.zeros((s_tr, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((s_tr, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

count_x = 0
print("Processing for X test data !")
for n,path in tqdm(enumerate(X_path_test), total=len(X_path_test)):
    image = imread(path)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    image = skimage.color.gray2rgb(image)
    image = resize(image, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
    X_test[count_x] = image
    count_x = count_x+1

count_y = 0
print("Processing for Y test data !")
for n,path in tqdm(enumerate(Y_path_test), total=len(Y_path_test)):
    image = imread(path)
    image = np.expand_dims(resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
    Y_test[count_y] = image
    count_y = count_y+1

#########################################loading_Model#################################################

model  = tf.keras.models.load_model('model_X_ray_heart_mask.h5')
model.summary()

##########################################Pridictions#################################################
preds_test = model.predict(X_test, verbose = 1)

preds_test_t = (preds_test > 0.5).astype(np.uint8)

ix = 22
imshow(X_test[ix])
plt.show()
imshow(np.squeeze(Y_test[ix]))
plt.show()
imshow(np.squeeze(preds_test_t[ix]))
plt.show()
imshow(np.squeeze(preds_test[ix]))
plt.show()

# performs better than U-net
