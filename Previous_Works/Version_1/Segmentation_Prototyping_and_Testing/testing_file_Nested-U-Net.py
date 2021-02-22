import tensorflow as tf

import numpy as np
import os

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

##################################### image preprocessing ###########################################


###########################################Test_Dataset##############################################
print("For Test Dataset !")

#DATA_TEST ="Dataset_1/Test/"
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

#########################################loading_Model#################################################

model  = tf.keras.models.load_model('model_D2_for_Covid-19_Nested_U_net.h5')
model.summary()

##########################################Pridictions#################################################
preds_test = model.predict(X_test, verbose = 1)

preds_test_t = (preds_test > 0.5).astype(np.uint8)

ix = 78
imshow(X_test[ix])
plt.show()
imshow(np.squeeze(Y_test[ix]))
plt.show()
imshow(np.squeeze(preds_test_t[ix]))
plt.show()
imshow(np.squeeze(preds_test[ix]))
plt.show()

# performs better than U-net
