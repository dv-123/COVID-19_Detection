import os
import random
import tensorflow as tf
import time
import numpy as np
from tqdm import tqdm
from vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge,Input
from sklearn.utils import shuffle
from keras.models import Model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

seed = 80
np.random.seed = seed
# original shape = 512x512
NAME = "test_3-{}".format(int(time.time()))

TRAINING_DIR = "Dataset/Train/"
TESTING_DIR = "Dataset/Test/"

data_dir_list = os.listdir(TRAINING_DIR)
temp1 = data_dir_list[0]
temp2 = data_dir_list[2]
data_dir_list = [temp1,temp2]

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(TRAINING_DIR+dataset+'/')
    print('Loading the images of dataset - '+ '{}\n'.format(dataset))
    for img in tqdm(img_list):
        img_path = TRAINING_DIR+dataset+'/'+img
        img = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        img_data_list.append(x)

img_data = np.array(img_data_list)
img_data = np.rollaxis(img_data,1,0)
img_data = img_data[0]
print(img_data.shape)

num_classes = 2
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples), dtype = 'int64')

labels[0:333] = 0
labels[333:727] = 1

names = ['Covid-19','Pneumonia']

Y = np_utils.to_categorical(labels, num_classes)

x,y = shuffle(img_data,Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.05, random_state = 2)

######model --> Transfer learning ############

image_input = Input(shape=(224,224,3))

model = VGG16(input_tensor=image_input, include_top = True, weights = 'imagenet')

model.summary()

last_layer = model.get_layer('fc2').output
x = Dense(128, activation='relu', name='fc3')(last_layer)
x = Dense(128, activation='relu', name='fc4')(x)
out = Dense(num_classes, activation = 'softmax', name = 'output')(x)
custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()

for layer in custom_vgg_model.layers[:-3]:
    layer.trainable = False

custom_vgg_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

custom_vgg_model.summary()


##############

callbacks = [
    #tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs\{}'.format(NAME)),
    tf.keras.callbacks.ModelCheckpoint('test_model_3.h5', save_best_only=True)]

custom_vgg_model.fit(X_train, y_train, batch_size=10, epochs=16, verbose=1, validation_data = (X_test, y_test), callbacks=callbacks)

# to run the tensorboard command
# tensorboard --logdir=logs/
# now run the command and the oprn the browser
# you will get a local host address copy that address
# in the browser address bar type following:
# pat the copied address
# and press enter
