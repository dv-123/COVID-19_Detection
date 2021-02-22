import numpy as np
import os
import time
import tensorflow as tf
from resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten
from tqdm import tqdm
from imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 80
np.random.seed = seed

NAME = "test_2-{}".format(int(time.time()))

TRAINING_DIR = "Dataset/Train/"
TESTING_DIR = "Dataset/Test/"

data_dir_list = os.listdir(TRAINING_DIR)

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

num_classes = 3
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples), dtype = 'int64')

labels[0:340] = 0
labels[340:788] = 1
labels[788:1236] = 2

names = ['Covid-19','No_findings','Pneumonia']

Y = np_utils.to_categorical(labels, num_classes)

x,y = shuffle(img_data,Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 2)

######model --> Transfer learning ############

model = ResNet50(include_top = False, weights = 'imagenet')

model.summary()

last_layer = model.output
x = GlobalAveragePooling2D()(last_layer)
x = Dense(512, activation='relu', name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', name='fc-2')(x)
X = Dropout(0.5)(x)
out = Dense(num_classes, activation = 'softmax', name = 'output')(x)
custom_resnet_model = Model(model.input, out)

for layer in custom_resnet_model.layers[:-6]:
    layer.trainable = False

custom_resnet_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

custom_resnet_model.summary()

##############

callbacks = [
    #tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs\{}'.format(NAME)),
    tf.keras.callbacks.ModelCheckpoint('test_model_2.h5', save_best_only=True)]

custom_resnet_model.fit(X_train, y_train, batch_size=10, epochs=50, verbose=1, validation_data = (X_test, y_test), callbacks=callbacks)

# to run the tensorboard command
# tensorboard --logdir=logs/
# now run the command and the oprn the browser
# you will get a local host address copy that address
# in the browser address bar type following:
# pat the copied address
# and press enter
