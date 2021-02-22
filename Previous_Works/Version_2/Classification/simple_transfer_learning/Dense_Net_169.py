import os
import random
import tensorflow as tf
import time
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Input
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils import np_utils
from sklearn.model_selection import train_test_split

seed = 80
np.random.seed = seed
# original shape = 512x512
NAME = "test_densenet_2-{}".format(int(time.time()))

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
img_data = img_data.astype('int32')
img_data = np.abs(img_data)
#img_data = img_data/255 # can be removed if accuracy is not good
print(img_data.shape)

num_classes = 3
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples), dtype = 'int64')

labels[0:333] = 0
labels[333:720] = 1
labels[720:1114] = 2

names = ['Covid-19','No_findings','Pneumonia']

Y = np_utils.to_categorical(labels, num_classes)

x,y = shuffle(img_data,Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 2)

######model --> Transfer learning ############

model = tf.keras.applications.DenseNet169(
    include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
    pooling=None, classes=1000
)

image_input = Input(shape=(224,224,3))

last_layer = model.get_layer('avg_pool').output
x = Dense(1000, activation='relu', name='fc1')(last_layer)
x = Dense(512, activation='relu', name='fc2')(x)
x = Dense(128, activation='relu', name='fc3')(x)
out = Dense(num_classes, activation = 'softmax', name = 'output')(x)
custom_model = Model(model.input, out)

for layer in custom_model.layers[:-4]:
    layer.trainable = False

custom_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

custom_model.summary()


callbacks = [
    #tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs\{}'.format(NAME)),
    tf.keras.callbacks.ModelCheckpoint('test_model_dense_net_2.h5', save_best_only=True)]


custom_model.fit(X_train, y_train, batch_size=10, epochs=100, verbose=1, validation_data = (X_test, y_test), callbacks=callbacks)

# to run the tensorboard command
# tensorboard --logdir=logs/
# now run the command and the oprn the browser
# you will get a local host address copy that address
# in the browser address bar type following:
# pat the copied address
# and press enter
