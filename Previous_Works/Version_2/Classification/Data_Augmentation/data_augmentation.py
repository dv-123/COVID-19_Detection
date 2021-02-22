import os
import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

DATA_DIR = "Dataset"

data_dir_list = os.listdir(DATA_DIR)

img_data_list = []

for dataset in data_dir_list:
    img_list = os.listdir(DATA_DIR+dataset+'/')
    print('Loading the images of dataset - '+ '{}\n'.format(dataset))
    for img in tqdm(img_list):
        img_path = DATA_DIR+dataset+'/'+img
        img = image.load_img(img_path, target_size=(512,512))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        img_data_list.append(x)

img_data = np.array(img_data_list)
img_data = np.rollaxis(img_data,1,0)
img_data = img_data[0]
print(img_data.shape)


#########################################

x_covid  = img_data[0:333]
x_no_findings = img_data[333:720]
x_pneumonia = img_data[720:1114]

datagen = ImageDataGenerator(rotation_range = 90, width_shift_range = 0.2, height_shift_range = 0.2,
                            shear_range = 0.2, zoom_range=0.2, horizontal_flip = True, fill_mode='nearest')

print("Generating Data for Covis-19 !")
i = 0
for batch in tqdm(datagen.flow(x_covid, batch_size = 1, save_to_dir = 'preview/Covid-19', save_prefix='covid_19',save_format = '.jpg')):
    i += 1
    if i>5000:
        break

print("Generating Data for No-Findings !")
i = 0
for batch in tqdm(datagen.flow(x_no_findings, batch_size = 10, save_to_dir = 'preview/No_findings', save_prefix='no_findings',save_format = '.jpg')):
    i += 1
    if i >5000:
        break

print("Generating Data for Pneumonia !")
i = 0
for batch in tqdm(datagen.flow(x_pneumonia, batch_size = 10, save_to_dir = 'preview/Pneumonia', save_prefix='pneumonia',save_format = '.jpg')):
    i += 1
    if i >5000:
        break
#########################################
