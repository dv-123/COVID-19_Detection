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

#########################################loading_Model#################################################

model  = tf.keras.models.load_model('model_X_ray_lung_mask.h5')
model.summary()

##############################Saving the Image in size 512x512 and 128x128 pixels#################################

#output path
output_path = "Outputs/"

orig_size_out_path = output_path + "Original_Size/lung_masked_image/"
resized_out_path = output_path + "128x128/lung_masked_image/"

#input path
input_images_path = "Dataset/Images/"

ids_path = next(os.walk(input_images_path))[2]

paths_input = []

for id in ids_path:
    path_1 = input_images_path + id
    paths_input.append(path_1)

s_im = len(paths_input)
images_128 = np.zeros((s_im, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
images_2048 = np.zeros((s_im, 2048, 2048, 3), dtype=np.uint8)
#segmenting and saving the images
print("Preprocessing Images !")
count = 0
for n, path in tqdm(enumerate(paths_input), total=len(paths_input)):
    ip_image = imread(path)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    ip_image = clahe.apply(ip_image)
    ip_image = skimage.color.gray2rgb(ip_image)
    images_2048[count] = ip_image
    image = resize(ip_image, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
    images_128[count] = image
    count = count + 1

print("Using the model to predict masks !")
preds = model.predict(images_128, verbose = 1)

print("Masking the images and svaing ! ")
place = 0
for i in tqdm(range(len(preds))):
    #masking the images
    mask = preds[i]
    mask = (mask > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(mask, (2048,2048), interpolation = cv2.INTER_CUBIC)
    image_seg = images_128[i]
    image_seg_resized = images_2048[i]

    rows, cols, channels = image_seg.shape
    rows_resized, cols_resized, channels_resized = image_seg_resized.shape

    image_seg = image_seg[0:rows, 0:cols]
    image_seg_resized = image_seg_resized[0:rows_resized, 0:cols_resized]

    segmented_image = cv2.bitwise_and(image_seg, image_seg, mask = mask)
    segmented_image_resized = cv2.bitwise_and(image_seg_resized, image_seg_resized, mask = mask_resized)

    segmented_image = segmented_image[0:rows, 0:cols]
    segmented_image_resized = segmented_image_resized[0:rows_resized, 0:cols_resized]

    # saving the images
    cv2.imwrite(orig_size_out_path + "{}.png".format(place), segmented_image_resized)
    cv2.imwrite(resized_out_path+"{}.png".format(place),segmented_image)
    place = place+1
