import tensorflow as tf

import numpy as np
import os
import cv2

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

#########################################loading_Model#################################################

model  = tf.keras.models.load_model('model_infection_mask.h5')
model.summary()

##############################Saving the Image in size 512x512 and 128x128 pixels#################################

#output path
output_path = "Outputs/infection_masked_images/"
orig_size_out_path = output_path + "original_size_512x512/"

resized_out_path = output_path + "resized_128x128/"

resized_colored_out = resized_out_path + "colored/"
resized_greyscale_out = resized_out_path + "greyscale/"

#input path
input_images_path_1 = "Dataset_1/Train/ct_scans/"
input_images_path_2 = "Dataset_1/Test/ct_scans/"

ids_path_1 = next(os.walk(input_images_path_1))[1]
ids_path_2 = next(os.walk(input_images_path_2))[1]

paths_input = []

for id in ids_path_1:
    path_1 = input_images_path_1 + id +'/'
    paths_input.append(path_1)

for id in ids_path_2:
    path_2 = input_images_path_2 + id +'/'
    paths_input.append(path_2)

#segmenting and saving the images
seq = 0
for n, path in tqdm(enumerate(paths_input), total=len(paths_input)):
    ids = next(os.walk(path))
    ids = ids[2]
    s_im = len(ids)
    images_128 = np.zeros((s_im, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    images_512 = np.zeros((s_im, 512, 512), dtype=np.uint8)
    count = 0
    for img in ids:
        ip_image = imread(path + img)
        images_512[count] = ip_image
        image = resize(ip_image, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
        images_128[count] = image
        count = count+1

    preds = model.predict(images_128, verbose = 1)

    for i in range(len(preds)):
        #masking the images
        mask = preds[i]
        mask = (mask > 0.5).astype(np.uint8)
        mask_resized = cv2.resize(mask, (512,512), interpolation = cv2.INTER_CUBIC)
        image_seg = images_128[i]
        image_seg_resized = images_512[i]

        rows, cols, channels = image_seg.shape
        rows_resized, cols_resized = image_seg_resized.shape

        image_seg = image_seg[0:rows, 0:cols]
        image_seg_resized = image_seg_resized[0:rows_resized, 0:cols_resized]

        segmented_image = cv2.bitwise_or(image_seg, image_seg, mask = mask)
        segmented_image_resized = cv2.bitwise_or(image_seg_resized, image_seg_resized, mask = mask_resized)

        segmented_image = segmented_image[0:rows, 0:cols]
        segmented_image_resized = segmented_image_resized[0:rows_resized, 0:cols_resized]

        segmented_image_gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

        # saving the images
        place = i+seq
        cv2.imwrite(orig_size_out_path + "{}.png".format(place), segmented_image_resized)
        cv2.imwrite(resized_colored_out+"{}.png".format(place),segmented_image)
        cv2.imwrite(resized_greyscale_out+"{}.png".format(place),segmented_image_gray)

    seq = seq + i
