import numpy as np
import os

from tqdm import tqdm
import cv2

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt


IMG_WIDTH = 1024
IMG_HEIGHT = 1024
IMG_CHANNELS = 3

path_images = "Images/"
path_lung_masks = "Resultant_Masks/Lungs/"
path_clavicle_masks = "Resultant_Masks/Clavicle/"
path_heart_masks = "Resultant_Masks/Heart/"


images_id = next(os.walk(path_images))[2]
lung_mask_id = next(os.walk(path_lung_masks))[2]
clavicle_mask_id = next(os.walk(path_clavicle_masks))[2]
heart_mask_id = next(os.walk(path_heart_masks))[2]

destination_images = "Output/Images/"
destination_lung_mask = "Output/Lung_mask/"
destination_heart_mask = "Output/Heart_mask/"
destination_clavicle_mask = "Output/Clavicle_mask/"

length = len(images_id) #all len is same

for i in tqdm(range(0,length)):
    img_image = imread(path_images + images_id[i])
    img_lung_mask = imread(path_lung_masks + lung_mask_id[i])
    img_heart_mask = imread(path_heart_masks + heart_mask_id[i])
    img_clavicle_mask = imread(path_clavicle_masks + clavicle_mask_id[i])

    image_1 = img_image
    image_2 = cv2.flip(img_image, 1)
    image_3 = cv2.flip(img_image, 0)
    image_4 = cv2.flip(image_2, 0)

    lung_mask_1 = img_lung_mask
    lung_mask_2 = cv2.flip(img_lung_mask, 1)
    lung_mask_3 = cv2.flip(img_lung_mask, 0)
    lung_mask_4 = cv2.flip(lung_mask_2, 0)

    heart_mask_1 = img_heart_mask
    heart_mask_2 = cv2.flip(img_heart_mask, 1)
    heart_mask_3 = cv2.flip(img_heart_mask, 0)
    heart_mask_4 = cv2.flip(heart_mask_2, 0)

    clavicle_mask_1 = img_clavicle_mask
    clavicle_mask_2 = cv2.flip(img_clavicle_mask, 1)
    clavicle_mask_3 = cv2.flip(img_clavicle_mask, 0)
    clavicle_mask_4 = cv2.flip(clavicle_mask_2, 0)

    cv2.imwrite(destination_images + "image_{}_1.png".format(i), image_1)
    cv2.imwrite(destination_images + "image_{}_2.png".format(i), image_2)
    cv2.imwrite(destination_images + "image_{}_3.png".format(i), image_3)
    cv2.imwrite(destination_images + "image_{}_4.png".format(i), image_4)

    cv2.imwrite(destination_lung_mask + "lung_mask_{}_1.png".format(i), lung_mask_1)
    cv2.imwrite(destination_lung_mask + "lung_mask_{}_2.png".format(i), lung_mask_2)
    cv2.imwrite(destination_lung_mask + "lung_mask_{}_3.png".format(i), lung_mask_3)
    cv2.imwrite(destination_lung_mask + "lung_mask_{}_4.png".format(i), lung_mask_4)

    cv2.imwrite(destination_heart_mask + "heart_mask_{}_1.png".format(i), heart_mask_1)
    cv2.imwrite(destination_heart_mask + "heart_mask_{}_2.png".format(i), heart_mask_2)
    cv2.imwrite(destination_heart_mask + "heart_mask_{}_3.png".format(i), heart_mask_3)
    cv2.imwrite(destination_heart_mask + "heart_mask_{}_4.png".format(i), heart_mask_4)

    cv2.imwrite(destination_clavicle_mask + "clavicle_mask_{}_1.png".format(i), clavicle_mask_1)
    cv2.imwrite(destination_clavicle_mask + "clavicle_mask_{}_2.png".format(i), clavicle_mask_2)
    cv2.imwrite(destination_clavicle_mask + "clavicle_mask_{}_3.png".format(i), clavicle_mask_3)
    cv2.imwrite(destination_clavicle_mask + "clavicle_mask_{}_4.png".format(i), clavicle_mask_4)
