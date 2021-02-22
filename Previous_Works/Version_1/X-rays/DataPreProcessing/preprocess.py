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

Destination_Lung_mask = "Resultant_Masks/Lungs/"
Destination_Clavicle_mask = "Resultant_Masks/Clavicle/"

Path_Clavicle = "Masks/Clavicle/"
Path_Lung = "Masks/Lungs/"

Clavicle_Left = Path_Clavicle + "Left/"
Clavicle_Right = Path_Clavicle + "Right/"

Lung_Left = Path_Lung + "Left/"
Lung_Right = Path_Lung + "Right/"

Clavicle_Left_id = next(os.walk(Clavicle_Left))[2]
Clavicle_Right_id = next(os.walk(Clavicle_Right))[2]

Lung_Left_id = next(os.walk(Lung_Left))[2]
Lung_Right_id = next(os.walk(Lung_Right))[2]

length = len(Lung_Left_id) # All lengths are same

#Lung_masks = np.zeros((length, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
#Clavicle_masks = np.zeros((length, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

for i in tqdm(range(0,length)):
    path_lung_left = Lung_Left + Lung_Left_id[i]
    path_lung_right = Lung_Right + Lung_Right_id[i]
    path_clavicle_left = Clavicle_Left + Clavicle_Left_id[i]
    path_clavicle_right = Clavicle_Right + Clavicle_Right_id[i]

    mask_lung_left = imread(path_lung_left)
    mask_lung_right = imread(path_lung_right)
    mask_clavicle_left = imread(path_clavicle_left)
    mask_clavicle_right = imread(path_clavicle_right)

    mask_lung = np.maximum(mask_lung_left, mask_lung_right)
    mask_clavicle = np.maximum(mask_clavicle_left, mask_clavicle_right)

    cv2.imwrite(Destination_Lung_mask + "Lung_Mask_{}.png".format(i), mask_lung)
    cv2.imwrite(Destination_Clavicle_mask + "Clavicle_Mask_{}.png".format(i), mask_clavicle)
