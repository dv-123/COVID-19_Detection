import scipy, shutil, os, nibabel
import sys, getopt
import imageio
import numpy as np
import matplotlib.pyplot as plt

inputfile = 'F:\\Major Project\\Datasets\\covid19-ct-scans\\lung_mask\\radiopaedia_40_86625_0.nii'
outputfile = 'F:\\Major Project\\preperations\\working with nii image format\\dataset\\lung_mask\\output_20_corona\\'

if not os.path.exists(outputfile):
    os.makedirs(outputfile)
    print("Created ouput directory: " + outputfile)

print('Reading NIfTI file...')

image_array = nibabel.load(inputfile)
data = image_array.get_fdata()
data = data.T
data = np.array(data)

total_slices = data.shape[0]
slice_counter = 0
for current_slice in range(0, total_slices):

    if (slice_counter % 1) == 0:
        print('Saving image...')
        image_name = inputfile[:-4] + "_z" + "{:0>3}".format(str(current_slice+1))+ ".png"
        #imageio.imwrite(image_name, np.flipud(data[current_slice]))
        imageio.imwrite(image_name, np.fliplr(data[current_slice]))
        print('Saved.')

        print('Moving image...')
        src = image_name
        shutil.move(src, outputfile)
        slice_counter += 1
        print('Moved.')

print('Finished converting images')
