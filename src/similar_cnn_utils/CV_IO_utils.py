"""

 CV_IO_utils.py  (author: Anson Wong / git: ankonzoid)

"""
import os
import random

import skimage.io
from multiprocessing import Pool

# Read image
def read_img(filePath):
    return skimage.io.imread(filePath, as_gray=False)

# Read images with common extensions from a directory
def read_imgs_dir(file_names, parallel=True):
    if parallel:
        pool = Pool()
        imgs = pool.map(read_img, file_names)
        pool.close()
        pool.join()
    else:
        imgs = [read_img(arg) for arg in file_names]
    return list(filter(lambda x: x is not None, imgs))

# Save image to file
def save_img(filePath, img):
    skimage.io.imsave(filePath, img)