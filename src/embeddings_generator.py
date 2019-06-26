import os
import random
from uuid import UUID

import numpy as np
import pandas
import tables
import tensorflow as tf
from tqdm import tqdm

from src.similar_cnn_utils.CV_IO_utils import read_imgs_dir
from src.similar_cnn_utils.CV_transform_utils import apply_transformer
from src.similar_cnn_utils.CV_transform_utils import resize_img, normalize_img
import nmslib
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import manifold

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# TAG = "makeup "
# TAG = "auschwitz"
TAG = "planewindow"
# TAG = "sample"
SEEK_RESULT_PATH = "../data/seek_result/{}.h5".format(TAG)
IMAGES_PATH = "../data/thumbs/{}/".format(TAG)
OUTPUT_PATH = "../data/embeddings/{}.index".format(TAG)

BATCH_SIZE = 10

class ImageTransformer(object):
    def __init__(self, shape_resize):
        self.shape_resize = shape_resize

    def __call__(self, img):
        img_transformed = resize_img(img, self.shape_resize)
        img_transformed = normalize_img(img_transformed)
        return img_transformed

def main():
    seek_result = tables.open_file(SEEK_RESULT_PATH, mode="r")
    n_images = seek_result.root["image_ids"].table.shape[0]
    seek_result.close()
    processed = 0

    index = nmslib.init(method='hnsw', space='cosinesimil')
    first = True

    progress_bar = tqdm(total=n_images)

    while n_images - processed > 0:
        batch_image_ids = range(processed, processed + min(n_images - processed, BATCH_SIZE))
        batch_image_names = [IMAGES_PATH + str(image_id) + ".jpg" for image_id in batch_image_ids]
        images = read_imgs_dir(batch_image_names, parallel=True)

        if first:
            first = False
            shape_img = images[0].shape
            model = tf.keras.applications.VGG19(weights='imagenet', include_top=False,
                                                input_shape=shape_img)
            shape_img_resize = tuple([int(x) for x in model.input.shape[1:]])
            input_shape_model = tuple([int(x) for x in model.input.shape[1:]])
            output_shape_model = tuple([int(x) for x in model.output.shape[1:]])
            transformer = ImageTransformer(shape_img_resize)

        transformed_images = apply_transformer(images, transformer, parallel=True)

        image_data = np.array(transformed_images).reshape((-1,) + input_shape_model)
        embeddings = model.predict(image_data)
        flatten_embeddings = embeddings.reshape((-1, np.prod(output_shape_model)))

        index.addDataPointBatch(flatten_embeddings, list(map(int, batch_image_ids)))
        processed += BATCH_SIZE
        progress_bar.update(BATCH_SIZE)

    index.createIndex({'post': 2}, print_progress=True)
    index.saveIndex(OUTPUT_PATH, save_data=True)


if __name__ == '__main__':
    main()