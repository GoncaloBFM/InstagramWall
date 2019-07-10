import numpy as np
import tensorflow as tf
import pandas
from tqdm import tqdm

from similar_cnn_utils.CV_IO_utils import read_imgs_dir
from similar_cnn_utils.CV_transform_utils import apply_transformer
from similar_cnn_utils.CV_transform_utils import resize_img, normalize_img
import nmslib

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

TAG = "makeup"
# TAG = "auschwitz"
# TAG = "planewindow"
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
    n_images = pandas.read_hdf(SEEK_RESULT_PATH, key="image_ids", start=-1).iloc[0]["image_ids"] + 1

    index = nmslib.init(method='hnsw', space='cosinesimil')

    progress_bar = tqdm(total=n_images)

    processed = 0
    while n_images - processed > 0:
        batch_image_ids = range(processed, processed + min(n_images - processed, BATCH_SIZE))
        batch_image_names = [IMAGES_PATH + str(image_id) + ".jpg" for image_id in batch_image_ids]

        flatten_embeddings = EmbeddingsGenerator.get_embeddings(batch_image_names)

        index.addDataPointBatch(flatten_embeddings, list(map(int, batch_image_ids)))
        processed += BATCH_SIZE
        progress_bar.update(BATCH_SIZE)

    index.createIndex({'post': 2}, print_progress=True)
    index.saveIndex(OUTPUT_PATH, save_data=True)


class EmbeddingsGenerator:
    loaded_model = None

    @staticmethod
    def get_embeddings(batch_image_names):
        images = read_imgs_dir(batch_image_names, parallel=True)

        if EmbeddingsGenerator.loaded_model is None:
            loaded_model = EmbeddingsGenerator.loaded_model = {"shape_img": images[0].shape}
            loaded_model["model"] = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=loaded_model["image_shape"]),
            loaded_model["shape_img_resize"] = tuple([int(x) for x in loaded_model["model"].input.shape[1:]])
            loaded_model["input_shape_model"] = tuple([int(x) for x in loaded_model["model"].input.shape[1:]])
            loaded_model["output_shape_model"] = tuple([int(x) for x in loaded_model["model"].output.shape[1:]])
            loaded_model["transformer"] = ImageTransformer(loaded_model["shape_img_resize"])
        else:
            loaded_model = EmbeddingsGenerator.loaded_model

        transformed_images = apply_transformer(images, loaded_model["transformer"], parallel=True)

        image_data = np.array(transformed_images).reshape((-1,) + loaded_model["input_shape_model"])
        embeddings = loaded_model["model"].predict(image_data)
        flatten_embeddings = embeddings.reshape((-1, np.prod(loaded_model["output_shape_model"])))
        return flatten_embeddings


if __name__ == '__main__':
    main()
