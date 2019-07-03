import nmslib
import pandas

from similar_cnn_utils.CV_IO_utils import read_imgs_dir, read_img
from similar_cnn_utils.CV_plot_utils import plot_query_retrieval

TAG = "makeup"
# TAG = "auschwitz"
# TAG = "planewindow"
# TAG = "sample"

IMAGES_PATH = "../data/thumbs/{}/".format(TAG)
EMBEDDINGS_PATH = "../data/embeddings/{}.index".format(TAG)
K_DIR = "../results/k/{}/".format(TAG)
K10_DIR = "../results/k10/{}/".format(TAG)
SEEK_RESULT = pandas.read_pickle("../data/seek_result/{}.pickle".format(TAG))


def main():
    # knn()
    knn_10_for_all()


def knn_10_for_all():
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.loadIndex(EMBEDDINGS_PATH, load_data=True)
    image_ids = list(SEEK_RESULT["image_id"])
    image_index = 0
    for image_id in image_ids:
        if image_id == -1:
            continue
        embeddings = index[image_index]
        neighbors, distances = index.knnQuery(embeddings, k=20)
        if 1 < sum(distances) < 2:
            result = read_imgs_dir([IMAGES_PATH + image_ids[neighbor_id] + ".jpg" for neighbor_id in neighbors], parallel=True)
            plot_query_retrieval(image_id, neighbors, read_img(IMAGES_PATH + image_id + ".jpg"), result, K10_DIR + image_id + ".jpg")
            print("{} of {} ({}%)".format(image_index + 1, len(image_ids), ((image_index + 1) * 100) / len(image_ids)))
        image_index += 1


def knn():
    target_id = "173"
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.loadIndex(EMBEDDINGS_PATH, load_data=True)
    target_index = list(SEEK_RESULT[SEEK_RESULT["image_id"].notnull()]["image_id"]).index(target_id)
    image_ids = list(SEEK_RESULT["image_id"])
    embeddings = index[target_index]
    neighbors, distances = index.knnQuery(embeddings, k=200)
    result = read_imgs_dir([IMAGES_PATH + image_ids[neighbor_id] + ".jpg" for neighbor_id in neighbors], parallel=True)
    plot_query_retrieval(target_id, neighbors, read_img(IMAGES_PATH + target_id + ".jpg"), result, K_DIR + target_id + ".jpg")


if __name__ == '__main__':
    main()
