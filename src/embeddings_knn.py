import nmslib
import pandas
import heapq

from scipy import spatial

from similar_cnn_utils.CV_IO_utils import read_imgs_dir, read_img
from similar_cnn_utils.CV_plot_utils import plot_query_retrieval

TAG = "makeup"
# TAG = "auschwitz"
# TAG = "planewindow"
# TAG = "sample"

IMAGES_PATH = "../data/thumbs/{}/".format(TAG)
EMBEDDINGS_PATH = "../data/embeddings/{}.index".format(TAG)
K_DIR = "../results/k_1/{}/".format(TAG)
K10_DIR = "../results/k_all/{}/".format(TAG)
BRUTE_DIR = "../results/k_brute/{}/".format(TAG)
SEEK_RESULT = pandas.read_hdf("../data/seek_result/{}.h5".format(TAG))
SEEK_RESULT_PATH = "../data/seek_result/{}.h5".format(TAG)


def main():
    # knn()
    # knn_for_all()
    brute_force_neighbors()


def knn_for_all():
    k = 20

    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.loadIndex(EMBEDDINGS_PATH, load_data=True)
    image_ids = list(SEEK_RESULT["image_ids"])
    image_index = 0
    for image_id in image_ids:
        if image_id == -1:
            continue
        embeddings = index[image_index]
        neighbors, distances = index.knnQuery(embeddings, k=k)
        result = read_imgs_dir([IMAGES_PATH + str(neighbor_id) + ".jpg" for neighbor_id in neighbors], parallel=True)
        plot_query_retrieval(image_id, neighbors, read_img(IMAGES_PATH + str(image_id) + ".jpg"), result, K10_DIR + str(image_id) + ".jpg")
        print("{} of {} ({}%)".format(image_index + 1, len(image_ids), ((image_index + 1) * 100) / len(image_ids)))
        image_index += 1


def knn():
    target_id = "173"
    k = 200

    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.loadIndex(EMBEDDINGS_PATH, load_data=True)
    target_index = list(SEEK_RESULT[SEEK_RESULT["image_id"].notnull()]["image_id"]).index(target_id)
    image_ids = list(SEEK_RESULT["image_id"])
    embeddings = index[target_index]
    neighbors, distances = index.knnQuery(embeddings, k=k)
    result = read_imgs_dir([IMAGES_PATH + image_ids[neighbor_id] + ".jpg" for neighbor_id in neighbors], parallel=True)
    plot_query_retrieval(target_id, neighbors, read_img(IMAGES_PATH + target_id + ".jpg"), result, K_DIR + target_id + ".jpg")


def brute_force_neighbors():
    k = 100

    target_id = "173"
    n_images = pandas.read_hdf(SEEK_RESULT_PATH, key="image_ids", start=-1).iloc[0]["image_ids"] + 1
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.loadIndex(EMBEDDINGS_PATH, load_data=True)

    heap = []

    target_embeddings = index[int(target_id)]
    for i in range(1000): #n_images and parallel
        image_id = str(i)
        if image_id == target_id:
            continue
        other_embeddings = index[i]

        distance = spatial.distance.cosine(target_embeddings, other_embeddings)
        heapq.heappush(heap, (-distance, image_id))
        if len(heap) > k:
            heapq.heappop()
    neighbors_ids = map(lambda x: x[1], sorted(list(map(lambda x: (-x[0], x[1]), heap))))

    result = read_imgs_dir([IMAGES_PATH + neighbor_id + ".jpg" for neighbor_id in neighbors_ids], parallel=True)
    plot_query_retrieval(target_id, neighbors_ids, read_img(IMAGES_PATH + target_id + ".jpg"), result, K_DIR + target_id + ".jpg")




if __name__ == '__main__':
    main()
