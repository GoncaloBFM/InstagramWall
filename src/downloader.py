import shutil
import threading
from concurrent.futures.thread import ThreadPoolExecutor
from multiprocessing import Pool

import pandas
import requests
import socket

from requests import HTTPError, ConnectionError
from tqdm import tqdm
from utils.url_parsing import InstagramURLParser

TAG = "makeup"
# TAG = "auschwitz"
# TAG = "planewindow"
# TAG = "sample"
SIMULTANEOUS = 2
IMAGE_SIZE = 2  # 1 - 5

OUTPUT_DIRECTORY = "../data/thumbs/{}/".format(TAG)
SEEK_RESULT_PATH = "../data/seek_result/{}.h5".format(TAG)
THUMB_SIZE_CODE = "thumb{}".format(IMAGE_SIZE)

socket.setdefaulttimeout(10)


class IdGenerator:
    def __init__(self):
        self.current = 0
        self.lock = threading.Lock()

    def generate(self):
        with self.lock:
            result = self.current
            self.current += 1
        return result


id_generator = IdGenerator()


def main():
    pool = Pool(SIMULTANEOUS)
    seek_result = list(pandas.read_hdf(SEEK_RESULT_PATH, key='urls', columns=[THUMB_SIZE_CODE])[THUMB_SIZE_CODE])
    executor = ThreadPoolExecutor(max_workers=2)
    result = pandas.DataFrame(list(tqdm(executor.map(try_download, seek_result), total=len(seek_result))), columns=["image_ids"])
    pool.close()
    result.to_hdf(SEEK_RESULT_PATH, key='image_ids', format="t", data_columns=True)


def try_download(url):
    url = InstagramURLParser.reconstruct_thumb_url(url)

    try:
        return download(url)
    except socket.timeout:
        print("Timeout")
        return None
    except HTTPError:
        print("Bad HTTP status")
        return None
    except ConnectionError:
        print("Connection error")
        return None


def download(link):
    response = requests.get(link, stream=True)
    if response.status_code != 200:
        raise HTTPError
    image_id = id_generator.generate()
    with open(OUTPUT_DIRECTORY + str(image_id) + ".jpg", "wb") as f:
        shutil.copyfileobj(response.raw, f)
    return image_id


if __name__ == '__main__':
    main()
