import os
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
SIMULTANEOUS = 6
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
    seek_result = list(pandas.read_hdf(SEEK_RESULT_PATH, key='urls', columns=[THUMB_SIZE_CODE])[THUMB_SIZE_CODE])
    with ThreadPoolExecutor(max_workers=SIMULTANEOUS) as executor:
        tmp_files = list(tqdm(executor.map(try_download, seek_result), total=len(seek_result)))
    seek_result = None

    print("Renaming")
    image_id = 0
    result = []
    for tmp_file in tmp_files:
        if tmp_file is None:
            result.append(None)
        else:
            os.rename(tmp_file, OUTPUT_DIRECTORY + str(image_id) + ".jpg")
            result.append(image_id)
            image_id += 1

    result = pandas.DataFrame(list(result), columns=["image_ids"])
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
    temp_file = OUTPUT_DIRECTORY + "tmp" + str(image_id) + ".jpg"
    with open(temp_file, "wb") as f:
        shutil.copyfileobj(response.raw, f)
    return temp_file


if __name__ == '__main__':
    main()
