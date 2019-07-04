import sys

import numpy
import os
import shutil
import threading
from concurrent.futures.thread import ThreadPoolExecutor
from multiprocessing import Pool

import pandas
import requests
import socket

import skimage
from requests import HTTPError, ConnectionError
from tqdm import tqdm
from utils.url_parsing import InstagramURLParser

TAG = "makeup"
# TAG = "auschwitz"
# TAG = "planewindow"
# TAG = "sample"
SIMULTANEOUS = 1
IMAGE_SIZE = 2  # 1 - 5

OUTPUT_DIRECTORY = "../data/thumbs/{}/".format(TAG)
SEEK_RESULT_PATH = "../data/seek_result/{}.h5".format(TAG)
THUMB_SIZE_CODE = "thumb{}".format(IMAGE_SIZE)

socket.setdefaulttimeout(10)


class IdGenerator:
    def __init__(self):
        self.__current = 0
        self.__lock = threading.Lock()

    def generate(self):
        with self.__lock:
            result = self.__current
            self.__current += 1
        return result


id_generator = IdGenerator()


def log_decorator(func):
    def wrapper(*args, **kwargs):
        print("Executing {}".format(func.__name__))
        func(*args, **kwargs)
        print("Finished {}".format(func.__name__))
        print()

    return wrapper


@log_decorator
def download_dataset():
    seek_result = list(pandas.read_hdf(SEEK_RESULT_PATH, key='urls', columns=[THUMB_SIZE_CODE])[THUMB_SIZE_CODE])
    with ThreadPoolExecutor(max_workers=SIMULTANEOUS) as executor:
        tmp_files = list(tqdm(executor.map(try_download, seek_result), total=len(seek_result)))
    seek_result = None
    result = pandas.DataFrame({"image_ids": pandas.Series(tmp_files, dtype="int16")})
    result.to_hdf(SEEK_RESULT_PATH, key='image_ids', format="t", data_columns=True)


@log_decorator
def mark_all_dirty():
    for file_name in tqdm(os.listdir(OUTPUT_DIRECTORY)):
        old_file = OUTPUT_DIRECTORY + file_name
        os.rename(old_file, old_file + ".dirty")


@log_decorator
def delete_dataset():
    for file_name in tqdm(os.listdir(OUTPUT_DIRECTORY)):
        old_file = OUTPUT_DIRECTORY + file_name
        os.remove(old_file)


@log_decorator
def clean_dataset(image_ids=None):
    if image_ids is None:
        image_ids = list(pandas.read_hdf(SEEK_RESULT_PATH, key='image_ids')['image_ids'])

    new_id = 0
    for index, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        if image_id == -1:
            continue
        old_file = OUTPUT_DIRECTORY + str(image_id) + ".jpg.dirty"
        try:
            skimage.io.imread(old_file, as_gray=False)
        except ValueError as ex:
            os.remove(old_file)
            image_ids[index] = -1
            print("Faulty file: {}".format(old_file))
            print()
        except FileNotFoundError as ex:
            image_ids[index] = -1
            print("File not found: {}".format(old_file))
            print()
        else:
            os.rename(old_file, OUTPUT_DIRECTORY + str(new_id) + ".jpg")
            image_ids[index] = new_id
            new_id += 1
    result = pandas.DataFrame({"image_ids": pandas.Series(image_ids, dtype="int16")})
    result.to_hdf(SEEK_RESULT_PATH, key='image_ids', format="t", data_columns=True)


def try_download(url):
    url = InstagramURLParser.reconstruct_thumb_url(url)
    try:
        return download(url)
    except socket.timeout:
        print("Timeout: {}".format(url))
        return -1
    except HTTPError:
        print("Bad HTTP status: {}".format(url))
        return -1
    except ConnectionError:
        print("Connection error: {}".format(url))
        return -1


def download(link):
    response = requests.get(link, stream=True)
    if response.status_code != 200:
        raise HTTPError
    image_id = id_generator.generate()
    temp_file = OUTPUT_DIRECTORY + str(image_id) + ".jpg.dirty"
    with open(temp_file, "wb") as f:
        shutil.copyfileobj(response.raw, f)
    return image_id


def main():
    if len(sys.argv) != 2:
        print("Error: wrong number of arguments. Use 'download' or 'd', 'cleanup' or 'c' and 'remove' or 'r'.")
        return
    if sys.argv[1] == 'd' or sys.argv[1] == 'download':
        download_dataset()
        clean_dataset()
        return
    if sys.argv[1] == 'c' or sys.argv[1] == 'cleanup':
        clean_dataset(mark_all_dirty())
        return
    if sys.argv[1] == 'r' or sys.argv[1] == 'remove':
        delete_dataset()
        return
    print("Wrong argument. Use 'download' or 'd', 'cleanup' or 'c' and 'remove' or 'r'.")


if __name__ == '__main__':
    main()
