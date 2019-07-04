import random

import numpy
import pickle

import cv2

TAG = "other"
CROPS_DIR = "../data/crops/{}/".format(TAG)


def main():
    image_name = "flower.jpg"
    assemble(image_name)


class Canvas:
    def __init__(self, width, height):
        self._canvas = 255 * numpy.ones(shape=[height, width, 3], dtype=numpy.uint8)

    def paint(self, x, y, image):
        self._canvas[y: y + image.shape[0], x: x + image.shape[1]] = image * random.random() * 2

    def show(self):
        cv2.imshow("canvas", self._canvas)
        cv2.waitKey(0)


def assemble(image_name):
    crops_dir = "{}{}/".format(CROPS_DIR, image_name)
    cropped_image_details = pickle.load(open(crops_dir + "details.pickle", "rb"))
    image_width, image_height = cropped_image_details["width"], cropped_image_details["height"]
    canvas = Canvas(image_width, image_height)
    canvas.show()

    crop_details = cropped_image_details["crop_details"]
    for crop_detail in crop_details:
        crop = cv2.imread(crops_dir + crop_detail["file_name"])
        x, y = crop_detail["x"], crop_detail["y"]
        canvas.paint(x, y, crop)
        canvas.show()

if __name__ == '__main__':
    main()
