import json
import random

import numpy
import pickle

import cv2

from utils.image import cv2_read_image


class Canvas:
    def __init__(self, width, height):
        self._canvas = 255 * numpy.ones(shape=[height, width, 3], dtype=numpy.uint8)

    def paint(self, x, y, image):
        self._canvas[y: y + image.shape[0], x: x + image.shape[1]] = image

    def show(self):
        cv2.imshow("canvas", self._canvas)
        cv2.waitKey(0)

    def save(self):
        cv2.imwrite("../data/scraps/x.jpg", self._canvas)


def assemble(crop_details):
    image_width, image_height = crop_details[0]["source"]["width"], crop_details[0]["source"]["height"]
    canvas = Canvas(image_width, image_height)

    for crop_detail in crop_details:
        crop_path = crop_detail["source"]["crop_dir"] + crop_detail["source"]["crop_name"]
        crop = cv2_read_image(crop_path)
        if crop is None:
            raise FileNotFoundError("Crop not found: {}".format(crop_path))
        x, y = crop_detail["x"], crop_detail["y"]
        canvas.paint(x, y, crop)
    canvas.save()


def main():
    pass


if __name__ == '__main__':
    main()
