import json
import math
import os
import pickle
import shutil

import cv2
from abc import ABC, abstractmethod

from utils.image import cv2_read_image


def main():
    images_dir = "../data/scraps/other/"
    crops_dir = "../data/scraps/"
    image_name = "portrait.jpg"
    RegularCropper(4, 4).do_cropping(images_dir, image_name, crops_dir)
    # SquareCropper(4).do_cropping(IMAGES_DIR, image_name, CROPS_DIR)


class GenericCropper(ABC):
    def do_cropping(self, image_dir, image_name, output_dir):
        image = cv2_read_image(image_dir + image_name)
        crops = self.crop_function(image)

        if output_dir is None:
            for x, y, width, height, crop in crops:
                cv2.imshow("x: {}, y: {}".format(x, y), crop)
                cv2.waitKey(0)
            return

        crop_details = []
        crops_dir = "{}{}{}/".format(output_dir, image_name, ".crops")
        if os.path.isdir(crops_dir):
            shutil.rmtree(crops_dir)
        os.mkdir(crops_dir)
        for crop_id, (x, y, width, height, crop) in enumerate(crops):
            crop_file = "{}.jpg".format(crop_id)
            crop_details.append({"x": x,
                                 "y": y,
                                 "width": width,
                                 "height": height,
                                 "source":
                                     {
                                         "width": image.shape[1],
                                         "height": image.shape[0],
                                         "image_dir": image_dir,
                                         "image_name": image_name,
                                         "crop_dir": crops_dir,
                                         "crop_name": crop_file
                                     }
                                 })

            cv2.imwrite(crops_dir + crop_file, crop)

        with open(crops_dir + "details.json", "w") as f:
            json.dump(crop_details, f, indent=4)
        return crops_dir

    @abstractmethod
    def crop_function(self, image):
        pass


class RegularCropper(GenericCropper):
    def __init__(self, horizontal, vertical):
        super().__init__()
        self._horizontal = horizontal
        self._vertical = vertical

    def crop_function(self, image):
        image_width = image.shape[1]
        image_height = image.shape[0]
        crop_width = math.ceil(image_width / self._horizontal)
        crop_height = math.ceil(image_height / self._vertical)
        crops = []
        for y in range(0, image_height, crop_height):
            for x in range(0, image_width, crop_width):
                crop = image[y:y + crop_height, x:x + crop_width]
                crops.append((x, y, crop_height, crop_width, crop))
        return crops


class SquareCropper(RegularCropper):
    def __init__(self, n_squares):
        super().__init__(0, 0)
        self._n_squares = n_squares

    def crop_function(self, image):
        image_width = image.shape[1]
        image_height = image.shape[0]

        self._horizontal = image_width / self._n_squares
        self._vertical = image_height / self._n_squares
        return super().crop_function(image)


if __name__ == '__main__':
    main()
