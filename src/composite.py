import json
import os
import pprint
import random

from tqdm import tqdm

from cropping import RegularCropper
from src import assembly
from src import cropping

COMPOSITE_DIR = "../data/composite/"
PROJECT_DIR = "eye/"
TARGET_FILE = "17.jpg"


def main():
    cropper = RegularCropper(6, 6)
    project_dir = COMPOSITE_DIR + PROJECT_DIR
    generate_composite(project_dir, cropper)


def generate_composite(project_dir, cropper):
    originals_dir = "{}originals/".format(project_dir)
    crops_dir = "{}crops/".format(project_dir)
    target_crops_dir = cropper.do_cropping(originals_dir, TARGET_FILE, crops_dir)
    originals_crops_dir = []
    print("Generating crops")
    for image in tqdm(os.listdir(originals_dir)):
        originals_crops_dir.append(cropper.do_cropping(originals_dir, image, crops_dir))
    print("Finished generating crops")
    print()

    print("Creating composite from {}".format(random_pick_crops.__name__))
    picked_crops = random_pick_crops(target_crops_dir, originals_crops_dir)
    print("Finished creating composite")
    print()

    assembly.assemble(picked_crops)


def random_pick_crops(target_crops_dir, originals_crops_dir):
    target_crops = json.load(open(target_crops_dir + "details.json", "r"))
    final_width, final_height = target_crops[0]["source"]["width"], target_crops[0]["source"]["height"]
    picked_crops = []
    for index, target_crop in tqdm(enumerate(target_crops), total=len(target_crops)):
        picked_crop_dir = originals_crops_dir[random.randint(0, len(originals_crops_dir) - 1)]
        picked_crop = json.load(open(picked_crop_dir + "details.json", "r"))[index]
        original_width, original_height = (picked_crop["source"]["width"], picked_crop["source"]["height"])
        if (original_width, original_height) != (final_width, final_height):
            raise Exception(
                "Crops in {} have wrong size, expected  {}, {}, found {}, {}".
                    format(picked_crop_dir, final_width, final_height, original_width, original_height))
        picked_crops.append(picked_crop)
    return picked_crops


def similar_pick_crops(target_crops_dir, originals_crops_dir):
    pass


if __name__ == '__main__':
    main()
