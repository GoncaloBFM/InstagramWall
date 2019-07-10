import cv2


def cv2_read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found: {}".format(image_path))
    return image
