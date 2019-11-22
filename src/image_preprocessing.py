import cv2

MAX_WIDTH_HEIGHT = 800


def set_max_dimensions(img):
    width = img.shape[1]
    height = img.shape[0]
    dim = (min(MAX_WIDTH_HEIGHT, width), min(MAX_WIDTH_HEIGHT, height))
    img = cv2.resize(img, dim)
    return img


def preprocess_image(img):
    img = set_max_dimensions(img)

    return img

