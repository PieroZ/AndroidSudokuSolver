import cv2 as cv

MAX_WIDTH_HEIGHT = 800


def set_max_dimensions(img):
    width = img.shape[1]
    height = img.shape[0]
    dim = (min(MAX_WIDTH_HEIGHT, width), min(MAX_WIDTH_HEIGHT, height))
    img = cv.resize(img, dim)
    return img


def preprocess_image(src):
    src = set_max_dimensions(src)
    dst = cv.Canny(src, 50, 200, None, 3)
    return dst

