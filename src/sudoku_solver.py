import cv2
import sys
import numpy as np

MAX_WIDTH_HEIGHT = 800


def set_max_dimensions(img):
    width = img.shape[1]
    height = img.shape[0]
    dim = (min(MAX_WIDTH_HEIGHT, width), min(MAX_WIDTH_HEIGHT, height))
    img = cv2.resize(img, dim)
    return img


def app(argv):
    default_file = '../resources/grids/1.png'
    filename = argv[0] if len(argv) > 0 else default_file
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = set_max_dimensions(img)
    cv2.imshow('image', img)
    print('Original Dimensions : ', img.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app(sys.argv[1:])
