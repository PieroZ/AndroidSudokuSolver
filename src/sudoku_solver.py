import cv2
import sys
import numpy as np
import image_preprocessing as imgpro


def app(argv):
    default_file = '../resources/grids/1.png'
    filename = argv[0] if len(argv) > 0 else default_file
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded properly
    if img is None:
        print('Error opening image: ' + filename)
        return -1

    print('Original Dimensions:', img.shape)
    img = imgpro.preprocess_image(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app(sys.argv[1:])
