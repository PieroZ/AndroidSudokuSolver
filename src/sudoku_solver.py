import cv2
import sys
import image_preprocessing as imgpro
import param_config
import logging


def load_config():
    param_config.SudokuConfig()


def setup_logger():
    logger = logging.getLogger(param_config.SudokuConfig().Config.get('Globals', 'AppLogName'))
    logger.setLevel((param_config.SudokuConfig().Config.getint("Logging",
                                                               param_config.SudokuConfig().Config.get('Normal',
                                                                                                      'LogLevel'))))


def app(argv):
    load_config()
    setup_logger()
    default_file = '../resources/grids/2.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded properly
    if img is None:
        print('Error opening image: ' + filename)
        return -1

    print('Original Dimensions:', img.shape)
    [src, dst, dstP] = imgpro.preprocess_image(img)
    # cv2.imshow('Source', src)
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", dst)
    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", dstP)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app(sys.argv[1:])
