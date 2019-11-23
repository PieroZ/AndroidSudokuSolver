import cv2 as cv
import numpy as np
import math
import param_config
import logging

MAX_WIDTH_HEIGHT = 800


def set_max_dimensions(img):
    width = img.shape[1]
    height = img.shape[0]
    dim = (min(MAX_WIDTH_HEIGHT, width), min(MAX_WIDTH_HEIGHT, height))
    img = cv.resize(img, dim)
    return img


def preprocess_image(src):
    logger = logging.getLogger(param_config.SudokuConfig().Config.get('Globals', 'AppLogName'))
    src = set_max_dimensions(src)
    # edges	=	cv.Canny(	image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]	)
    blurred = apply_gaussian_blur(src)
    adaptive_thresholded = apply_adaptive_threshold(blurred)
    highlighted_borders = apply_bitwise_not(adaptive_thresholded)
    repaired_disconnected_parts = repair_disconnected_parts(highlighted_borders)
    if logger.level <= logging.INFO:
        cv.imshow('blurred', blurred)
        cv.imshow('adaptive thresholded', adaptive_thresholded)
    if logger.level <= logging.DEBUG:
        cv.imshow('highlighted_borders', highlighted_borders)
    cv.imshow('repaired_disconnected_parts', repaired_disconnected_parts)

    canny_output = cv.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(canny_output, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    hough_lines(canny_output, cdst, cdstP)

    return [src, cdst, cdstP]


def apply_gaussian_blur(src):
    return cv.GaussianBlur(src, (param_config.SudokuConfig().Config.getint('Normal', 'GaussianKernelSize'),
                                 param_config.SudokuConfig().Config.getint('Normal', 'GaussianKernelSize')),
                           param_config.SudokuConfig().Config.getint('Normal', 'GaussianSigmaX'))


def apply_adaptive_threshold(src):
    return cv.adaptiveThreshold(src, param_config.SudokuConfig().Config.getint('Normal', 'AdaptiveThresholdMaxValue'),
                                cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
                                param_config.SudokuConfig().Config.getint('Normal', 'AdaptiveThresholdWindow'),
                                param_config.SudokuConfig().Config.getint('Normal', 'AdaptiveSubtractConstant'))


def apply_bitwise_not(src):
    """ Gets blurred and thresholded image and convert its borders to white (along with other noise)

    :param src: Image after gaussian blur and adaptive threshold operations
    :return: image with inverted colors
    """
    return cv.bitwise_not(src)


def repair_disconnected_parts(src):
    """ Thresholding operation can disconnect connected pairs like lines.
    In order to fill up holes threshold might have caused dilate operation is being used.

    :param src:
    :return dilated image:
    """

    # cross shaped kernel DilatationKernelSize
    di_kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # print(di_kernel)
    return cv.dilate(src, di_kernel)


def hough_lines(canny_output, cdst, cdstP):
    lines = cv.HoughLines(canny_output, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    linesP = cv.HoughLinesP(canny_output, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
