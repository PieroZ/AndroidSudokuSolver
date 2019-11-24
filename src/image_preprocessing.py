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
    # src = set_max_dimensions(src)
    # edges	=	cv.Canny(	image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]	)
    blurred = apply_gaussian_blur(src)
    adaptive_thresholded = apply_adaptive_threshold(blurred)
    highlighted_borders = apply_bitwise_not(adaptive_thresholded)
    repaired_disconnected_parts = repair_disconnected_parts(highlighted_borders)
    if logger.level <= logging.INFO:
        cv.imshow('repaired_disconnected_parts', repaired_disconnected_parts)
    outer_box = find_biggest_blob(repaired_disconnected_parts)
    outer_box = erode_outer_grid_lines(outer_box)
    find_lines_in_outer_box(outer_box)

    if logger.level <= logging.INFO:
        cv.imshow('blurred', blurred)
        cv.imshow('adaptive thresholded', adaptive_thresholded)
    if logger.level <= logging.DEBUG:
        cv.imshow('highlighted_borders', highlighted_borders)
    cv.imshow('outer_box', outer_box)

    canny_output = cv.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(canny_output, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    # hough_lines(canny_output, cdst, cdstP)

    return [src, cdst, cdstP]


def find_lines_in_outer_box(src):
    lines = cv.HoughLines(src, 1, np.pi / 180,
                          param_config.SudokuConfig().Config.getint('Normal', 'HoughLinesThreshold'))
    draw_lines(src, lines)


def draw_lines(src, lines):
    """ Draw lines using polar system representation, as described in:
    https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html

    :param src:
    :param lines:
    :return:
    """
    if lines is not None:
        for i in range(0, len(lines)):
            #if lines[i][0][1] is not 0.0:
            if not math.isclose(lines[i][0][1], 0.0, abs_tol=0.00001):
                m = -1 / np.tan(lines[i][0][1])
                c = lines[i][0][0] / np.sin(lines[i][0][1])

                cv.line(src, (0, c), (src.shape[0], int(m * src.shape[0] + c)), (127, 127, 127))
            else:
                cv.line(src, (lines[i][0][0], 0), (lines[i][0][0], int(src.shape[1])), (0, 0, 255))


            # rho = lines[i][0][0]
            # theta = lines[i][0][1]
            #
            # print(m, c, rho, theta)

    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    #         cv.line(src, pt1, pt2, (0, 0, 255))


def find_biggest_blob(src):
    """ Biggest thing in the image is assumed to be the puzzle. Hence the biggest blob should be the puzzle

    :param src:
    :return:
    """
    # return src
    return find_biggest_bounding_box(src)


def find_biggest_bounding_box(src):
    max = -1
    max_pt = None
    # grab the image dimensions
    # h = src.shape[0]
    # w = src.shape[1]
    h, w = src.shape[:2]

    mask = np.zeros((h + 2, w + 2), np.uint8)

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            if src[y, x] >= 128:
                # area = cv.floodFill(src, src(y, x), (0, 0, 64))
                retval = cv.floodFill(src, mask, (x, y), 64)
                area = retval[0]
                if area > max:
                    max_pt = (x, y)
                    max = area

    mask = np.zeros((h + 2, w + 2), np.uint8)
    retval = cv.floodFill(src, mask, max_pt, 255)
    mask = np.zeros((h + 2, w + 2), np.uint8)

    for y in range(0, h):
        for x in range(0, w):
            if src[y, x] == 64 and x is not max_pt[0] and y is not max_pt[1]:
                retval = cv.floodFill(src, mask, (x, y), 0)

    return retval[1]


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


def erode_outer_grid_lines(src):
    er_kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    return cv.erode(src, er_kernel)


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
