import cv2 as cv
import numpy as np
import math

MAX_WIDTH_HEIGHT = 800


def set_max_dimensions(img):
    width = img.shape[1]
    height = img.shape[0]
    dim = (min(MAX_WIDTH_HEIGHT, width), min(MAX_WIDTH_HEIGHT, height))
    img = cv.resize(img, dim)
    return img


def preprocess_image(src):
    src = set_max_dimensions(src)
    # edges	=	cv.Canny(	image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]	)
    canny_output = cv.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(canny_output, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    hough_lines(canny_output, cdst, cdstP)

    return [src, cdst, cdstP]


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

