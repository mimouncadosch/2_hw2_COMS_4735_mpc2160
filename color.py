import numpy as np
import cv2
from histogram import histogram

"""Computes the similarity metric between two images
idx1	: index of first image
idx2 	: index of second image
b 		: number of bins
"""


def color_similarity(idx1, idx2, b, black_lb):
    # Parse ids to be consistent with image names
    if idx1 < 10: idx1 = "0" + str(idx1)
    if idx2 < 10: idx2 = "0" + str(idx2)

    img1 = cv2.imread("../Images/i" + str(idx1) + ".ppm")
    img2 = cv2.imread("../images/i" + str(idx2) + ".ppm")  # Define how many pixels are considered "black"

    # This custom mask is flexible in defining
    # what range of pixels is considered "black"
    lb = (black_lb, black_lb, black_lb)
    ub = (255, 255, 255)
    mask = cv2.inRange(img1, lb, ub)

    # Creates 3D histograms
    hist1 = cv2.calcHist([img1], [0, 1, 2], mask, [b, b, b], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], mask, [b, b, b], [0, 256, 0, 256, 0, 256])

    pixels = 89 * 60

    # L1 distance
    D = abs(hist2 - hist1)
    d_l1 = np.sum(D) / (2 * pixels)

    # Similarity: 1 - distance
    s = 1 - d_l1

    return s