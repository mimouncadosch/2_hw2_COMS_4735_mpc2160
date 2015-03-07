# coding=utf-8
import numpy as np
import cv2

from matplotlib import pyplot as plt

def texture_similarity(idx1, idx2, b, b_lb):
    # Parse ids to be consistent with image names
    if idx1 < 10: idx1 = "0" + str(idx1)
    if idx2 < 10: idx2 = "0" + str(idx2)

    img1 = cv2.imread("../Images/i" + str(idx1) + ".ppm", 0)
    img2 = cv2.imread("../Images/i" + str(idx2) + ".ppm", 0)

    kernel = np.matrix([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    l1 = cv2.filter2D(img1, cv2.CV_32F, kernel)
    l2 = cv2.filter2D(img2, cv2.CV_32F, kernel)

    masked_l1 = l1[img1 > b_lb]
    masked_l2 = l2[img2 > b_lb]

    h1 = create_histogram(masked_l1, b, idx1)
    h2 = create_histogram(masked_l2, b, idx2)

    pixels = 89 * 60

    # L1 distance
    d = abs(h2 - h1)    # Difference between two histograms
    d_l1 = float(np.sum(d)) / (2 * pixels)

    # Similarity: 1 - distance
    s = 1 - d_l1
    return s

    return True

def create_histogram(laplacian, n_bins, idx):
    """Creates histogram from Laplacian image
    """
    hist, bins = np.histogram(laplacian, bins=n_bins)
    # width = (bins[1] - bins[0])
    # center = (bins[:-1] + bins[1:]) / 2

    # plt.bar(center, hist, align='center', width=width, color='m')
    # plt.savefig("text_bkg/spec_10:" + str(idx) + ".png")
    # plt.show()

    return hist

# if __name__ == "__main__":
    # tot_similarities = 0
    #
    # text = [1,3,4,5,6,7,8,9,10,11, 12]
    #
    # smooth = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    # for i in xrange(0, len(text)):
    #     for j in xrange(0, len(smooth)):
    #         tot_similarities += texture_similarity(text[i], smooth[j], 100, 0)
    # texture_similarity(1, 18, 100, 5)

    # print tot_similarities