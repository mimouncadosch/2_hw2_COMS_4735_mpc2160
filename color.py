import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from histogram import histogram
import timeit


"""Computes the similarity metric between two images
idx1	: index of first image
idx2 	: index of second image
b 		: number of bins
b_lb    : the lower bound of what is considered a "black" pixel
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


if __name__ == "__main__":
    bins_sims = []
    blb_sims = []
    runtimes = []
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # x =[1,2,3,4,5,6,7,8,9,10]
    # y =[5,6,2,3,13,4,1,2,4,8]
    # z =[2,3,3,3,5,7,9,11,9,10]




    # x = []
    # y = []
    # z = []
    #
    # for b in np.arange(1, 256, 10):
    #     for l in np.arange(0, 5, 0.5):
    #         x.append(b)
    #         y.append(l)
    #         similarity = (color_similarity(5, 6, b, l))
    #         z.append(similarity)
    #
    #
    # ax.scatter(x, y, z, c='r', marker='o')
    #
    # ax.set_xlabel('Bins')
    # ax.set_ylabel('Black pixels')
    # ax.set_zlabel('Similarity')
    #
    # plt.show()


    # # Testing for different nbins values
    # # for s in np.arange(1, 256, 10):
    # #     # print s
    # #     start = timeit.default_timer()
    # #     similarity = (color_similarity(5, 6, s, 0))
    # #     stop = timeit.default_timer()
    # #     # print stop - start
    # #     runtimes.append(stop-start)
    # #     bins_sims.append(similarity)
    # # plt.plot((sims))
    # # plt.plot((runtimes))
    # # plt.plot(np.diff(sims))
    # # plt.plot(np.diff(runtimes))
    #
    # Testing for different b_lb values
    for s in np.arange(0, 200,1):
        # print s
        similarity = (color_similarity(17, 18, 10, s))
        # print similarity
        blb_sims.append(similarity)
    plt.plot((blb_sims))
    plt.show()



    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # # ax.plot(bins_sims, blb_sims)
    #
