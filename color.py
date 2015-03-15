import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from mpl_toolkits.mplot3d import Axes3D
from histogram import histogram
from scipy.optimize import curve_fit

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
    bins_sims_1 = []
    bins_sims_2 = []
    bins_sims_3 = []
    bins_sims_4 = []
    bins_sims_5 = []
    bins_sims_6 = []
    # diffs = []
    blb_sims = []
    deriv = []
    # runtimes = []
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # x =[1,2,3,4,5,6,7,8,9,10]
    # y =[5,6,2,3,13,4,1,2,4,8]
    # z =[2,3,3,3,5,7,9,11,9,10]


    #
    # for lb in np.arange(0, 100, 0.90):
    #     similarity1 = color_similarity(5, 6, 12, lb)
    #     blb_sims.append(similarity1)
    #     deriv.append(np.log(lb)/5)
    #
    #
    # plt.xlabel("Black Pixel Lower Bound (unit: pixel/100)")
    # plt.ylabel("Similarity")
    #
    # plt.plot(deriv)
    #
    # plt.plot(blb_sims)
    # # plt.plot(np.diff(blb_sims))
    #
    # plt.show()

    # color_similarity(5,6,32,10)

    # x = []
    # y = []
    # z = []
    #
    # for b in np.arange(1, 256, 10):
    #     for l in np.arange(0, 5, 0.5):
    #         x.append(b)
    #         y.append(l)
    #         similarity = (color_similarity(3, 5, b, l))
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


    # one = 29
    # # Testing for different nbins values
    # for b in np.arange(1, 256, 1):
    #
    #     # start = timeit.default_timer()
    #     # similarity_1 = (color_similarity(3, 4, b, 0))
    #     # similarity_2 = (color_similarity(5, 6, b, 0))
    #     # similarity_3 = (color_similarity(13, 14, b, 0))
    #     # similarity_4 = (color_similarity(17, 18, b, 0))
    #     # similarity_5 = (color_similarity(7, 9, b, 0))
    #     # similarity_6 = (color_similarity(29, 30, b, 0))
    #
    #     diff2 = (color_similarity(one, 3, b, 0))
    #     diff3 = (color_similarity(one, 5, b, 0))
    #     diff4 = (color_similarity(one, 13, b, 0))
    #     diff5 = (color_similarity(one, 17, b, 0))
    #     diff6 = (color_similarity(one, 7, b, 0))
    #
    #
    #     # stop = timeit.default_timer()
    #     # print stop - start
    #     # runtimes.append(stop-start)
    #
    #     # bins_sims_1.append((similarity_1))
    #     # bins_sims_2.append((similarity_2))
    #     bins_sims_1.append((diff2))
    #     bins_sims_2.append((diff3))
    #     bins_sims_3.append((diff4))
    #     bins_sims_4.append((diff5))
    #     bins_sims_5.append((diff6))
    #
    #     # bins_sims_6.append((similarity_6))
    #     # diffs.append(diff)
    #
    #
    # fig, ax = plt.subplots()
    #
    # # g1 = "A"
    # # g2 = "E"
    # g1 = "d / d(bins) similarity(F,A)"
    # g2 = "d / d(bins) similarity(F,B)"
    # g3 = "d / d(bins) similarity(F,C)"
    # g4 = "d / d(bins) similarity(F,D)"
    # g5 = "d / d(bins) similarity(F,E)"
    # # g6 = "F"
    # ax.plot(np.diff(bins_sims_1), label=g1)
    # ax.plot(np.diff(bins_sims_2), label=g2)
    # ax.plot(np.diff(bins_sims_3), label=g3)
    # ax.plot(np.diff(bins_sims_4), label=g4)
    # ax.plot(np.diff(bins_sims_5), label=g5)
    # # ax.plot((bins_sims_6), label="Similarity(" + g6 + ")")
    #
    # # ax.plot(bins_sims_3, label="Similarity (" + g1 + " , " + g2 + ")")
    # # ax.plot(np.diff(bins_sims_3), label="Diff Similarity (" + g1 + ") - Similarity(" + g2 + ")")
    #
    # legend = ax.legend(loc='lower right', shadow=False)
    #
    # frame = legend.get_frame()
    # frame.set_facecolor('1.0')
    #
    # plt.xlabel("Number of Bins")
    # plt.ylabel("d / d(bins) Similarity")
    #
    # for label in legend.get_texts():
    #     label.set_fontsize('large')
    #
    # for label in legend.get_lines():
    #     label.set_linewidth(1.5)  # the legend line width
    #
    # plt.savefig("../new_similarities/deriv_fix_F.png")
    # plt.show()


    # plt.plot((runtimes))
    # plt.plot(np.diff(bins_sims))
    # plt.plot(np.diff(runtimes))

    one = 3
    # Testing for different b_lb values
    for s in np.arange(0, 256,1):
        similarity_1 = (color_similarity(29, 30, 41, s))
        similarity_2 = (color_similarity(5, 29, 41, s))
        # similarity_3 = (color_similarity(13, 14, 12, s))
        # similarity_4 = (color_similarity(17, 18, 12, s))
        # similarity_5 = (color_similarity(7, 9, 12, s))
        # similarity_6 = (color_similarity(29, 30, 12, s))

        # diff2 = (color_similarity(one, 5, 41, s))
        # diff3 = (color_similarity(one, 5, 41, s))
        # diff4 = (color_similarity(one, 13, 41, s))
        # diff5 = (color_similarity(one, 17, 41, s))
        # diff6 = (color_similarity(one, 7, 41, s))
        #
        diff = similarity_1 - similarity_2
        bins_sims_1.append(similarity_1)
        bins_sims_2.append(similarity_2)
        bins_sims_3.append(diff)

        # bins_sims_1.append((diff2))
        # bins_sims_2.append((diff3))
        # bins_sims_3.append((diff4))
        # bins_sims_4.append((diff5))
        # bins_sims_5.append((diff6))

    fig, ax = plt.subplots()

    # g1 = "similarity(A,B)"
    # g2 = "similarity(A,C)"
    # g3 = "similarity(A,D)"
    # g4 = "similarity(A,E)"
    # g5 = "similarity(A,F)"

    # g1 = "d / d(b_lb) similarity(F,A)"
    # g2 = "d / d(b_lb) similarity(F,B)"
    # g3 = "d / d(b_lb) similarity(F,C)"
    # g4 = "d / d(b_lb) similarity(F,D)"
    # g5 = "d / d(b_lb) similarity(F,E)"

    # g1 = "similarity(A) - similarity(B)"
    # g2 = "similarity(A) - similarity(C)"
    # g3 = "similarity(A) - similarity(D)"
    # g4 = "similarity(A) - similarity(E)"
    # g5 = "similarity(A) - similarity(F)"
    g1 = "Similarity B"
    g2 = "Similarity(A,B)"
    g3 = "Similarity B - Similarity(A,B)"

    # ax.plot((bins_sims_1), label=g1)
    # ax.plot((bins_sims_2), label=g2)
    # ax.plot((bins_sims_3), label=g3)
    # ax.plot((bins_sims_4), label=g4)
    # ax.plot((bins_sims_5), label=g5)

    ax.plot((bins_sims_1), label=g1)
    ax.plot((bins_sims_2), label=g2)
    ax.plot((bins_sims_3), label=g3)

    # ax.plot(np.diff(bins_sims_1), label=g1)
    # ax.plot(np.diff(bins_sims_2), label=g2)
    # ax.plot(np.diff(bins_sims_3), label=g3)
    # ax.plot(np.diff(bins_sims_4), label=g4)
    # ax.plot(np.diff(bins_sims_5), label=g5)

    # plt.axvline(x=50, color='r')
    # plt.axvline(x=150, color='r')


    # ax.plot(np.diff(bins_sims_3), label="deriv")
    legend = ax.legend(loc='upper right', shadow=False)

    # print np.argmin(bins_sims_3)

    frame = legend.get_frame()
    frame.set_facecolor('1.0')

    plt.xlabel("Black Pixel Lower Bound")
    plt.ylabel("Similarity")

    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    # plt.savefig("../new_similarities/Texture_Similarity--Fix_F.png")
    plt.show()


    #
    #
    #
    # # # fig = plt.figure()
    # # # ax = fig.add_subplot(111, projection='3d')
    # # # ax.plot(bins_sims, blb_sims)
    # #
