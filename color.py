import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
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
    # bins_sims_1 = []
    # bins_sims_2 = []
    # bins_sims_3 = []
    # bins_sims_4 = []
    # bins_sims_5 = []
    # bins_sims_6 = []
    # diffs = []
    blb_sims = []
    # runtimes = []
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # x =[1,2,3,4,5,6,7,8,9,10]
    # y =[5,6,2,3,13,4,1,2,4,8]
    # z =[2,3,3,3,5,7,9,11,9,10]



    for lb in np.arange(0, 40, 1):
        similarity1 = color_similarity(5, 6, 14, lb)
        blb_sims.append(similarity1)

    plt.xlabel("Black Pixel Lower Bound")
    plt.ylabel("Similarity")
    plt.plot(blb_sims)
    plt.show()

    # color_similarity(5,6,32,10)

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
    # for b in np.arange(1, 20, 1):
    #
    #     # start = timeit.default_timer()
    #     similarity_1 = (color_similarity(3, 4, b, 0))
    #     similarity_2 = (color_similarity(5, 6, b, 0))
    #     similarity_3 = (color_similarity(13, 14, b, 0))
    #     similarity_4 = (color_similarity(17, 18, b, 0))
    #     similarity_5 = (color_similarity(7, 9, b, 0))
    #     similarity_6 = (color_similarity(29, 30, b, 0))
    #     diff2 = similarity_1 - similarity_2
    #     diff3 = similarity_6 - similarity_2
    #     diff4 = similarity_6 - similarity_3
    #     diff5 = similarity_6 - similarity_4
    #     diff6 = similarity_6 - similarity_5
    #     # stop = timeit.default_timer()
    #     # print stop - start
    #     # runtimes.append(stop-start)
    #
    #     bins_sims_1.append((diff2))
    #     bins_sims_2.append((diff3))
    #     bins_sims_3.append((diff4))
    #     bins_sims_4.append((diff5))
    #     bins_sims_5.append((diff6))
    #     # bins_sims_6.append((similarity_6))
    #     # diffs.append(diff)
    #
    #
    # fig, ax = plt.subplots()
    #
    # g1 = "similarity(B) - similarity(A)"
    # g2 = "similarity(B) - similarity(C)"
    # g3 = "similarity(B) - similarity(D)"
    # g4 = "similarity(B) - similarity(E)"
    # g5 = "similarity(B) - similarity(F)"
    # # g6 = "F"
    # ax.plot((bins_sims_1), label=g1)
    # ax.plot((bins_sims_2), label=g2)
    # ax.plot((bins_sims_3), label=g3)
    # ax.plot((bins_sims_4), label=g4)
    # ax.plot((bins_sims_5), label=g5)
    # ax.plot((bins_sims_6), label="Similarity(" + g6 + ")")

    # ax.plot(diffs, label="Similarity (" + g1 + ") - Similarity(" + g2 + ")")

    # legend = ax.legend(loc='upper right', shadow=False)
    #
    # frame = legend.get_frame()
    # frame.set_facecolor('1.0')
    #
    # plt.xlabel("Number of Bins")
    # plt.ylabel("Similarity")
    #
    # for label in legend.get_texts():
    #     label.set_fontsize('large')
    #
    # for label in legend.get_lines():
    #     label.set_linewidth(1.5)  # the legend line width
    #
    # plt.savefig("../similarities/Similarity_TOT_rangeB.png")
    # plt.show()


    # plt.plot((runtimes))
    # plt.plot(np.diff(bins_sims))
    # plt.plot(np.diff(runtimes))
    #
    # Testing for different b_lb values
    # for s in np.arange(0, 10,0.1):
    #     similarity_1 = (color_similarity(3, 4, 14, s))
    #     similarity_2 = (color_similarity(5, 6, 14, s))
    #     similarity_3 = (color_similarity(13, 14, 14, s))
    #     similarity_4 = (color_similarity(17, 18, 14, s))
    #     similarity_5 = (color_similarity(7, 9, 14, s))
    #     similarity_6 = (color_similarity(29, 30, 14, s))
    #
    #     diff2 = similarity_1 - similarity_2
    #     diff3 = similarity_1 - similarity_3
    #     diff4 = similarity_1 - similarity_4
    #     diff5 = similarity_1 - similarity_5
    #     diff6 = similarity_1 - similarity_6
    #
    #     bins_sims_1.append(similarity_1)
    #     bins_sims_2.append(similarity_2)
    #     bins_sims_3.append(diff2)
    #
    #     # bins_sims_1.append((diff2))
    #     # bins_sims_2.append((diff3))
    #     # bins_sims_3.append((diff4))
    #     # bins_sims_4.append((diff5))
    #     # bins_sims_5.append((diff6))
    #
    # fig, ax = plt.subplots()
    #
    # g1 = "similarity(A)"
    # g2 = "similarity(B)"
    # g3 = "similarity(A) - similarity(B)"

    # g1 = "similarity(A) - similarity(B)"
    # g2 = "similarity(A) - similarity(C)"
    # g3 = "similarity(A) - similarity(D)"
    # g4 = "similarity(A) - similarity(E)"
    # g5 = "similarity(A) - similarity(F)"
    # g6 = "F"

    # ax.plot((bins_sims_1), label=g1)
    # ax.plot((bins_sims_2), label=g2)
    # ax.plot((bins_sims_3), label=g3)
    # ax.plot((bins_sims_4), label=g4)
    # ax.plot((bins_sims_5), label=g5)

    # legend = ax.legend(loc='upper right', shadow=False)
    #
    # frame = legend.get_frame()
    # frame.set_facecolor('1.0')
    #
    # plt.xlabel("Number of Bins")
    # plt.ylabel("Similarity")
    #
    # for label in legend.get_texts():
    #     label.set_fontsize('large')
    #
    # for label in legend.get_lines():
    #     label.set_linewidth(1.5)  # the legend line width
    #
    # plt.savefig("../similarities/Texture_Similarity_Fix_A.png")
    # plt.show()





    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # # ax.plot(bins_sims, blb_sims)
    #
