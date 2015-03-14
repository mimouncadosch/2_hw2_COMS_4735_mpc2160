import numpy as np
import cv2
from matplotlib import pyplot as plt

def histogram(nbins, idx, b_lb):

    if idx < 10: idx = "0" + str(idx)
    img = cv2.imread("../Images/i" + str(idx) + ".ppm")

    # 3, 2D histograms
    fig = plt.figure()
    chans = cv2.split(img)

    lb = (b_lb, b_lb, b_lb)
    ub = (255, 255, 255)
    mask = cv2.inRange(img, lb, ub)

    # plot a 2D color histogram for green and blue
    ax = fig.add_subplot(131)
    hist = cv2.calcHist([chans[1], chans[0]], [0, 1], mask,
        [nbins, nbins], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation = "nearest")
    ax.set_title("2D Color Histogram for Green and Blue")
    plt.colorbar(p)

    # plot a 2D color histogram for green and red
    ax = fig.add_subplot(132)
    hist = cv2.calcHist([chans[1], chans[2]], [0, 1], mask,
	[nbins, nbins], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation = "nearest")
    ax.set_title("2D Color Histogram for Green and Red")
    plt.colorbar(p)

    # plot a 2D color histogram for blue and red
    ax = fig.add_subplot(133)
    hist = cv2.calcHist([chans[0], chans[2]], [0, 1], mask,
	[nbins, nbins], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation = "nearest")
    ax.set_title("2D Color Histogram for Blue and Red")
    plt.colorbar(p)

    # finally, let's examine the dimensionality of one of
    # the 2D histograms
    print "2D histogram shape: %s, with %d values" % (
	hist.shape, hist.flatten().shape[0])

    plt.show()


    #
    # rows, cols, depth = img.shape
    # print rows, cols
    #
    # v = (256 / bin_size) + 1
    # hist = [np.zeros(v), np.zeros(v), np.zeros(v)]
    #
    # for r in xrange(0, rows):
    #     for c in xrange(0, cols):
    #         pixel = img[r,c]    # 3x1 array
    #         for i in xrange(0,3):
    #             pixel[i] = round(pixel[i]/bin_size)
    #             hist[i][pixel[i]] += 1
    #
    # hist = np.array(hist).ravel()
    return True

if __name__ == "__main__":
    histogram(100, 5, 40)
