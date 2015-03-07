import numpy as np
import cv2

def histogram(bin_size, idx):
    if idx < 10: idx1 = "0" + str(idx)

    img = cv2.imread("../Images/i" + str(idx) + ".ppm")

    rows, cols, depth = img.shape
    print rows, cols

    v = (256 / bin_size) + 1
    hist = [np.zeros(v), np.zeros(v), np.zeros(v)]

    for r in xrange(0, rows):
        for c in xrange(0, cols):
            pixel = img[r,c]    # 3x1 array
            for i in xrange(0,3):
                pixel[i] = round(pixel[i]/bin_size)
                hist[i][pixel[i]] += 1

    hist = np.array(hist).ravel()
    return hist
