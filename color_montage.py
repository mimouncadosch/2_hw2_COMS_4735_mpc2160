import numpy as np
import cv2
from color import color_similarity


def color_similarity_matrix(b, b_lb):
    """Computes the similarity matrix for the set of images
    b 	    : number of bins
    b_lb    : lower bound for pixel values considered black. In other words, all values < b_lb will be considered black
    """
    M = np.zeros((40, 40))

    # Numpy matrix indexed from 0 to 39, images labeled from 1 to 40
    for i in xrange(0, 40):
        for j in xrange(0, 40):
            M[j, i] = color_similarity(i+1, j+1, b, b_lb)

    return M


def color_montage(M):
    """Creates the image montage of {image} {3 most similar images} {3 most dissimilar images}
	"""
    top = np.zeros((3))
    bottom = np.zeros((3))

    montage = np.zeros((90*40, 89*7, 3))

    # Columns in the matrix range from 0 to 39
    # Images names range from 1 to 40
    # For each column in the matrix
    for i in xrange(0, 40):
        # This array has the sorted indices of the values of the column
        M_sorted = np.argsort(M[:,i])

        # unlike and like have the indices of the most dissimilar and similar images to the image of index (i+1)
        # The indices in unlike and like also need to be ++ to correspond to image names
        unlike = M_sorted[0:3]
        like = M_sorted[36:39]

        # Image has index (i+1)
        idx = parse_id(i)
        img = cv2.imread("../Images/i" + idx + ".ppm")

        mat = np.ones((30, 89,3))
        cv2.putText(mat, str(idx), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        mat = np.uint8(mat)

        img = np.vstack((img, mat))
        row = np.zeros((90, 89*7))

        # Add most unlike images
        for j in xrange(0, 3):
            img2 = retrieve_img(unlike[j])

            # Image label
            mat = np.ones((30, 89,3))
            cv2.putText(mat, str(unlike[j]), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
            mat = np.uint8(mat)

            img2 = np.vstack((img2, mat))

            if j == 0:
                row = np.hstack((img, img2))
            else:
                row = np.hstack((row, img2))

            # TODO: Image labels

        for j in xrange(0, 3):
            img3 = retrieve_img(like[j])

            # Image label
            mat = np.ones((30, 89,3))
            cv2.putText(mat, str(like[j]), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
            mat = np.uint8(mat)

            img3 = np.vstack((img3, mat))
            row = np.hstack((row, img3))

        if i == 0:
            montage = row
        else:
            montage = np.vstack((montage, row))

    cv2.imwrite('color_montage.png', montage)
    cv2.imshow('montage', montage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return True

def retrieve_img(idx):
    idx = parse_id(idx)
    img = cv2.imread("../Images/i" + idx + ".ppm")
    return img


def parse_id(idx):
    if (idx+1) < 10:
        idx = "0" + str(idx+1)
    else:
        idx = str(idx+1)
    return idx