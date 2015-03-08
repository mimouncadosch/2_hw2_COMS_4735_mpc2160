import numpy as np
import cv2

def combined_similarity_montage(clusters, id):
    """Creates the image montage of the seven clusters, as clustered by combined similarity
    :param clusters: list of images indexed by the cluster they belong to
    :param id: id of the cluster for the montage
    """

    # 4 columns per row
    img_list = np.where(clusters == id)[0]
    num_imgs = len(img_list)
    num_rows = np.ceil(num_imgs/4)
    montage = np.zeros((60*(num_rows+1), 89*4, 3))
    montage = np.uint8(montage)

    row = 0
    for col in xrange(0, num_imgs):
        # Image has index (i+1)
        idx = parse_id(img_list[col])
        img = cv2.imread("../Images/i" + idx + ".ppm")

        if col > 0 and col%4 == 0:
            row += 1

        montage[row*60:(row+1)*60, (col%4)*89:((col%4)+1)*89,:] = img

    # cv2.imshow("montage", montage)
    cv2.imwrite("../clusters/cluster-" + str(id) + ".png", montage)
    # cv2.waitKey(0)

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