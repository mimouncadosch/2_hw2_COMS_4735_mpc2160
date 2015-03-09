import numpy as np
from Cluster import *
from scipy.cluster.hierarchy import linkage, fcluster


def img_cluster(M):


    L = linkage(M, method='complete', metric='euclidean')
    fl = fcluster(L, 7, criterion='maxclust')

    return fl

    # print "ok"
    # rows, cols = D.shape
    #
    # D = np.triu(D) # take only upper half, since matrix is symmetric
    # max = np.max(D)
    # D[D == 0.0] = max + 10 # arbitrarily large number
    #
    # for i in xrange(0, rows):
    #     rng = xrange(0, cols)
    #     min_ids = np.argmin(D, axis=0)  # index of min value in each column
    #     mins = D[min_ids, rng]          # min value in each column
    #
    #     min_idx = np.argsort(mins)      # min_ids sorted by index
    #     # mins[min_idx]                   # minimum value across columns
    #     merge(min_idx[0], i)
    #
    #     print "hey"
    # D = np.diff(1, S)


# def merge(idx1, idx2):
#     return True

# def dist_cluster_to_image(D, c1, idx):

#
#     """
#     :param c1:  cluster index
#     :param idx: image index
#     :return:    distance using complete link
#     """

    # return True

# if __name__ == "__main__":
#     img_cluster()