from color_montage import *
from texture_montage import *
from combine import *

def group_similarity(similarity_type, b, b_lb, r, g1, g2):
    """
    similarity_type       : matrix representing the type of similarity
    (C for color-based, T for texture-based, S for mixed similarity)
    b       : number of bins
    b_lb    : pixels with value lower than b_lb are considered black
    r       : mixture coefficient for C and T
    g1      : image group 1
    g2      : image group 2
    return  : total sum of similarities of all images
    """

    if similarity_type == 'C':
        M = color_similarity_matrix(b, b_lb)

    elif similarity_type == 'T':
        M = texture_similarity_matrix(b, b_lb)

    elif similarity_type == 'S':
        C = color_similarity_matrix(b, b_lb)
        T = texture_similarity_matrix(b, b_lb)
        M = combine(C, T, r)

    g_to_g_similarity = 0
    for i in xrange(0, len(g1)):
        for j in xrange(0, len(g2)):
            g_to_g_similarity += M[g1[i], g2[j]]

    g1_inter_similarity = 0
    for i in xrange(0, len(g1)):
        for j in xrange(0, len(g1)):
            g1_inter_similarity += M[g1[i], g1[j]]

    g2_inter_similarity = 0
    for i in xrange(0, len(g2)):
        for j in xrange(0, len(g2)):
            g2_inter_similarity += M[g2[i], g2[j]]


    res = (g_to_g_similarity, g1_inter_similarity, g2_inter_similarity)

    return res




