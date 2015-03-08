from color_montage import *
import texture
import color
from texture_montage import *
from combine import *
from group_similarity import *
from img_cluster import *

if __name__ == "__main__":

    textured    = [1,2,3,4,5,6,7,8,9, 10, 11]
    smooth      = [12,13,14,15,16,17, 18, 19, 20]

    b           = 32
    r           = 1
    b_lb_max    = 10
    b_lb_step   = 0.5
    r_max       = 1
    r_step      = 0.1


    # The following code is to choose the right parameters b, b_lb
    g_to_g      = np.zeros((b_lb_max/b_lb_step))
    g1_inter    = np.zeros((b_lb_max/b_lb_step))
    g2_inter    = np.zeros((b_lb_max/b_lb_step))

    for i in np.arange(0, b_lb_max, b_lb_step):
        c_similarities  = group_similarity('C', b, i, r, textured, smooth)
        g_to_g[i]       = c_similarities[0]
        g1_inter[i]     = c_similarities[1]
        g2_inter[i]     = c_similarities[2]


    # plt.plot(g_to_g, label="intra-group")
    # plt.plot(g1_inter, label="g1_inter")
    # plt.plot(g2_inter, label="g2_inter")
    plt.plot(g_to_g-g1_inter, label="g^2 - g_inter")
    plt.legend(bbox_to_anchor=(0.90, 1), loc=2, borderaxespad=0.)
    plt.show()

    # for i in np.arange(0, b_lb_max, b_lb_step):
    #     t_similarities = group_similarity('T', b, i, r, textured, smooth)


    # TODO: Decide parameters:
    b_lb = 10
    b = 100

    # for i in np.arange(0, r_max, r_step):
    #     s_similarities = group_similarity('S', b, b_lb, i, textured, smooth)


    # Parameters b and b_lb are selected


    C = color_similarity_matrix(b, b_lb)
    T = texture_similarity_matrix(b, b_lb)
    # color_montage(C)
    # texture_montage(T)
    S = combine(C, T, r)

    img_cluster(S)

