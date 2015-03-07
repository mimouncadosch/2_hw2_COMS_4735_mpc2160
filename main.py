from color_montage import *
import texture
import color
from texture_montage import *
from combine import *

if __name__ == "__main__":
    C = color_similarity_matrix(32, 0)
    color_montage(C)

    T = texture_similarity_matrix(100, 0)
    texture_montage(T)

    combine(C, T)
