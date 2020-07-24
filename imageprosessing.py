import numpy as np
import mahotas as mh
from skimage.morphology import extrema


def imextendedmax(I, H, conn=4):
    if conn == 4:
        structuring_element = np.array([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]],
                                       dtype=np.bool)
    elif conn == 8:
        structuring_element = np.ones([3, 3],
                                      dtype=np.bool)

    h_maxima_result = extrema.h_maxima(I, H, selem=structuring_element)
    extended_maxima_result = mh.regmax(h_maxima_result, Bc=structuring_element)

    return extended_maxima_result
