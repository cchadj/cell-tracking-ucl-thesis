import numpy as np
import mahotas as mh
import tqdm
from skimage.morphology import extrema
from skimage.exposure import match_histograms

import skimage
def imhmaxima(I, H):
    dtype_orig = I.dtype
    I = np.float32(I)
    return skimage.morphology.reconstruction((I - H), I).astype(dtype_orig)

def imextendedmax(I, H, conn=8):
    if conn == 4:
        structuring_element = np.array([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]],
                                       dtype=np.bool)
    elif conn == 8:
        structuring_element = np.ones([3, 3],
                                      dtype=np.bool)

    h_maxima_result = imhmaxima(I, H)
    extended_maxima_result = mh.regmax(h_maxima_result, Bc=structuring_element)

    return extended_maxima_result


def normalize_data(data, target_range=(0, 1), data_range=None):
    """ Normalizes data to a given target range. (Min Max Normalisation)

    Args:
        data: The data to normalise. Can be of any shape.
        target_range (tuple): (a, b) The target lowest and highest value
        data_range (float): Current data min and max.
                          If None then uses the data minimum and maximum value.
                          Useful when max value may not be observed in the data given.
    Returns:
        Normalised data within target range.
        Same type as data.

    """
    if data_range is None:
        data_range = data.min(), data.max()

    data_min, data_max = data_range
    alpha, beta = target_range

    assert alpha < beta, f'Target range should be from small to big, target range given: {target_range}.'
    assert data_min < data_max, f'Data range should be from small to big, data range given: {data_range}.'

    return ((beta - alpha) * ((data - data_min) / (data_max - data_min)) + alpha).astype(data.dtype)


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


def hist_match_images(images, reference):
    hist_matched_images = np.empty_like(images)

    for i, im in enumerate(tqdm.tqdm(images)):
        hist_matched_images[i] = match_histograms(im, reference)

    return hist_matched_images
