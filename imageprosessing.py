import numpy as np
import mahotas as mh
from skimage.morphology import extrema

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


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float32)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float32)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def normalize_data(data, target_range=(0, 1), data_range=None):
    """ Normalizes data to a given target range

    Args:
        data: The data to normalise. Can be of any shape.
        target_range (tuple): (a, b) The target lowest and highest value
        data_range (float): Current data min and max.
                          If None then uses the data minimum and maximum value.
                          Useful when max value may not be observed in the data given.
    Returns:
        Normalised data within target range.

    """
    if data_range is None:
        data_range = data.min(), data.max()

    data_min, data_max = data_range
    alpha, beta = target_range

    assert alpha < beta, f'Target range should be from small to big, target range given: {target_range}.'
    assert data_min < data_max, f'Data range should be from small to big, data range given: {data_range}.'

    return (beta - alpha) * ((data - data_min) / (data_max - data_min)) + alpha


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


def hist_match_images(images, template):
    cell_images_equalized = np.empty_like(images)

    for i, im in enumerate(images):
        if i == 0:
            pass
        hist_matched_image = hist_match(im, template)
        cell_images_equalized[i, ...] = hist_matched_image

    return cell_images_equalized
