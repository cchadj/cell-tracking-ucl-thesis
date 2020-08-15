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


def imagestack_to_vstacked_image(framestack):
    return np.hstack(np.split(framestack, len(framestack), axis=0)).squeeze()


def vstacked_image_to_imagestack(stacked_image, n_images):
    return np.array(np.split(stacked_image, n_images, axis=0))


def imagestack_to_hstacked_image(framestack):
    return np.hstack([f.squeeze() for f in np.split(framestack, len(framestack))])


def hstacked_image_to_imagestack(stacked_image, n_images):
    return np.array(np.split(stacked_image, n_images, axis=1))


def hist_match_2(images, reference):
    return vstacked_image_to_imagestack(match_histograms(imagestack_to_vstacked_image(images), reference), len(images))


from skimage import exposure
from imageprosessing import normalize_data


def crop_mask(mask, left_crop=50):
    """ Crop mask or stack of masks by amount of pixels.

    Can provide a single mask HxW or stack of masks NxHxW
    """

    new_mask = mask.copy()
    if len(new_mask.shape) == 2:
        # If single mask HxW -> 1xHxW
        new_mask = new_mask[np.newaxis, ...]

    for mask_count, m in enumerate(new_mask):
        if np.any(m[:, 0]):
            # edge case where there's a line at the first column in which case we say that the mean edge
            # pixel is the at the first column
            mean_edge_x = 0
        else:
            ys, xs = np.where(np.diff(m))
            edge_ys = np.unique(ys)
            edge_xs = np.empty_like(edge_ys)
            for i, y in enumerate(edge_ys):
                edge_xs[i] = (np.min(xs[np.where(ys == y)[0]]))

            mean_edge_x = np.mean(edge_xs)

        new_mask[mask_count, :, :int(mean_edge_x + left_crop)] = 0

    return new_mask.squeeze()


def std_image(frames,
              masks=None,
              method='de_castro',
              adapt_hist=False,
              sigma=0.75):
    # 20. J. Tam, J. A. Martin, and A. Roorda, “Noninvasive visualization and analysis of parafoveal capillaries in humans,”
    # Invest. Ophthalmol. Vis. Sci. 51(3), 1691–1698 (2010).
    assert method in [None, 'j_tam', 'de_castro']

    frames = frames.copy()
    if frames.dtype == np.uint8:
        frames = np.float32(frames) / 255

    sigma = 1

    if sigma >= 0.125:
        for i, frame in enumerate(frames):
            frames[i, ...] = mh.gaussian_filter(frame, sigma)

    if masks is None:
        masked_frames = frames.copy()
    else:
        # crop ~15 pixels from the left part of the mask to avoid vertical streak artifacts.
        mask_frames = crop_mask(masks, 15)
        masked_frames = np.ma.masked_array(frames, ~mask_frames)
    if method == 'de_castro':
        # We invert the mask because True values mean that the values are masked and therefor invalid
        # https://numpy.org/doc/stable/reference/maskedarray.generic.html
        for i, masked_frame in enumerate(masked_frames):
            masked_frames[i, ...] = masked_frame / np.ma.mean(masked_frame)

        m = np.ma.mean(masked_frames, axis=0)
        for i, masked_frame in enumerate(masked_frames):
            masked_frames[i, ...] = masked_frame / m

        final_processed_frames = masked_frames
    elif method == 'j_tam':
        # Create division frames by dividing consecutive frames (last frame remains unprocessed)
        division_frames = masked_frames
        for j in range(len(masked_frames) - 1):
            division_frames[j] = masked_frames[j] / masked_frames[j + 1]
        division_frames = division_frames[:-1]

        # Create multiframe frames by averaging consecutive division frames (last frame remains unprocessed)
        multiframe_div_frames = division_frames.copy()
        for j in range(len(division_frames) - 1):
            multiframe_div_frames[j] = (division_frames[j] + division_frames[j + 1]) / 2
            print(multiframe_div_frames[j].filled(multiframe_div_frames[j].mean()).min(),
                  multiframe_div_frames[j].filled(multiframe_div_frames[j].mean()).max())
            if adapt_hist:
                try:
                    multiframe_div_frames[j] = exposure.equalize_adapthist(
                        normalize_data(multiframe_div_frames[j].filled(multiframe_div_frames[j].mean()), (0, 1)))
                except:
                    return multiframe_div_frames[j], division_frames[j], division_frames[j + 1]
        final_processed_frames = multiframe_div_frames[:-1]
    elif method is None:
        final_processed_frames = masked_frames

    return final_processed_frames.std(0)


if __name__ == '__main__':
    from sharedvariables import get_video_sessions
    from matplotlib import pyplot as plt
    import os

    video_sessions = get_video_sessions(should_have_marked_video=True)
    vs = [vs for vs in video_sessions if os.path.basename(vs.video_file) == 'Subject47_Session375_OD_(0,-1)_1.04x1.04_3056_OA790nm1_extract_reg_cropped.avi'][0]

    plt.imshow(std_image(vs.frames_oa850, vs.mask_frames_oa850, sigma=0, method='j_tam', adapt_hist=True), cmap='gray')
    plt.show()
