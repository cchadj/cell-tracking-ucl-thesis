from typing import List, Any
from matplotlib import pyplot as plt
import numpy as np
import mahotas as mh
import tqdm
from skimage.morphology import extrema
from skimage.exposure import match_histograms
from plotutils import cvimshow

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


def hist_equalize(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


def equalize_hist_images(images):
    equalized_images = np.empty_like(images)
    for i, im in enumerate(images):
        equalized_images[i] = hist_equalize(im)[0]

    return equalized_images


def equalize_adapthist_images(images):
    from skimage import exposure
    equalized_images = np.empty_like(images)
    for i, im in enumerate(images):
        equalized_images[i] = np.uint8(exposure.equalize_adapthist(im) * 255)

    return equalized_images


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


def frame_differencing(frames, sigma=0):
    import numpy as np
    frames = np.float32(frames.copy())

    background = frames.mean(0)
    if sigma >= 0.125:
        background = mh.gaussian_filter(background, sigma)

    difference_images = np.empty_like(frames)
    for j in range(len(frames) - 1):
        difference_images[j] = mh.gaussian_filter(frames[j], sigma) - background

    return difference_images[:-1]

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


def enhance_motion_contrast(frames, sigma=1, adapt_hist=False, mask_crop_pixels=15, method='j_tam'):
    from skimage import exposure
    idx_to_remove = []
    for idx, frame in enumerate(frames):
        if frame.mean() < 20:
            idx_to_remove.append(idx)

    frames = np.delete(frames, idx_to_remove, 0)
    # avoid division with 0
    if frames.dtype == np.uint8:
        frames = np.float64(frames) / 255

    if sigma >= 0.125:
        for i, frame in enumerate(frames):
            frames[i, ...] = mh.gaussian_filter(frame, sigma)

    # Create division frames by dividing consecutive frames (last frame remains unprocessed)
    division_frames = np.ma.empty_like(frames)
    for j in range(len(frames) - 1):
        division_frames[j] = frames[j] / frames[j + 1]
        # print(division_frames[j].min(),
        #       division_frames[j].max(),
        #       division_frames[j].mean())
    division_frames = division_frames[:-1]
    division_frames[division_frames > 2] = 2

    # Create multiframe frames by averaging consecutive division frames (last frame remains unprocessed)
    multiframe_div_frames = np.ma.empty_like(division_frames)
    for j in range(len(division_frames) - 1):
        multiframe_div_frames[j] = (division_frames[j] + division_frames[j + 1]) / 2
        # print(multiframe_div_frames[j].filled(multiframe_div_frames[j].mean()).min(),
        #       multiframe_div_frames[j].filled(multiframe_div_frames[j].mean()).max(),
        #       multiframe_div_frames[j].filled(multiframe_div_frames[j].mean()).mean())
        if adapt_hist:
            try:
                multiframe_div_frames[j] = exposure.equalize_adapthist(
                    normalize_data(multiframe_div_frames[j].filled(multiframe_div_frames[j].mean()), (0, 1)))
            except:
                return multiframe_div_frames[j], division_frames[j], division_frames[j + 1]
    final_processed_frames = multiframe_div_frames[:-1]

    final_processed_frames = normalize_data(final_processed_frames, (0, 255))
    # print(final_processed_frames.min(), final_processed_frames.max(), final_processed_frames.mean())
    final_processed_frames = np.uint8(final_processed_frames)

    return final_processed_frames


def std_image(frames,
              masks=None,
              method='de_castro',
              adapt_hist=False,
              sigma=0.75):
    # 20. J. Tam, J. A. Martin, and A. Roorda, “Noninvasive visualization and analysis of parafoveal capillaries in humans,”
    # Invest. Ophthalmol. Vis. Sci. 51(3), 1691–1698 (2010).
    assert method in [None, 'j_tam', 'de_castro']
    from skimage import exposure

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


class SessionPreprocessor(object):
    preprocess_functions: List[Any]

    from sharedvariables import VideoSession
    session: VideoSession

    def __init__(self, session, preprocess_functions=None):
        from collections.abc import Iterable
        if preprocess_functions is None:
            preprocess_functions = []
        elif not isinstance(preprocess_functions, Iterable):
            preprocess_functions = [preprocess_functions]

        self.preprocess_functions = preprocess_functions
        self.session = session

    def _apply_preprocessing(self, masked_frames):
        for fun in self.preprocess_functions:
            masked_frames = fun(masked_frames)
        frames = masked_frames.filled(masked_frames.mean(0))
        return frames

    def apply_preprocessing_to_oa790(self):
        self.session.frames_oa790 = self._apply_preprocessing(self.session.masked_frames_oa790)

    def apply_preprocessing_to_oa850(self):
        self.session.frames_oa850 = self._apply_preprocessing(self.session.masked_frames_oa850)

    def apply_preprocessing_to_confocal(self):
        self.session.frames_confocal = self._apply_preprocessing(self.session.masked_frames_confocal)

    def apply_preprocessing(self):
        self.apply_preprocessing_to_confocal()
        self.apply_preprocessing_to_oa790()
        self.apply_preprocessing_to_oa850()


if __name__ == '__main__':
    from sharedvariables import get_video_sessions

    video_sessions = get_video_sessions(should_have_marked_video=True)
    vs = video_sessions[0]

    preprocessor = SessionPreprocessor(vs, [
        lambda frames: frame_differencing(frames, sigma=1.2),
        lambda frames: np.uint8(normalize_data(frames, (0, 255))),
        lambda frames: enhance_motion_contrast(frames, sigma=1.2)
    ])
    preprocessor.apply_preprocessing_to_oa790()

    # plt.subplot(121)
    cvimshow('', vs.frames_oa790[0])

    pass
