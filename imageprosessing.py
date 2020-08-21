from typing import List, Any

import evaluation
import numpy as np
import mahotas as mh
import tqdm
from skimage.morphology import extrema
from skimage.exposure import match_histograms
from plotutils import cvimshow

import cv2
from learningutils import ImageDataset
import torch.utils.data
import torchvision
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


def center_crop_images(images, patch_size):
    """ Crops the centre of the stack of images and returns the result

    Args:
        images: NxHxWxC (or NxHxW)
        patch_size (int or tuple): The dimensions of the patch to crop in the middle

    Returns:
      N x patch height x patch width x C ( or Nx patch height x patch width) numpy array
    """
    crop_transform = [torchvision.transforms.CenterCrop(patch_size)]
    dataset = ImageDataset(images, data_augmentation_transforms=crop_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    for batch in loader:
        # NxCxHxW -> NxHxWxC
        return batch.permute(0, 2, 3, 1).cpu().numpy().squeeze().astype(images.dtype)


class ImageRegistator(object):
    def __init__(self, source, target):
        self.source = source
        self.target = target

        # Warp affine doesn't work with boolean
        if source.dtype == np.bool8:
            self.source = np.float32(self.source)
        if target.dtype == np.bool8:
            self.target = np.float32(self.target)

        self.best_dice = evaluation.dice(source, target)
        self.dices = [self.best_dice]
        self.vertical_displacement = 0
        self.horizontal_displacement = 0
        self.registered_source = source

    def register_vertically(self):
        dx = 0
        dys = np.int32(np.arange(1, 200, 1))

        # fig, axes = plt.subplots(len(dys), 1, figsize=(100, 100))

        dices = []
        for i, dy in enumerate(dys):
            translation = np.float32([[1, 0, dx],
                                      [0, 1, dy]])

            height, width = self.source.shape[:2]

            translated_source = cv2.warpAffine(self.source, translation, (width, height))
            dice_v = evaluation.dice(self.target, translated_source)
            dices.append(dice_v)

        # Get displacement that gives best dice coefficient.
        dy = dys[np.argmax(dices)]

        translation = np.float32([[1, 0, dx],
                                  [0, 1, dy]])
        height, width = self.source.shape[:2]
        translated_source = cv2.warpAffine(self.source, translation, (width, height))

        self.registered_source = translated_source
        self.vertical_displacement = dy
        self.best_dice = max(dices)
        self.dices = dices

        return self.registered_source

    def apply_registration(self, im):
        dx, dy = self.horizontal_displacement, self.vertical_displacement
        if im.dtype == np.bool:
            im = np.float32(im)

        translation = np.float32([[1, 0, dx],
                                  [0, 1, dy]])
        height, width = im.shape[:2]
        im = cv2.warpAffine(im, translation, (width, height))

        return im

    @staticmethod
    def vertical_image_registration(source, target):
        dx = 0
        dys = np.int32(np.arange(1, 200, 1))

        # fig, axes = plt.subplots(len(dys), 1, figsize=(100, 100))

        dices = []
        for i, dy in enumerate(dys):
            translation = np.float32([[1, 0, dx],
                                      [0, 1, dy]])

            height, width = source.shape[:2]

            translated_source = cv2.warpAffine(source, translation, (width, height))
            dice_v = evaluation.dice(target, translated_source)
            dices.append(dice_v)

        return translated_source, dy


if __name__ == '__main__':
    from sharedvariables import get_video_sessions
    from video_session import SessionPreprocessor

    video_sessions = get_video_sessions(should_have_marked_cells=True)
    vs = video_sessions[0]

    preprocessor = SessionPreprocessor(vs, [
        lambda frames: frame_differencing(frames, sigma=1.2),
        lambda frames: np.uint8(normalize_data(frames, (0, 255))),
        lambda frames: enhance_motion_contrast(frames, sigma=1.2)
    ])
    preprocessor.apply_preprocessing_to_oa790()

    # plt.subplot(121)
    cvimshow('', vs.frames_oa790[0])

