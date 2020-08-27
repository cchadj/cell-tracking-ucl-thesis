import skimage
import skimage.filters
import skimage.exposure
from matplotlib import pylab as plt
import numpy as np
import cv2
import mahotas as mh
from imageprosessing import crop_mask, normalize_data


def create_average_and_stdev_image(frames, masks=None):
    if masks is None:
        masks = np.ones_like(frames, dtype=np.bool8)

    summation = np.zeros(frames.shape[1:3], dtype=np.float64)
    votes = np.zeros(frames.shape[1:3], dtype=np.float64)
    for frame, mask in zip(frames, masks):
        summation[mask] += frame[mask]
        votes[mask] += 1

    average_img = summation / votes

    stdev_img = (summation - average_img) ** 2 / votes

    return average_img, stdev_img


def binarize_vessel_image(
        vessel_image,
        n_iterations=1,
        normalise_fangi=False,
        equalize_frangi_hist=False,

        opening_kernel_size=5,
        closing_kernel_size=13,
        padding=cv2.BORDER_REPLICATE,
        padding_size=30,
        padding_value=None,

        visualise_intermediate_steps=False):
    from skimage import morphology
    vessel_image = cv2.copyMakeBorder(vessel_image,
                                      padding_size,
                                      padding_size,
                                      padding_size,
                                      padding_size,
                                      padding,
                                      value=padding_value)
    # opening_kernel = morphology.disk(opening_kernel_size)
    # closing_kernel = morphology.disk(closing_kernel_size)
    print('helo')
    opening_kernel = morphology.square(opening_kernel_size)
    closing_kernel = morphology.square(closing_kernel_size)

    for it in range(n_iterations):
        sigma = 2
        if sigma > 0:
            vessel_image_blurred = skimage.filters.gaussian(vessel_image, sigma=sigma)
        else:
            vessel_image_blurred = vessel_image
        frangi_image = skimage.filters.frangi(vessel_image_blurred, alpha=.5, beta=.5, black_ridges=False)

        if normalise_fangi:
            frangi_image = cv2.normalize(frangi_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        if equalize_frangi_hist:
            frangi_image = skimage.exposure.equalize_adapthist(frangi_image)

        binary_threshold = skimage.filters.threshold_otsu(frangi_image, nbins=256)
        BW = np.zeros_like(frangi_image)
        BW[frangi_image > binary_threshold * 0.5] = 1
        BW = np.uint8(BW)

        # Opening get's rid of small specles
        opening = cv2.morphologyEx(BW, cv2.MORPH_OPEN, opening_kernel)
        # Closing connects components
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, closing_kernel)

        if visualise_intermediate_steps:
            fig, axes = plt.subplots(1, 6, figsize=(60, 60))
            axes[0].imshow(vessel_image[padding_size:-padding_size, padding_size:-padding_size], cmap='gray')
            axes[0].set_title('Vessel image', fontsize=60)
            axes[1].imshow(vessel_image_blurred[padding_size:-padding_size, padding_size:-padding_size], cmap='gray')
            axes[1].set_title('Blurred image', fontsize=60)
            axes[2].imshow(frangi_image[padding_size:-padding_size, padding_size:-padding_size])
            axes[2].set_title('Frangi image', fontsize=60)
            axes[3].imshow(BW[padding_size:-padding_size, padding_size:-padding_size])
            axes[3].set_title('Binary image', fontsize=60)
            axes[4].imshow(opening[padding_size:-padding_size, padding_size:-padding_size])
            axes[4].set_title('Opening', fontsize=60)
            axes[5].imshow(closing[padding_size:-padding_size, padding_size:-padding_size])
            axes[5].set_title('Closing', fontsize=60)

        vessel_image = closing

    vessel_image = vessel_image[padding_size:-padding_size, padding_size:-padding_size]
    vessel_image[:, :20] = 0
    vessel_image[:, -20:] = 0

    return np.bool8(vessel_image)


def create_vessel_mask_from_frames(frames, masks=None, de_castro=True, sigma=1, adapt_hist=True,
                                   equalize_frangi_hist=True):
    from imageprosessing import enhance_motion_contrast_j_tam, enhance_motion_contrast_de_castro, stack_to_masked_array, \
        gaussian_blur_stack
    frames = stack_to_masked_array(frames, masks)

    frames = gaussian_blur_stack(frames, sigma=sigma)

    if de_castro:
        frames = enhance_motion_contrast_de_castro(frames, masks, sigma=0)
    frames = enhance_motion_contrast_j_tam(frames, sigma=0, adapt_hist=adapt_hist)

    std_img = frames.std(0)
    std_img = std_img.filled(std_img.mean())
    std_img = skimage.exposure.equalize_adapthist(std_img)
    mask = binarize_vessel_image(std_img, equalize_frangi_hist=True,
                                 opening_kernel_size=5, closing_kernel_size=8,
                                 visualise_intermediate_steps=True)
    return mask
