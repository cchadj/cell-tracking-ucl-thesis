import skimage.filters
from matplotlib import pylab as plt
import numpy as np
import cv2


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


def veselness_detection(im,
                        visualise_intermediate_results=False):


    # im_blurred = skimage.filters.median(im,  mode='nearest', cval=0)
    im_blurred = im
    # im_blurred = skimage.filters.gaussian(im_blurred, sigma=3)
    frangi_image = skimage.filters.frangi(im_blurred,
                                          alpha=.5,
                                          beta=.5,
                                          black_ridges=False)
    frangi_image_normalised = cv2.normalize(frangi_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    minLineLength = 1
    maxLineGap = 150

    binary_threshold = skimage.filters.threshold_otsu(frangi_image_normalised, nbins=256)
    BW = np.zeros_like(frangi_image_normalised)
    BW[frangi_image_normalised > binary_threshold * 0.5] = 1

    kernel = np.ones((13, 13),np.uint8)
    # closing = cv2.morphologyEx(BW, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(BW, kernel, iterations=1)
    kernel = np.ones((7, 7),np.uint8)
    errosion = cv2.erode(dilation, kernel, iterations=2)

    if visualise_intermediate_results:
        fig, axes = plt.subplots(1, 6, figsize=(60, 60))
        axes[0].imshow(im, cmap='gray')
        axes[1].imshow(im_blurred, cmap='gray')
        axes[2].imshow(frangi_image)
        axes[2].set_title(f'frangi image')
        axes[3].imshow(BW)
        axes[3].set_title('Binary image')
        axes[4].imshow(dilation)
        axes[4].set_title('Dilation')
        axes[5].imshow(errosion)
        axes[5].set_title('Erosion')

    return errosion


def create_vessel_mask(vessel_image,
                       n_iterations=1,
                       opening_kernel_size=5,
                       closing_kernel_size=13,
                       padding=cv2.BORDER_REPLICATE,
                       visualise_intermediate_steps=False):
    vessel_image = vessel_image.copy()
    vessel_image = cv2.copyMakeBorder(vessel_image, dst, top, bottom, left, right, borderType, value)

    try:
        opening_kernel_size = opening_kernel_size, opening_kernel_size
    except:
        assert isinstance(opening_kernel_size, tuple) and len(opening_kernel_size) == 2
        pass

    try:
        closing_kernel_size = closing_kernel_size, closing_kernel_size
    except:
        assert isinstance(closing_kernel_size, tuple) and len(closing_kernel_size) == 2
        pass

    opening_kernel = np.ones(opening_kernel_size, np.uint8)
    closing_kernel = np.ones(closing_kernel_size, np.uint8)
    for it in range(n_iterations):
        vessel_image_blurred = vessel_image
        # sample_vessel_image_blurred = skimage.filters.gaussian(sample_vessel_image_blurred, sigma=3)
        frangi_image = skimage.filters.frangi(vessel_image_blurred,
                                              alpha=.5, beta=.5, black_ridges=False)
        frangi_image_normalised = cv2.normalize(frangi_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        minLineLength = 1
        maxLineGap = 150


        binary_threshold = skimage.filters.threshold_otsu(frangi_image_normalised, nbins=256)
        BW = np.zeros_like(frangi_image_normalised)
        BW[frangi_image_normalised > binary_threshold * 0.05] = 1
        BW = np.uint8(BW)

        # Opening get's rid of small specles
        opening = cv2.morphologyEx(BW, cv2.MORPH_OPEN, opening_kernel)
        # Closing connects components
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, closing_kernel)

        if visualise_intermediate_steps:
            fig, axes = plt.subplots(1, 6, figsize=(60, 60))
            axes[0].imshow(vessel_image, cmap='gray')
            axes[0].set_title('Vessel image', fontsize=60)
            axes[1].imshow(vessel_image_blurred, cmap='gray')
            axes[1].set_title('Blurred image', fontsize=60)
            axes[2].imshow(frangi_image)
            axes[2].set_title('Frangi image', fontsize=60)
            axes[3].imshow(BW)
            axes[3].set_title('Binary image', fontsize=60)
            axes[4].imshow(opening)
            axes[4].set_title('Opening', fontsize=60)
            axes[5].imshow(closing)
            axes[5].set_title('Closing', fontsize=60)

        vessel_image = closing

    return vessel_image
