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
