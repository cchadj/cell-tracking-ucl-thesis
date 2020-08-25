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


def detect_vessels(im, visualise_intermediate_results=False):
    # im_blurred = skimage.filters.median(im,  mode='nearest', cval=0)
    im_blurred = im
    # im_blurred = skimage.filters.gaussian(im_blurred, sigma=3)
    frangi_image = skimage.filters.frangi(im_blurred,
                                          alpha=.5,
                                          beta=.5,
                                          black_ridges=False)
    frangi_image_normalised = cv2.normalize(frangi_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    binary_threshold = skimage.filters.threshold_otsu(frangi_image_normalised, nbins=256)
    BW = np.zeros_like(frangi_image_normalised)
    BW[frangi_image_normalised > binary_threshold * 0.5] = 1

    kernel = np.ones((13, 13), np.uint8)
    # closing = cv2.morphologyEx(BW, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(BW, kernel, iterations=1)
    kernel = np.ones((7, 7), np.uint8)
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
    vessel_image = cv2.copyMakeBorder(vessel_image,
                                      padding_size,
                                      padding_size,
                                      padding_size,
                                      padding_size,
                                      padding,
                                      value=padding_value)

    try:
        opening_kernel_size = opening_kernel_size, opening_kernel_size
    except:
        assert isinstance(opening_kernel_size, tuple) and len(opening_kernel_size) == 2

    try:
        closing_kernel_size = closing_kernel_size, closing_kernel_size
    except:
        assert isinstance(closing_kernel_size, tuple) and len(closing_kernel_size) == 2

    opening_kernel = np.ones(opening_kernel_size, np.uint8)
    closing_kernel = np.ones(closing_kernel_size, np.uint8)
    for it in range(n_iterations):
        vessel_image_blurred = vessel_image
        # sample_vessel_image_blurred = skimage.filters.gaussian(sample_vessel_image_blurred, sigma=3)
        frangi_image = skimage.filters.frangi(vessel_image_blurred,
                                              alpha=.5, beta=.5, black_ridges=False)
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

    return vessel_image


def create_vessel_image(frames,
                        masks=None,
                        left_crop=7,
                        method='j_tam',
                        adapt_hist=True,
                        sigma=0.75):
    # 20. J. Tam, J. A. Martin, and A. Roorda, “Noninvasive visualization and analysis of parafoveal capillaries in humans,”
    # Invest. Ophthalmol. Vis. Sci. 51(3), 1691–1698 (2010).
    from skimage import exposure, filters

    assert method in [None, 'j_tam', 'de_castro']

    if frames.dtype == np.uint8:
        frames = frames / 255
    else:
        frames = frames.copy()

    if not np.ma.is_masked(frames):
        if masks is None:
            masks = np.ones_like(frames, np.bool8)
        else:
            masks = crop_mask(masks, left_crop)

        frames = np.ma.masked_array(frames, ~masks)

    if sigma >= 0.125:
        masks = ~frames.mask
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            # make everything outside of mask the mean of the image
            frame[~mask] = frame.mean()
            frames[i, ...] = filters.gaussian(frame, sigma)

    if method == 'de_castro':
        # We invert the mask because True values mean that the values are masked and therefor invalid
        # https://numpy.org/doc/stable/reference/maskedarray.generic.html
        for i, frame in enumerate(frames):
            frames[i] /= np.ma.mean(frame)

        m = np.ma.mean(frames, axis=0)
        for i, frame in enumerate(frames):
            frames[i] /= m

        final_processed_frames = frames
    elif method == 'j_tam':
        # Create division frames by dividing consecutive frames (last frame remains unprocessed)
        division_frames = frames
        for j in range(len(frames) - 1):
            division_frames[j] /= frames[j + 1]
        division_frames = division_frames[:-1]

        # Create multiframe frames by averaging consecutive division frames (last frame remains unprocessed)
        multiframe_div_frames = division_frames
        for j in range(len(division_frames) - 1):
            multiframe_div_frames[j] = (division_frames[j] + division_frames[j + 1]) / 2
            if adapt_hist:
                multiframe_div_frames[j] = exposure.equalize_adapthist(
                    normalize_data(multiframe_div_frames[j].filled(multiframe_div_frames[j].mean()), (0, 1)))

        final_processed_frames = multiframe_div_frames[:-1]
    elif method is None:
        final_processed_frames = frames

    return skimage.exposure.equalize_adapthist(final_processed_frames.std(0))
