import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numbers
from sharedvariables import VideoSession


def get_patch(im, x, y, patch_size):
    """ Get a patch from image

    Args:
        im: The image to get the patch from. HxW or HxWxC
        x (int): The patch center x component (left to right)
        y (int): The patch center y component (top to bot)
        patch_size (tuple): The patch height, width

    Returns:
        height x width x C
    """
    height, width = patch_size
    return im[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2), ...]


def extract_patches(img,
                    patch_size=(21, 21),
                    padding='valid'
                    ):
    """
    Extract patches around every pixel of the image.

    To get the row, col coordinates of the ith patch do:

    rows, cols = np.unravel_index(np.arange(patches.shape[0]), frame.shape[0:2])
    # ith patch coordinates
    row, col = rows[i], cols[i]

    Arguments:
        img (array): HxWxC or HxW(grayscale) image.
        patch_size (tuple): The patch height, width.
        padding:
            'valid' If you want only patches that are entirely inside the image.
                If not valid then one of : [
                    cv2.BORDER_REPLICATE,
                    cv2.BORDER_REFLECT,
                    cv2.BORDER_WRAP,
                    cv2.BORDER_ISOLATED
                ]

    Returns:
        (np.array):  NxHxWxC
    """
    if isinstance(patch_size, numbers.Number):
        patch_height, patch_width = patch_size, patch_size
    elif isinstance(patch_size, tuple):
        patch_height, patch_width = patch_size

    if padding != 'valid':
        padding_height, padding_width = int((patch_height - 1) / 2), int((patch_width - 1) / 2)
        img = cv2.copyMakeBorder(img,
                                 padding_height,
                                 padding_height,
                                 padding_width,
                                 padding_width,
                                 padding)
    kernel_height, kernel_width = patch_height, patch_width

    inp = torch.from_numpy(img)

    if len(inp.shape) == 3:
        inp = inp.permute(-1, 0, 1)
    elif len(inp.shape) == 2:
        inp = inp[None, ...]
    inp = inp[None, ...]

    # print("Inp.shape", inp.shape)
    patches = inp.unfold(2, kernel_height, 1).unfold(3, kernel_width, 1)
    # Shape -> 1 x 1 x H x W x Hpatch x Wpatch
    # print("Patches shape 1", patches.shape)

    patches = patches.permute(2, 3, 1, -2, -1, 0)[..., 0]
    # Shape -> H x W x C x Hpatch x Wpatch
    # print("Patches shape 2", patches.shape)
    # patch = patches[80, 28, ...].cpu().numpy()
    # patch = patch.transpose(1, 2, 0)
    # plt.imshow(patch.squeeze())

    patches = patches.contiguous().flatten(0, 1)
    # print("Patches shape 3", patches.shape)
    # Shape -> H*W x C x Hpatch x Wpatch

    patches = patches.permute(0, 2, 3, 1)
    #  Output Shape -> H*W x Hpatch x Wpatch x C
    # print("Patches output shape", patches.shape)

    # ------ To get patch at row=y col=x
    # x, y = 131, 63
    # cone_patch_index = np.ravel_multi_index([[y], [x]], dims=unpadded_image_shape).item()
    # print(cone_patch_index)
    # patch = patches[cone_patch_index, ...].cpu().numpy()
    # plt.imshow(patch.squeeze())

    return patches.cpu().numpy()


def extract_patches_at_positions(image,
                                 positions,
                                 patch_size=(21, 21),
                                 padding='valid',
                                 mask=None,
                                 visualize_patches=False):
    """ Extract patches from images at positions

    Arguments:
        image: HxW or HxWxC image
        positions: shape:(2,) list of (x, y) positions. x left to right, y top to bottom
        patch_size (tuple):  Size of each patch.
        padding:
            'valid' If you want only patches that are entirely inside the image.
            If not valid then one of : [
                cv2.BORDER_REPLICATE,
                cv2.BORDER_REFLECT,
                cv2.BORDER_WRAP,
                cv2.BORDER_ISOLATED
            ]

    Returns:
        (np.array): NxHxWxC patches.
    """
    assert 2 <= len(image.shape) <= 3
    if type(patch_size) is tuple:
        patch_height, patch_width = patch_size
    elif type(patch_size) is int:
        patch_height, patch_width = patch_size, patch_size
    else:
        raise TypeError('Patch_size must be int or type. Type given: ', type(patch_size))

    padding_height, padding_width = 0, 0
    n_patches_max = positions.shape[0]

    if padding != 'valid':
        padding_height, padding_width = int((patch_height - 1) / 2), int((patch_width - 1) / 2)
        image = cv2.copyMakeBorder(image,
                                   padding_height,
                                   padding_height,
                                   padding_width,
                                   padding_width,
                                   padding)
    assert positions[:, 1].max() < image.shape[0] and positions[:, 0].max() < image.shape[1],\
        'Position coordinates must not go outside of image boundaries.'
    if len(image.shape) == 2:
        n_channels = 1
        image = image[:, :, np.newaxis]
    elif len(image.shape) == 3:
        n_channels = image.shape[-1]

    patches = np.zeros_like(image, shape=[n_patches_max, patch_height, patch_width, n_channels])

    if visualize_patches:
        fig, ax = plt.subplots(1, figsize=(20, 10))

    if mask is None:
        mask = np.ones_like(image, dtype=np.bool8)

    patch_count = 0
    for x, y in np.int32(positions):
        if not mask[y, x]:
            continue
        # Offset to adjust for padding
        x, y = x + padding_width, y + padding_height
        patch = get_patch(image, x, y, patch_size)

        #  If patch shape is same as patch size then it means that it's valid
        if patch.shape[:2] == (patch_height, patch_width):
            # print("Heyo ", patches[patch_count].shape)
            patches[patch_count, :, :, :] = patch
            patch_count += 1

            if visualize_patches:
                # Rectangle ( xy -> bottom and left rect coords, )
                rect = matplotlib.patches.Rectangle((x - patch_width / 2,
                                                     y - patch_height / 2),
                                                    patch_width, patch_height, linewidth=1,
                                                    edgecolor='r', facecolor='none')

                ax.imshow(np.squeeze(image), cmap='gray')
                ax.add_patch(rect)
                ax.scatter(x, y)
                ax.annotate(patch_count - 1, (x, y))

    patches = patches[:patch_count, ...]
    return patches.squeeze()


class SessionPatchExtractor(object):
    def __init__(self,
                 session,
                 patch_size=21,
                 n_negatives_per_positives=3,):
        """

        Args:
            session (VideoSession):  The video session to extract patches from
        """
        self.session = session
        assert type(patch_size) is int or type(patch_size) is tuple

        if type(patch_size) is tuple:
            self.patch_size = patch_size
        if type(patch_size) is int:
            self.patch_size = patch_size, patch_size

        self._all_patches_oa790 = None
        self._all_patches_oa850 = None

        self._cell_patches_oa790 = None
        self._cell_patches_oa850 = None

        self._marked_cell_patches_oa790 = None
        self._marked_cell_patches_oa850 = None

    @property
    def all_patches_oa790(self):
        if self._all_patches_oa790 is None:
            self._all_patches_oa790 = np.zeros((0, *self.patch_size), dtype=self.session.frames_oa790.dtype)
            for frame in self.session.frames_oa790:
                cur_frame_patches = extract_patches(frame, patch_size=self.patch_size)
                self._all_patches_oa790 = np.concatenate((self._all_patches_oa790, cur_frame_patches), axis=0)

        return self._all_patches_oa790

    @property
    def all_patches_oa850(self):
        if self._all_patches_oa850 is None:
            self._all_patches_oa850 = np.zeros((0, *self.patch_size), dtype=self.session.frames_oa850.dtype)
            for frame in self.session.frames_oa850:
                cur_frame_patches = extract_patches(frame, patch_size=self.patch_size)
                self._all_patches_oa850 = np.concatenate((self._all_patches_oa850, cur_frame_patches), axis=0)

        return self._all_patches_oa850

    def _extract_cell_patches(self, session_frames, cell_positions):
        cell_patches = np.zeros((0, *self.patch_size), dtype=session_frames.dtype)
        # note, frame_idx in the csv files is 1 indexed while python is 0 indexed => do frame_idx - 1 to get frame
        for frame_idx, cell_positions in cell_positions.items():
            frame = session_frames[frame_idx - 1]
            cur_frame_cell_patches = extract_patches_at_positions(frame, cell_positions, patch_size=self.patch_size)
            cell_patches = np.concatenate((cell_patches, cur_frame_cell_patches), axis=0)

        return cell_patches

    def _extract_cell_patches_single_frame(self, session_frames, cell_positions, frame_idx):
        # note, frame_idx in the csv files is 1 indexed while python is 0 indexed => do frame_idx - 1 to get frame
        frame = session_frames[frame_idx - 1]
        return extract_patches_at_positions(frame, cell_positions[frame_idx], patch_size=self.patch_size)

    def cell_patches_oa790_at_frame(self, frame_idx):
        return self._extract_cell_patches_single_frame(self.session.frames_oa790, self.session.cell_positions, frame_idx)

    @property
    def cell_patches_oa790(self):
        if self._cell_patches_oa790 is None:
            self._cell_patches_oa790 = self._extract_cell_patches(self.session.frames_oa790, self.session.cell_positions)
        return self._cell_patches_oa790

    @property
    def marked_cell_patches_oa790(self):
        if self._marked_cell_patches_oa790 is None:
            self._marked_cell_patches_oa790 = self._extract_cell_patches(self.session.marked_frames_oa790,
                                                                         self.session.cell_positions)
        return self._marked_cell_patches_oa790

    @property
    def cell_patches_oa850(self):
        raise NotImplementedError('Cell patches only for oa790 channel currently')
        # if self._cell_patches_oa850 is None:
        #     self._cell_patches_oa850 = self._extract_cell_patches(self.session.frames_oa850,
        #                                                           self.session.cell_positions)
        # return self._cell_patches_oa850

    @property
    def marked_cell_patches_oa850(self):
        raise NotImplementedError('Cell patches only for oa790 channel currently')
        # if self._marked_cell_patches_oa850 is None:
        #     self._marked_cell_patches_oa850 = self._extract_cell_patches(self.session.marked_frames_oa850,
        #                                                                  self.session.cell_positions)
        # return self._marked_cell_patches_oa850
