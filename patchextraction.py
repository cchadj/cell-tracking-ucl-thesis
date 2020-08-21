from typing import Dict

import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import numbers

from sharedvariables import VideoSession
from nearest_neighbors import get_nearest_neighbor, get_nearest_neighbor_distances


def get_random_points_on_circles(points,n_points_per_circle=1, max_radius=None, ret_radii=False):
    assert 1 <= n_points_per_circle <= 7, f'Points per circle must be between 1 and 7 not {n_points_per_circle}'

    neighbor_distances, _ = get_nearest_neighbor(points, 2)

    # dist_flat = neighbor_distances.flatten()
    # dist_flat = np.delete(dist_flat, np.where(dist_flat > (dist_flat.mean() + 0 * dist_flat.std()))[0])
    # mean_distance = dist_flat.mean()

    nnp = n_points_per_circle
    uniform_angle_displacements = np.array([0, np.math.pi, np.math.pi * 0.5, np.math.pi * 3 * 0.5, np.math.pi * 0.25,
                                            3 * np.math.pi * 0.25, 5 * np.math.pi * 0.25,
                                            7 * np.math.pi * 0.25]).squeeze()
    c = 0
    rxs = np.empty(len(points) * nnp, dtype=np.int32)
    rys = np.empty(len(points) * nnp, dtype=np.int32)
    radii = np.empty(len(points))
    for centre_point_idx, distances in enumerate(neighbor_distances):
        centre_point = points[centre_point_idx]

        r = np.ceil(np.min(distances) / 2)
        if r > max_radius:
            r = max_radius
        radii[centre_point_idx] = r
        cx, cy = centre_point

        angle = np.random.rand() * np.math.pi * 2
        random_angles = np.array(angle + uniform_angle_displacements).squeeze()

        rx = np.array([cx + np.cos(random_angles[:nnp]) * r], dtype=np.int32).squeeze()
        ry = np.array([cy + np.sin(random_angles[:nnp]) * r], dtype=np.int32).squeeze()

        rxs[c:c + nnp] = rx
        rys[c:c + nnp] = ry

        c += nnp

    if ret_radii:
        return rxs, rys, radii
    else:
        return rxs, rys


def get_random_points_on_rectangles(cx, cy, rect_size, n_points_per_rect=1):
    """ Get random points at patch perimeter.

    Args:
        n_points_per_rect: How many points to get on rectangle.
        cx: Rectangle center x component.
        cy: Rectangle center y component.
        rect_size (tuple, int): Rectangle height, width.

    Returns:
        Random points on the rectangles defined by centre cx, cy and height, width

    """
    np.random.seed(0)
    assert type(rect_size) is int or type(rect_size) is tuple
    if type(rect_size) is int:
        rect_size = rect_size, rect_size
    height, width = rect_size
    if type(cx) is int:
        cx, cy = np.array([cx]), np.array([cy])

    assert len(cx) == len(cy)

    rxs = np.zeros(0, dtype=np.int32)
    rys = np.zeros(0, dtype=np.int32)
    for i in range(n_points_per_rect):
        # random number to select edge along which to get the random point (up/down, left/right)
        # (0 or 1 displacement)
        r1 = np.random.rand(len(cx))

        # random  number for swapping displacements
        r2 = np.random.rand(len(cy))

        # Controls 0 or 1 displacement
        t = np.zeros(len(cx))
        t[r1 > 0.5] = 1

        dx = t * width
        dy = np.random.rand(len(cx)) * width

        # print('dx', dx)
        # print('dy', dy)

        dx[r2 > 0.5], dy[r2 > 0.5] = dy[r2 > 0.5], dx[r2 > 0.5]

        # print("r1", r1)
        # print("r2", r2)
        # print("t", t)
        #
        # print('dx', dx)
        # print('dy', dy)

        # if r2 > 0.5:
        #     dy, dx = dx, dy

        rx = (cx - width / 2) + dx
        ry = (cy - height / 2) + dy
        # print(rx.shape)
        # print(ry.shape)

        rxs = np.concatenate((rxs, rx))
        rys = np.concatenate((rys, ry))
        # print(rxs.shape)
        # print(rys.shape)

    return rxs, rys


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
    if type(patch_size) == int:
        patch_size = patch_size, patch_size
    height, width = patch_size
    return im[int(y - height / 2):int(y + height / 2),
           int(x - width / 2):int(x + width / 2), ...]


def extract_patches(img,
                    patch_size=(21, 21),
                    temporal_width=0,
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
        mask: A boolean mask HxW (same height and width as image).
            Only patches inside the mask are extracted.
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
    assert positions[:, 1].max() < image.shape[0] and positions[:, 0].max() < image.shape[1], \
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
            from matplotlib import pyplot as plt
            plt.imshow(mask)
            plt.scatter(x, y)
            print('hmm why?')

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
                # noinspection PyUnresolvedReferences
                rect = matplotlib.patches.Rectangle((x - patch_width / 2,
                                                     y - patch_height / 2),
                                                    patch_width, patch_height, linewidth=1,
                                                    edgecolor='r', facecolor='none')

                ax.imshow(np.squeeze(image), cmap='gray')
                ax.add_patch(rect)
                ax.scatter(x, y)
                ax.annotate(patch_count - 1, (x, y))
        else:
            print(f'Helllooo {patch.shape[:2]}')

    patches = patches[:patch_count, ...]
    return patches.squeeze()


def get_mask_bounds(mask):
    if np.any(mask[:, 0]):
        # edge case where there's a line at the first column in which case we say that the mean edge
        # pixel is the at the first column
        x_min = 0

        ys, xs = np.where(np.diff(mask))
        try:
            x_max = xs.max()
        except:
            print('hello')
    else:
        ys, xs = np.where(np.diff(mask))
        x_min, x_max = xs.min(), xs.max()

    y_min, y_max = ys.min(), ys.max()
    return x_min, x_max, y_min, y_max


class SessionPatchExtractor(object):
    _non_cell_positions: Dict[int, np.ndarray]
    _temporal_width: int

    def __init__(self,
                 session,
                 patch_size=21,
                 temporal_width=1,
                 n_negatives_per_positive=1,
                 negative_patch_extraction_radius=21,
                 random_rect_points=False,
                 ):
        """

        Args:
            session (VideoSession):  The video session to extract patches from
        """
        self.session = session

        self._cell_positions = {}
        self._non_cell_positions = {}

        assert type(patch_size) is int or type(patch_size) is tuple

        if type(patch_size) is tuple:
            self._patch_size = patch_size
        if type(patch_size) is int:
            self._patch_size = patch_size, patch_size

        self.temporal_width = temporal_width

        self.random_rect_points = random_rect_points
        self.frame_negative_search_radii = {}

        self._negative_patch_extraction_radius = negative_patch_extraction_radius

        self._n_negatives_per_positive = n_negatives_per_positive
        self._all_patches_oa790 = None
        self._all_patches_oa850 = None

        self._cell_patches_oa790 = None
        self._cell_patches_oa850 = None

        self._marked_cell_patches_oa790 = None
        self._marked_cell_patches_oa850 = None

        self._non_cell_patches_oa790 = None
        self._marked_non_cell_patches_oa790 = None

        self._cell_patches_oa790_at_frame = {}
        self._marked_cell_patches_oa790_at_frame = {}

        self._mixed_channel_cell_patches = None
        self._mixed_channel_non_cell_patches = None

        self._mixed_channel_marked_cell_patches = None
        self._mixed_channel_marked_non_cell_patches = None

        self._mixed_channel_cell_patches_at_frame = {}
        self._mixed_channel_non_cell_patches_at_frame = {}

        self._mixed_channel_marked_cell_patches_at_frame = {}
        self._mixed_channel_marked_non_cell_patches_at_frame = {}

        self._temporal_cell_patches_oa790 = None
        self._temporal_marked_cell_patches_oa790 = None

        self._temporal_non_cell_patches_oa790 = None
        self._temporal_marked_non_cell_patches_oa790 = None

        self._temporal_cell_patches_oa790_at_frame = {}
        self._temporal_marked_cell_patches_oa790_at_frame = {}

        self._temporal_non_cell_patches_oa790_at_frame = {}
        self._temporal_marked_non_cell_patches_oa790_at_frame = {}

        self._non_cell_patches_oa790_at_frame = {}
        self._marked_non_cell_patches_oa790_at_frame = {}

        self._reset_patches()

    @property
    def cell_positions(self):
        if len(self._cell_positions) == 0:
            for frame_idx, frame_cell_positions in self.session.cell_positions.items():
                mask = self.session.mask_frames_oa790[frame_idx]
                frame_cell_positions = self._delete_invalid_positions(frame_cell_positions, mask)
                self._cell_positions[frame_idx] = frame_cell_positions

        return self._cell_positions

    @property
    def non_cell_positions(self):
        if len(self._non_cell_positions) == 0:
            for frame_idx, frame_cell_positions in self.session.cell_positions.items():
                mask = self.session.mask_frames_oa790[frame_idx]
                cx, cy = frame_cell_positions[:, 0], frame_cell_positions[:, 1]

                frame_cell_positions = self._delete_invalid_positions(frame_cell_positions, mask=mask)
                if self.random_rect_points:
                    rx, ry = get_random_points_on_rectangles(cx, cy, rect_size=self.negative_patch_extraction_radius,
                                                             n_points_per_rect=self.n_negatives_per_positive)
                else:
                    rx, ry, radii = get_random_points_on_circles(frame_cell_positions,
                                                                 n_points_per_circle=self.n_negatives_per_positive,
                                                                 max_radius=self.negative_patch_extraction_radius,
                                                                 ret_radii=True,
                                                                 )
                    self.frame_negative_search_radii[frame_idx] = radii

                non_cell_positions = np.array([rx, ry]).T
                non_cell_positions = self._delete_invalid_positions(non_cell_positions, mask)
                non_cell_positions = np.int32(non_cell_positions)
                self._non_cell_positions[frame_idx] = non_cell_positions

        return self._non_cell_positions

    def _visualize_patch_extraction(self,
                                    session_frames,
                                    frame_idx_to_cell_patch_dict,
                                    frame_idx_to_non_cell_patch_dict,
                                    frame_idx=None, ax=None, figsize=(50, 40), linewidth=3, s=60):
        """ Shows the patch extraction on the first marked frame that has cell positions in it's csv.
        """
        from plotutils import plot_patch_rois_at_positions, plot_images_as_grid

        if frame_idx is None or frame_idx not in self.cell_positions:
            # Find the first frame index that has marked cell position and patches were extracted
            for frame_idx in list(self.cell_positions.keys()):
                # In case of temporal patches, if frame_idx < temporal width there aren't any patches for that frame
                if frame_idx in frame_idx_to_cell_patch_dict:
                    break

        frame = session_frames[frame_idx]

        cell_positions = self.cell_positions[frame_idx]
        cell_patches = frame_idx_to_cell_patch_dict[frame_idx]

        non_cell_positions = self.non_cell_positions[frame_idx]
        non_cell_patches = frame_idx_to_non_cell_patch_dict[frame_idx]

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.imshow(frame, cmap='gray')
        ax.set_title(f'Frame {frame_idx}', fontsize=20)

        plot_patch_rois_at_positions(cell_positions, self.patch_size, ax=ax,
                                     edgecolor='g', pointcolor='g', label='Cell positions', linewidth=linewidth)

        plot_patch_rois_at_positions(non_cell_positions, self.patch_size, ax=ax, label='Non cell patches',
                                     edgecolor=(1, 0, 0, .35), pointcolor='r', linewidth=linewidth * 0.75)

        if self.random_rect_points:
            plot_patch_rois_at_positions(cell_positions, self.negative_patch_extraction_radius, ax=ax,
                                         label='Negative patch extraction radius',
                                         edgecolor='y', pointcolor='gray', linewidth=linewidth)
        else:
            ax.scatter(non_cell_positions[..., 0], non_cell_positions[..., 1], c='r', s=s)

            for pos, r in zip(cell_positions, self.frame_negative_search_radii[frame_idx]):
                cx, cy = pos
                ax.add_artist(matplotlib.patches.Circle((cx, cy), r, fill=False, edgecolor='r', linewidth=linewidth, linestyle='--'))

        plot_images_as_grid(cell_patches, title='Cell patches')
        plot_images_as_grid(non_cell_patches, title='Non cell patches')
        ax.legend()

    def visualize_patch_extraction(self, **kwargs):
        self._visualize_patch_extraction(self.session.marked_frames_oa790,
                                         self.marked_cell_patches_oa790_at_frame,
                                         self.marked_non_cell_patches_oa790_at_frame,
                                         **kwargs)

    def visualize_temporal_patch_extraction(self, **kwargs):
        assert self.temporal_width == 1, 'Visualising temporal patches only works when temporal width is 1'
        self._visualize_patch_extraction(self.session.marked_frames_oa790,
                                         self.temporal_marked_cell_patches_oa790_at_frame,
                                         self.temporal_marked_non_cell_patches_oa790_at_frame,
                                         **kwargs)

    def visualize_mixed_channel_patch_extraction(self, **kwargs):
        self._visualize_patch_extraction(self.session.marked_frames_oa790,
                                         self.mixed_channel_marked_cell_patches_at_frame,
                                         self.mixed_channel_marked_non_cell_patches_at_frame,
                                         **kwargs)

    def _reset_positive_patches(self):
        self._cell_patches_oa790 = None
        self._marked_cell_patches_oa790 = None

        self._cell_patches_oa850 = None
        self._marked_cell_patches_oa850 = None

        self._temporal_cell_patches_oa790 = None
        self._temporal_marked_cell_patches_oa790 = None

        self._mixed_channel_cell_patches = None
        self._mixed_channel_marked_cell_patches = None

        self._cell_positions = {}

    def _reset_negative_patches(self):
        self._non_cell_patches_oa790 = None
        self._marked_non_cell_patches_oa790 = None

        self._temporal_non_cell_patches_oa790 = None
        self._temporal_marked_non_cell_patches_oa790 = None

        self._mixed_channel_non_cell_patches = None
        self._mixed_channel_marked_non_cell_patches = None

        self._non_cell_positions = {}

    def _reset_patches(self):
        self._reset_positive_patches()
        self._reset_negative_patches()

    def _reset_temporal_patches(self):
        self._temporal_cell_patches_oa790 = None
        self._temporal_non_cell_patches_oa790 = None
        self._temporal_marked_cell_patches_oa790 = None
        self._temporal_marked_non_cell_patches_oa790 = None

    @property
    def patch_size(self):
        return self._patch_size

    @patch_size.setter
    def patch_size(self, patch_size):
        assert type(patch_size) is int or type(patch_size) is tuple

        if type(patch_size) is tuple:
            self._patch_size = patch_size
        if type(patch_size) is int:
            self._patch_size = patch_size, patch_size

        self._reset_positive_patches()

    @property
    def negative_patch_extraction_radius(self):
        return self._negative_patch_extraction_radius

    @negative_patch_extraction_radius.setter
    def negative_patch_extraction_radius(self, radius):
        assert type(radius) == int
        self._negative_patch_extraction_radius = radius
        self._reset_negative_patches()

    @property
    def n_negatives_per_positive(self):
        return self._n_negatives_per_positive

    @n_negatives_per_positive.setter
    def n_negatives_per_positive(self, n):
        self._n_negatives_per_positive = n
        self._reset_negative_patches()

    @property
    def all_patches_oa850(self):
        if self._all_patches_oa850 is None:
            self._all_patches_oa850 = np.zeros((0, *self._patch_size), dtype=self.session.frames_oa850.dtype)
            for frame in self.session.frames_oa850:
                cur_frame_patches = extract_patches(frame, patch_size=self._patch_size)
                self._all_patches_oa850 = np.concatenate((self._all_patches_oa850, cur_frame_patches), axis=0)

        return self._all_patches_oa850

    @property
    def temporal_width(self):
        return self._temporal_width

    @temporal_width.setter
    def temporal_width(self, width):
        assert type(width) is int, f'Temporal width of patch should be an integer not {type(width)}'
        self._temporal_width = width
        self._reset_temporal_patches()

    def _delete_invalid_positions(self,
                                  positions,
                                  mask=None):
        _, frame_height, frame_width = self.session.frames_oa790.shape

        # remove positions whose patches get outside the frame
        positions = np.int32(positions)
        positions = np.delete(positions, np.where(positions[:, 0] - np.ceil(self._patch_size[1] / 2) < 0)[0], axis=0)
        positions = np.delete(positions, np.where(positions[:, 1] - np.ceil(self._patch_size[0] / 2) < 0)[0], axis=0)
        positions = np.delete(positions,
                              np.where(positions[:, 0] + np.ceil(self._patch_size[1] / 2) >= frame_width - 1)[0],
                              axis=0)
        positions = np.delete(positions,
                              np.where(positions[:, 1] + np.ceil(self._patch_size[0] / 2) >= frame_height - 1)[0],
                              axis=0)

        if mask is not None and not np.all(mask):
            # remove positions whose patches get outside the mask
            x_min, x_max, y_min, y_max = get_mask_bounds(mask)

            # The maximum difference between the mask maximum and minimum on the side of the mask is about 20px.
            x_max, y_max = x_max - 20, y_max - 20

            mask_ys, mask_xs = np.where(mask)
            mask_positions = np.int32(np.array((mask_xs, mask_ys)).T)

            positions = np.delete(positions, np.where(positions[:, 0] - np.ceil(self._patch_size[1] / 2) < x_min)[0],
                                  axis=0)
            positions = np.delete(positions, np.where(positions[:, 1] - np.ceil(self._patch_size[0] / 2) < y_min)[0],
                                  axis=0)
            positions = np.delete(positions, np.where(positions[:, 0] + np.ceil(self._patch_size[1] / 2) >= x_max)[0],
                                  axis=0)
            positions = np.delete(positions, np.where(positions[:, 1] + np.ceil(self._patch_size[0] / 2) >= y_max)[0],
                                  axis=0)

        return positions

    def _extract_patches_at_positions(self, session_frames, positions, frame_idx_to_patch_dict=None, masks=None):
        """ Extracts patches for each position and frame in positions dictionary

        Args:
            session_frames: The frames to extract the patches from
            positions: A DICTIONARY frame index -> N x 2  of positions per for frames at each frame index
            frame_idx_to_patch_dict:
                A dictionary to get the patches from that frame.
                Pass an empty dictionary to fill.
            masks:
                The mask of each frame. If position + patch size goes outside the mask the position is discarded.

        Returns:
            Nx patch height x patch width (x C) patches
        """
        patches = np.zeros((0, *self._patch_size), dtype=session_frames.dtype)

        for frame_idx, frame_cell_positions in positions.items():
            if frame_idx >= len(session_frames):
                break
            frame = session_frames[frame_idx]
            mask = None
            if masks is not None:
                mask = masks[frame_idx]

            frame_cell_positions = self._delete_invalid_positions(frame_cell_positions, mask)

            cur_frame_cell_patches = extract_patches_at_positions(frame, frame_cell_positions, mask=mask,
                                                                  patch_size=self._patch_size)
            if frame_idx_to_patch_dict is not None:
                frame_idx_to_patch_dict[frame_idx] = cur_frame_cell_patches

            patches = np.concatenate((patches, cur_frame_cell_patches), axis=0)

        return patches

    @property
    def cell_patches_oa790(self):
        if self._cell_patches_oa790 is None:
            self._cell_patches_oa790 = self._extract_patches_at_positions(self.session.frames_oa790,
                                                                          self.cell_positions,
                                                                          self._cell_patches_oa790_at_frame,
                                                                          masks=self.session.mask_frames_oa790)
        return self._cell_patches_oa790

    @property
    def marked_cell_patches_oa790(self):
        if self._marked_cell_patches_oa790 is None:
            self._marked_cell_patches_oa790 = self._extract_patches_at_positions(self.session.marked_frames_oa790,
                                                                                 self.cell_positions,
                                                                                 self._marked_cell_patches_oa790_at_frame,
                                                                                 masks=self.session.mask_frames_oa790)
        return self._marked_cell_patches_oa790

    @property
    def non_cell_patches_oa790(self):
        if self._non_cell_patches_oa790 is None:
            self._non_cell_patches_oa790 = self._extract_patches_at_positions(
                self.session.frames_oa790,
                self.non_cell_positions,
                frame_idx_to_patch_dict=self._non_cell_patches_oa790_at_frame,
                masks=self.session.mask_frames_oa790
            )
        return self._non_cell_patches_oa790

    @property
    def marked_non_cell_patches_oa790(self):
        if self._marked_non_cell_patches_oa790 is None:
            self._marked_non_cell_patches_oa790 = self._extract_patches_at_positions(
                self.session.marked_frames_oa790,
                self.non_cell_positions,
                frame_idx_to_patch_dict=self._marked_non_cell_patches_oa790_at_frame,
                masks=self.session.mask_frames_oa790)

        return self._marked_non_cell_patches_oa790

    @property
    def cell_patches_oa790_at_frame(self):
        # force the cell patches creation ot fill the dict
        tmp = self.cell_patches_oa790

        return self._cell_patches_oa790_at_frame

    @property
    def marked_cell_patches_oa790_at_frame(self):
        # force the cell patches creation to fill the dict
        tmp = self.marked_cell_patches_oa790

        return self._marked_cell_patches_oa790_at_frame

    @property
    def non_cell_patches_oa790_at_frame(self):
        # for the cell patches creation ot fill the dict
        tmp = self.non_cell_patches_oa790
        return self._non_cell_patches_oa790_at_frame

    @property
    def marked_non_cell_patches_oa790_at_frame(self):
        # for the cell patches creation ot fill the dict
        tmp = self.marked_non_cell_patches_oa790
        return self._marked_non_cell_patches_oa790_at_frame

    @property
    def all_patches_oa790(self):
        if self._all_patches_oa790 is None:
            self._all_patches_oa790 = np.zeros((0, *self._patch_size), dtype=self.session.frames_oa790.dtype)
            for frame in self.session.frames_oa790:
                cur_frame_patches = extract_patches(frame, patch_size=self._patch_size)
                self._all_patches_oa790 = np.concatenate((self._all_patches_oa790, cur_frame_patches), axis=0)

        return self._all_patches_oa790

    def _extract_temporal_patches(self,
                                  session_frames,
                                  positions,
                                  frame_idx_to_temporal_patch_dict,
                                  masks=None):
        n_channels = 2 * self.temporal_width + 1
        temporal_patches = np.empty((0, *self._patch_size, n_channels), dtype=np.uint8)

        _, frame_height, frame_width = session_frames.shape
        for frame_idx, frame_positions in positions.items():
            if frame_idx >= len(session_frames):
                break

            if frame_idx < self.temporal_width or frame_idx > len(session_frames) - self.temporal_width:
                continue

            if masks is not None:
                mask = masks[frame_idx]

            frame_positions = self._delete_invalid_positions(frame_positions, mask)
            frame_temporal_patches = np.empty((len(frame_positions), *self._patch_size, n_channels), dtype=np.uint8)

            for i, frame in enumerate(
                    session_frames[frame_idx - self.temporal_width:frame_idx + self.temporal_width + 1]):
                extract_patches_at_positions(frame,
                                             frame_positions,
                                             mask=mask,
                                             patch_size=self._patch_size)
                frame_positions = self._delete_invalid_positions(frame_positions, mask)

                frame_temporal_patches[..., i] = extract_patches_at_positions(frame,
                                                                              frame_positions,
                                                                              mask=mask,
                                                                              patch_size=self._patch_size)
            frame_idx_to_temporal_patch_dict[frame_idx] = frame_temporal_patches
            temporal_patches = np.concatenate((temporal_patches, frame_temporal_patches), axis=0)

        return temporal_patches

    @property
    def temporal_cell_patches_oa790(self):
        if self._temporal_cell_patches_oa790 is None:
            self._temporal_cell_patches_oa790_at_frame = {}
            self._temporal_cell_patches_oa790 = self._extract_temporal_patches(
                self.session.frames_oa790,
                self.cell_positions,
                frame_idx_to_temporal_patch_dict=self._temporal_cell_patches_oa790_at_frame,
                masks=self.session.mask_frames_oa790)
        return self._temporal_cell_patches_oa790

    @property
    def temporal_marked_cell_patches_oa790(self):
        if self._temporal_marked_cell_patches_oa790 is None:
            self._temporal_marked_cell_patches_oa790_at_frame = {}
            self._temporal_marked_cell_patches_oa790 = self._extract_temporal_patches(
                self.session.marked_frames_oa790,
                self.cell_positions,
                frame_idx_to_temporal_patch_dict=self._temporal_marked_cell_patches_oa790_at_frame,
                masks=self.session.mask_frames_oa790)
        return self._temporal_marked_cell_patches_oa790

    @property
    def temporal_non_cell_patches_oa790(self):
        if self._temporal_non_cell_patches_oa790 is None:
            self._temporal_non_cell_patches_oa790_at_frame = {}
            self._temporal_non_cell_patches_oa790 = self._extract_temporal_patches(
                self.session.frames_oa790,
                self.non_cell_positions,
                frame_idx_to_temporal_patch_dict=self._temporal_non_cell_patches_oa790_at_frame,
                masks=self.session.mask_frames_oa790)
        return self._temporal_non_cell_patches_oa790

    @property
    def temporal_marked_non_cell_patches_oa790(self):
        if self._temporal_marked_non_cell_patches_oa790 is None:
            self._temporal_marked_non_cell_patches_oa790_at_frame = {}
            self._temporal_marked_non_cell_patches_oa790 = self._extract_temporal_patches(
                self.session.marked_frames_oa790,
                self.non_cell_positions,
                frame_idx_to_temporal_patch_dict=self._temporal_marked_non_cell_patches_oa790_at_frame,
                masks=self.session.mask_frames_oa790,
            )
        return self._temporal_marked_non_cell_patches_oa790

    @property
    def temporal_cell_patches_oa790_at_frame(self):
        tmp = self.temporal_cell_patches_oa790
        return self._temporal_cell_patches_oa790_at_frame

    @property
    def temporal_marked_cell_patches_oa790_at_frame(self):
        tmp = self.temporal_marked_cell_patches_oa790
        return self._temporal_marked_cell_patches_oa790_at_frame

    @property
    def temporal_non_cell_patches_oa790_at_frame(self):
        tmp = self.temporal_non_cell_patches_oa790
        return self._temporal_non_cell_patches_oa790_at_frame

    @property
    def temporal_marked_non_cell_patches_oa790_at_frame(self):
        tmp = self.temporal_marked_non_cell_patches_oa790
        return self._temporal_marked_non_cell_patches_oa790_at_frame

    @property
    def all_temporal_patches_oa790(self):
        # TODO: extract all temporal patches, from 2nd frame to the penultimate frame.
        raise NotImplemented

    def _extract_mixed_channel_cell_patches(self,
                                            frames_oa790,
                                            positions,
                                            frame_idx_to_patch_dict=None
                                            ):
        from measure_velocity import ImageRegistator
        frames_oa850 = self.session.frames_oa850
        frames_confocal = self.session.frames_confocal
        masks_oa850 = self.session.mask_frames_oa850
        ir = ImageRegistator(source=self.session.vessel_mask_oa850, target=self.session.vessel_mask_confocal)
        ir.register_vertically()

        # Register all oa850 frames
        registered_frames_oa850 = np.empty_like(frames_oa850)
        registered_mask_frames_oa850 = np.empty_like(masks_oa850)
        for i, (frame, mask) in enumerate(zip(frames_oa850, masks_oa850)):
            registered_frames_oa850[i] = ir.apply_registration(frame)
            registered_mask_frames_oa850[i] = ir.apply_registration(mask)

        frame_idx_to_confocal_patches = {}
        frame_idx_to_oa790_patches = {}
        frame_idx_to_oa850_patches = {}
        cell_patches_confocal = self._extract_patches_at_positions(
            frames_confocal,
            positions,
            frame_idx_to_patch_dict=frame_idx_to_confocal_patches,
            masks=registered_mask_frames_oa850
        )
        cell_patches_oa790 = self._extract_patches_at_positions(
            frames_oa790,
            positions,
            frame_idx_to_patch_dict=frame_idx_to_oa790_patches,
            masks=registered_mask_frames_oa850
        )
        cell_patches_oa850 = self._extract_patches_at_positions(
            registered_frames_oa850,
            positions,
            frame_idx_to_patch_dict=frame_idx_to_oa850_patches,
            masks=registered_mask_frames_oa850
        )
        assert len(cell_patches_oa790) == len(cell_patches_oa850), 'Not the same of patches extracted'

        if frame_idx_to_patch_dict is not None:
            for frame_idx in frame_idx_to_oa850_patches.keys():
                patches_confocal = frame_idx_to_confocal_patches[frame_idx]
                patches_oa790 = frame_idx_to_oa790_patches[frame_idx]
                patches_oa850 = frame_idx_to_oa850_patches[frame_idx]

                mixed_channel_patches = np.empty([*patches_confocal.shape, 3], dtype=frames_oa790.dtype)
                mixed_channel_patches[..., 0] = patches_confocal
                mixed_channel_patches[..., 1] = patches_oa790
                mixed_channel_patches[..., 2] = patches_oa850

                frame_idx_to_patch_dict[frame_idx] = mixed_channel_patches

        mixed_channel_cell_patches = np.empty([*cell_patches_oa790.shape, 3], dtype=frames_oa790.dtype)
        mixed_channel_cell_patches[..., 0] = cell_patches_confocal
        mixed_channel_cell_patches[..., 1] = cell_patches_oa790
        mixed_channel_cell_patches[..., 2] = cell_patches_oa850

        return mixed_channel_cell_patches

    @property
    def mixed_channel_cell_patches(self):
        """
        Returns:
            3 channel patches from the confocal video, the oa790 video and the oa850 video.
            The first channel is from the confocal
            second channel is oa790,
            third channel is oa850,
        """
        if self._mixed_channel_cell_patches is None:
            self._mixed_channel_cell_patches_at_frame = {}
            self._mixed_channel_cell_patches = self._extract_mixed_channel_cell_patches(
                self.session.frames_oa790,
                self.cell_positions,
                frame_idx_to_patch_dict=self._mixed_channel_cell_patches_at_frame
            )
        return self._mixed_channel_cell_patches

    @property
    def mixed_channel_marked_cell_patches(self):
        if self._mixed_channel_marked_cell_patches is None:
            self._mixed_channel_marked_cell_patches_at_frame = {}
            self._mixed_channel_marked_cell_patches = self._extract_mixed_channel_cell_patches(
                self.session.marked_frames_oa790,
                self.cell_positions,
                frame_idx_to_patch_dict=self._mixed_channel_marked_cell_patches_at_frame,
            )

        return self._mixed_channel_marked_cell_patches

    @property
    def mixed_channel_non_cell_patches(self):
        if self._mixed_channel_non_cell_patches is None:
            self._mixed_channel_non_cell_patches_at_frame = {}
            self._mixed_channel_non_cell_patches = self._extract_mixed_channel_cell_patches(
                self.session.frames_oa790,
                self.non_cell_positions,
                frame_idx_to_patch_dict=self._mixed_channel_non_cell_patches_at_frame,
            )
        return self._mixed_channel_non_cell_patches

    @property
    def mixed_channel_marked_non_cell_patches(self):
        if self._mixed_channel_marked_non_cell_patches is None:
            self._mixed_channel_marked_non_cell_patches_at_frame = {}
            self._mixed_channel_marked_non_cell_patches = self._extract_mixed_channel_cell_patches(
                self.session.marked_frames_oa790,
                self.non_cell_positions,
                frame_idx_to_patch_dict=self._mixed_channel_marked_non_cell_patches_at_frame,
            )
        return self._mixed_channel_marked_non_cell_patches

    @property
    def mixed_channel_cell_patches_at_frame(self):
        tmp = self.mixed_channel_cell_patches
        return self._mixed_channel_cell_patches_at_frame

    @property
    def mixed_channel_marked_cell_patches_at_frame(self):
        tmp = self.mixed_channel_marked_cell_patches
        return self._mixed_channel_marked_cell_patches_at_frame

    @property
    def mixed_channel_non_cell_patches_at_frame(self):
        tmp = self.mixed_channel_non_cell_patches
        return self._mixed_channel_non_cell_patches_at_frame

    @property
    def mixed_channel_marked_non_cell_patches_at_frame(self):
        tmp = self.mixed_channel_marked_non_cell_patches
        return self._mixed_channel_marked_non_cell_patches_at_frame


if __name__ == '__main__':
    from sharedvariables import get_video_sessions
    from plotutils import plot_images_as_grid

    video_sessions = get_video_sessions(should_have_marked_cells=True)
    vs = video_sessions[0]

    patch_extractor = SessionPatchExtractor(vs, patch_size=21, temporal_width=1, n_negatives_per_positive=7)

    plot_images_as_grid(patch_extractor.temporal_cell_patches_oa790[:10], title='Temporal cell patches temporal width 1')
    plot_images_as_grid(patch_extractor.temporal_marked_cell_patches_oa790[:10])

    plot_images_as_grid(patch_extractor.temporal_non_cell_patches_oa790[:10], title='Temporal non cell patches temporal width 1')
    plot_images_as_grid(patch_extractor.temporal_marked_non_cell_patches_oa790[:10])
