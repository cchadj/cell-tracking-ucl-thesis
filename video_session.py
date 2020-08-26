import warnings
from typing import Any, List

import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from os.path import basename

from videoutils import get_frames_from_video
from vesseldetection import create_vessel_image, binarize_vessel_image
from imageprosessing import ImageRegistator


class VideoSession(object):
    def __init__(self, video_filename):
        from sharedvariables import find_filename_of_same_source, files_of_same_source
        from sharedvariables import marked_video_oa790_files, marked_video_oa850_files
        from sharedvariables import csv_cell_cords_oa790_filenames
        from sharedvariables import unmarked_video_oa790_filenames, unmarked_video_oa850_filenames
        from sharedvariables import unmarked_video_confocal_filenames, mask_video_oa790_files
        from sharedvariables import mask_video_oa850_files, mask_video_confocal_files
        from sharedvariables import vessel_mask_confocal_files, vessel_mask_oa790_files, vessel_mask_oa850_files
        from sharedvariables import std_image_confocal_files, std_image_oa790_files, std_image_oa850_files

        self.is_registered = '_reg_' in video_filename

        vid_marked_790_filename = find_filename_of_same_source(video_filename, marked_video_oa790_files)
        vid_marked_850_filename = find_filename_of_same_source(video_filename, marked_video_oa850_files)

        self._cell_position_csv_files = [csv_file for csv_file
                                         in csv_cell_cords_oa790_filenames
                                         if files_of_same_source(csv_file, video_filename)]

        self.has_marked_video = vid_marked_790_filename != '' or vid_marked_850_filename != ''
        self.has_marked_cells = len(self._cell_position_csv_files) > 0

        self.subject_number = -1
        self.session_number = -1
        for string in video_filename.split('_'):
            if 'Subject' in string:
                self.subject_number = int(re.search(r'\d+', string).group())
            if 'Session' in string:
                self.session_number = int(re.search(r'\d+', string).group())

        self._validation_frame_idx = None

        self.video_file = video_filename
        self.video_oa790_file = find_filename_of_same_source(video_filename, unmarked_video_oa790_filenames)
        self.video_oa850_file = find_filename_of_same_source(video_filename, unmarked_video_oa850_filenames)
        self.video_confocal_file = find_filename_of_same_source(video_filename, unmarked_video_confocal_filenames)

        self.marked_video_oa790_file = vid_marked_790_filename
        self.marked_video_oa850_file = vid_marked_850_filename

        self.mask_video_oa790_file = find_filename_of_same_source(video_filename, mask_video_oa790_files)
        self.mask_video_oa850_file = find_filename_of_same_source(video_filename, mask_video_oa850_files)
        self.mask_video_confocal_file = find_filename_of_same_source(video_filename, mask_video_confocal_files)

        self.vessel_mask_oa790_file = find_filename_of_same_source(video_filename, vessel_mask_oa790_files)
        self.vessel_mask_oa850_file = find_filename_of_same_source(video_filename, vessel_mask_oa850_files)
        self.vessel_mask_confocal_file = find_filename_of_same_source(video_filename, vessel_mask_confocal_files)

        self.std_image_oa790_file = find_filename_of_same_source(video_filename, std_image_oa790_files)
        self.std_image_oa850_file = find_filename_of_same_source(video_filename, std_image_oa850_files)
        self.std_image_confocal_file = find_filename_of_same_source(video_filename, std_image_confocal_files)

        self._frames_oa790 = None
        self._frames_oa850 = None
        self._frames_confocal = None

        self._registered_frames_oa850 = None
        self._registered_mask_frames_oa850 = None
        self._registered_vessel_mask_oa850 = None

        self._mask_frames_oa790 = None
        self._mask_frames_oa850 = None
        self._mask_frames_confocal = None

        self._masked_frames_oa790 = None
        self._masked_frames_oa850 = None
        self._masked_frames_confocal = None

        self._vessel_masked_frames_oa790 = None
        self._vessel_masked_frames_oa850 = None
        self._vessel_masked_frames_confocal = None

        self._fully_masked_frames_oa790 = None
        self._fully_masked_frames_oa850 = None
        self._fully_masked_frames_confocal = None

        self._marked_frames_oa790 = None
        self._marked_frames_oa850 = None

        self._cell_positions = {}

        self._std_image_oa790 = None
        self._std_image_oa850 = None
        self._std_image_confocal = None

        self._vessel_mask_oa790 = None
        self._vessel_mask_oa850 = None
        self._vessel_mask_confocal = None

    @staticmethod
    def _assert_frame_assignment(old_frames, new_frames):
        assert new_frames.shape[1:3] == old_frames.shape[1:3], \
            f'Assigned frames should have the same height and width. Old dims {old_frames.shape[1:3]} new dims {new_frames.shape[1:3]}'
        assert new_frames.dtype == old_frames.dtype, \
            f'Assigned frames should have the same type. Old type new {old_frames.dtype} type {new_frames.dtype}'
        assert len(new_frames.shape) == 3, f'The assigned frames should be grayscale (shape given {new_frames.shape})'

    @property
    def frames_oa790(self):
        if self._frames_oa790 is None:
            self._frames_oa790 = get_frames_from_video(self.video_oa790_file)[..., 0]
        return self._frames_oa790

    @frames_oa790.setter
    def frames_oa790(self, new_frames):
        VideoSession._assert_frame_assignment(self.frames_oa790, new_frames)

        self._masked_frames_oa790 = None
        self._vessel_masked_frames_oa790 = None
        self._fully_masked_frames_oa790 = None

        self._frames_oa790 = new_frames

    @property
    def frames_oa850(self):
        if self._frames_oa850 is None:
            self._frames_oa850 = get_frames_from_video(self.video_oa850_file)[..., 0]
        return self._frames_oa850

    @frames_oa850.setter
    def frames_oa850(self, new_frames):
        VideoSession._assert_frame_assignment(self.frames_oa850, new_frames)

        self._masked_frames_oa850 = None
        self._vessel_masked_frames_oa850 = None
        self._fully_masked_frames_oa850 = None

        self._registered_frames_oa850 = None
        self._registered_mask_frames_oa850 = None
        self._registered_vessel_mask_oa850 = None

        self._frames_oa850 = new_frames

    @property
    def registered_frames_oa850(self):
        if self._registered_frames_oa850 is None:
            ir = ImageRegistator(source=self.vessel_mask_oa850, target=self.vessel_mask_confocal)
            ir.register_vertically()

            self._registered_frames_oa850 = np.empty_like(self.frames_oa850)
            self._registered_mask_frames_oa850 = np.empty_like(self.mask_frames_oa850)
            self._registered_vessel_mask_oa850 = ir.apply_registration(self.vessel_mask_oa850)

            for i, (frame, mask) in enumerate(zip(self.frames_oa850, self.mask_frames_oa850)):
                self._registered_frames_oa850[i] = ir.apply_registration(frame)
                self._registered_mask_frames_oa850[i] = ir.apply_registration(mask)

        return self._registered_frames_oa850

    @property
    def registered_mask_frames_oa850(self):
        if self._registered_mask_frames_oa850 is None:
            tmp = self.registered_frames_oa850
        return self._registered_mask_frames_oa850

    @property
    def registered_vessel_mask_oa850(self):
        if self._registered_vessel_mask_oa850 is None:
            tmp = self.registered_frames_oa850
        return self._registered_vessel_mask_oa850

    @property
    def frames_confocal(self):
        if self._frames_confocal is None:
            self._frames_confocal = get_frames_from_video(self.video_confocal_file)[..., 0]
        return self._frames_confocal

    @frames_confocal.setter
    def frames_confocal(self, new_frames):
        VideoSession._assert_frame_assignment(self.frames_confocal, new_frames)
        self._masked_frames_confocal = None
        self._vessel_masked_frames_confocal = None
        self._fully_masked_frames_confocal = None
        self._frames_confocal = new_frames

    @staticmethod
    def _rectify_mask_frames(masks):
        """ Rectify masks so that they are square.

        The original mask shape is irregular which can cause some inconveniences and problems.
        Rectify mask by clipping the irregular borders to straight lines so that the final mask
        is a rectangle.
        """
        import cv2
        cropped_masks = np.zeros_like(masks)

        for i, mask in enumerate(masks):
            # add a small border around the mask to detect changes on that are on the border
            mask_padded = cv2.copyMakeBorder(np.uint8(mask), 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)

            # Find where pixels go from black to white and from white to black (searching left to right)
            ys, xs = np.where(np.diff(mask_padded, axis=-1))

            # xs smaller than mean are on the left side and xs bigger than mean are on the right
            left_xs = xs[xs < xs.mean()]
            right_xs = xs[xs > xs.mean()]

            # clip mask to make it have straight lines
            m = np.bool8(mask_padded)
            m[:, :left_xs.max() + 1] = False
            m[:, right_xs.min():] = 0

            # remove the borders
            m = np.bool8(m[1:-1, 1:-1])

            cropped_masks[i, ...] = m

        assert cropped_masks.shape == masks.shape
        return cropped_masks

    @property
    def mask_frames_oa790(self):
        if self._mask_frames_oa790 is None:
            if self.mask_video_oa790_file == '':
                self.mask_frames_oa790 = np.ones_like(self.frames_oa790, dtype=np.bool8)
            else:
                self.mask_frames_oa790 = get_frames_from_video(self.mask_video_oa790_file)[..., 0].astype(np.bool8)

        return self._mask_frames_oa790

    @mask_frames_oa790.setter
    def mask_frames_oa790(self, masks):
        assert masks.dtype == np.bool8, f'The mask type must be {np.bool8}'
        assert masks.shape == self.frames_oa790.shape, \
            f'The frame masks must have the same shape as the frames. ' \
            f'frames oa790 shape:{self.frames_oa790.shape}, masks given shape:{masks.shape}'
        self._mask_frames_oa790 = VideoSession._rectify_mask_frames(masks)
        self._masked_frames_oa790 = None
        self._fully_masked_frames_oa790 = None

    @property
    def mask_frames_oa850(self):
        if self._mask_frames_oa850 is None:
            if self.mask_video_oa850_file == '':
                self.mask_frames_oa850 = np.ones_like(self.frames_oa850, dtype=np.bool8)
            else:
                self.mask_frames_oa850 = get_frames_from_video(self.mask_video_oa850_file)[..., 0].astype(np.bool8)

        return self._mask_frames_oa850

    @mask_frames_oa850.setter
    def mask_frames_oa850(self, masks):
        assert masks.dtype == np.bool8, f'The mask type must be {np.bool8}'
        assert masks.shape == self.frames_oa850.shape, \
            f'The frame masks must have the same shape as the frames. ' \
            f'frames oa850 shape:{self.frames_oa850.shape}, masks given shape:{masks.shape}'
        self._mask_frames_oa850 = VideoSession._rectify_mask_frames(masks)
        self._masked_frames_oa850 = None
        self._fully_masked_frames_oa850 = None

    @property
    def mask_frames_confocal(self):
        if self._mask_frames_confocal is None:
            if self.mask_video_confocal_file == '':
                self.mask_frames_confocal = np.ones_like(self.frames_confocal, dtype=np.bool8)
            else:
                self.mask_frames_confocal = get_frames_from_video(self.mask_video_confocal_file)[..., 0].astype(np.bool8)

        return self._mask_frames_confocal

    @mask_frames_confocal.setter
    def mask_frames_confocal(self, masks):
        assert masks.dtype == np.bool8, f'The mask type must be {np.bool8}'
        assert masks.shape == self.frames_confocal.shape, \
            f'The frame masks must have the same shape as the frames. ' \
            f'frames confocal shape:{self.frames_confocal.shape}, masks given shape:{masks.shape}'
        self._mask_frames_confocal = VideoSession._rectify_mask_frames(masks)
        self._masked_frames_confocal = None
        self._fully_masked_frames_confocal = None

    @property
    def masked_frames_oa790(self):
        """ The frames from the oa790nm channel masked with the corresponding frames of the masked video.
        """
        if self._masked_frames_oa790 is None:
            # We invert the mask because True values mean that the values are masked and therefor invalid.
            # see: https://numpy.org/doc/stable/reference/maskedarray.generic.html
            self._masked_frames_oa790 = np.ma.masked_array(self.frames_oa790,
                                                           ~self.mask_frames_oa790)
        return self._masked_frames_oa790

    @property
    def masked_frames_oa850(self):
        """ The frames from the oa850nm channel masked with the corresponding frames of the masked video.
        """
        if self._masked_frames_oa850 is None:
            # We invert the mask because True values mean that the values are masked and therefor invalid.
            # see: https://numpy.org/doc/stable/reference/maskedarray.generic.html
            self._masked_frames_oa850 = np.ma.masked_array(self.frames_oa850,
                                                           ~self.mask_frames_oa850[:len(self.frames_oa850)])
        return self._masked_frames_oa850

    @property
    def masked_frames_confocal(self):
        """ The frames from the confocal channel masked with the corresponding frames of the masked video.
        """
        if self._masked_frames_confocal is None:
            # We invert the mask because True values mean that the values are masked and therefor invalid.
            # see: https://numpy.org/doc/stable/reference/maskedarray.generic.html
            self._masked_frames_confocal = np.ma.masked_array(self.frames_confocal,
                                                              ~self.mask_frames_confocal[:len(self.frames_oa850)])
        return self._masked_frames_confocal

    @property
    def vessel_masked_frames_oa790(self):
        """ The frames from the oa790nm channel masked with the vessel mask image.
        """
        if self._vessel_masked_frames_oa790 is None:
            self._vessel_masked_frames_oa790 = np.ma.empty_like(self.frames_oa790)
            for i, frame in enumerate(self.frames_oa790):
                self._vessel_masked_frames_oa790[i] = np.ma.masked_array(frame, ~self.vessel_mask_confocal)
        return self._vessel_masked_frames_oa790

    @property
    def vessel_masked_frames_oa850(self):
        """ The frames from the oa850nm channel masked with the vessel mask image.
        """
        if self._vessel_masked_frames_oa850 is None:
            self._vessel_masked_frames_oa850 = np.ma.empty_like(self.frames_oa850)
            for i, frame in enumerate(self.frames_oa850):
                self._vessel_masked_frames_oa850[i] = np.ma.masked_array(frame, ~self.vessel_mask_oa850)
        return self._vessel_masked_frames_oa850

    @property
    def vessel_masked_frames_confocal(self):
        """ The frames from the confocal channel masked with the vessel mask image.
        """
        if self._vessel_masked_frames_confocal is None:
            self._vessel_masked_frames_confocal = np.ma.empty_like(self.frames_confocal)
            for i, frame in enumerate(self.frames_confocal):
                self._vessel_masked_frames_confocal[i] = np.ma.masked_array(frame, ~self.vessel_mask_confocal)
        return self._vessel_masked_frames_confocal

    @property
    def fully_masked_frames_oa790(self):
        """ The frames from the oa790nm channel masked with the vessel mask image and the mask frames from the mask video.
        """
        if self._fully_masked_frames_oa790 is None:
            self._fully_masked_frames_oa790 = np.ma.empty_like(self.frames_oa790)
            for i, frame in enumerate(self.frames_oa790):
                self._fully_masked_frames_oa790[i] = np.ma.masked_array(frame, ~(
                        self.vessel_mask_confocal * self.mask_frames_oa790[i]))
        return self._fully_masked_frames_oa790

    @property
    def fully_masked_frames_oa850(self):
        """ The frames from the oa850nm channel masked with the vessel mask image and the mask frames from the mask video.
        """
        if self._fully_masked_frames_oa850 is None:
            self._fully_masked_frames_oa850 = np.ma.empty_like(self.frames_oa850)
            for i, frame in enumerate(self.frames_oa850):
                self._fully_masked_frames_oa850[i] = np.ma.masked_array(frame, ~(
                        self.vessel_mask_oa850 * self.masked_frames_oa850[i]))
        return self._fully_masked_frames_oa850

    @property
    def fully_masked_frames_confocal(self):
        """ The frames from the confocalnm channel masked with the vessel mask image and the mask frames from the mask video.
        """
        if self._fully_masked_frames_confocal is None:
            self._fully_masked_frames_confocal = np.ma.empty_like(self.frames_confocal)
            for i, frame in enumerate(self.frames_confocal):
                self._fully_masked_frames_confocal[i] = np.ma.masked_array(frame, ~(
                        self.vessel_mask_confocal * self.mask_frames_confocal[i]))
        return self._fully_masked_frames_confocal

    @property
    def marked_frames_oa790(self):
        if self._marked_frames_oa790 is None:
            if not self.has_marked_video:
                raise Exception(f"Video session '{basename(self.video_oa790_file)}' has no marked video.")
            self._marked_frames_oa790 = get_frames_from_video(self.marked_video_oa790_file)[..., 0]
        return self._marked_frames_oa790

    @property
    def marked_frames_oa850(self):
        if self._marked_frames_oa850 is None:
            if self.marked_video_oa850_file == '':
                raise Exception(f"Video session '{basename(self.video_oa790_file)}' has no oa850 marked video.")
            self._marked_frames_oa850 = get_frames_from_video(self.marked_video_oa850_file)[..., 0]
        return self._marked_frames_oa850

    @property
    def cell_position_csv_files(self):
        """ Immutable list of filenames of the csvs with the cell positions.
        """
        return self._cell_position_csv_files.copy()

    def _add_to_cell_positions(self, csv_file):
        """ Warning, assumes that each csv_file has unique indices and overwrites entries from those indices
        without checking the actual coordinates.
        """
        csv_cell_positions_df = pd.read_csv(csv_file, delimiter=',')

        csv_cell_positions_coordinates = np.int32(csv_cell_positions_df[['X', 'Y']].to_numpy())
        csv_cell_positions_frame_indices = np.int32(csv_cell_positions_df[['Slice']].to_numpy())

        # The csv file is 1 indexed but python is 0 indexed so we -1.
        frame_indices_all = np.int32(np.squeeze(csv_cell_positions_frame_indices - 1))
        frame_indices_unique = np.unique(frame_indices_all)

        # Number of cells in videos is the same as the number of entries in the csv_file
        for frame_idx in frame_indices_unique:
            curr_coordinates = csv_cell_positions_coordinates[
                np.where(frame_indices_all == frame_idx)[0]
            ]
            if frame_idx in self._cell_positions:
                warnings.warn(f"Same slice index, '{frame_idx + 1}', found in multiple csv_files."
                              f' Overwriting with the latest csv file coordinates.')

            # warning overwriting coordinates in  frame_idx if already exist
            self._cell_positions[frame_idx] = curr_coordinates

    @property
    def validation_frame_idx(self):
        if self._validation_frame_idx is None:
            print('hello')
            max_positions = 0
            max_positions_frame_idx = list(self.cell_positions.keys())[0]
            print(max_positions_frame_idx)

            for frame_idx in self.cell_positions:
                # find and assign the frame with the most cell positions as a validation frame.

                # We don't want frame index to be the first frame for the usual case of temporal width 1.
                # We also  want to have some distance from the last (in case of motion contrast enhanced frames)
                if frame_idx == 0 or frame_idx >= len(self.frames_oa790) - 3:
                    continue

                cur_coordinates = self.cell_positions[frame_idx]

                if len(cur_coordinates) > max_positions and frame_idx != 0 and frame_idx != len(self.frames_oa790) - 2:
                    max_positions = len(cur_coordinates)
                    max_positions_frame_idx = frame_idx

            self._validation_frame_idx = max_positions_frame_idx

        return self._validation_frame_idx

    @validation_frame_idx.setter
    def validation_frame_idx(self, idx):
        assert idx in self.cell_positions, f'Frame index {idx} is not marked. Please assign a marked frame frame idx for validation'
        self._validation_frame_idx = idx


    def _remove_cell_positions(self, csv_file):
        """ Warning, assumes that each csv_file has unique indices and removes entries from those indices
        without checking the actual coordinates.
        """

        csv_cell_positions_df = pd.read_csv(csv_file, delimiter=',')

        # csv_cell_positions_coordinates = np.int32(csv_cell_positions_df[['X', 'Y']].to_numpy())
        csv_cell_positions_frame_indices = np.int32(csv_cell_positions_df[['Slice']].to_numpy())

        # The csv file is 1 indexed but python is 0 indexed so we -1.
        frame_indices = np.int32(np.unique(csv_cell_positions_frame_indices) - 1)

        for frame_idx in frame_indices:
            del self._cell_positions[frame_idx]

    def _initialise_cell_positions(self):
        for csv_file in self._cell_position_csv_files:
            self._add_to_cell_positions(csv_file)

    def append_cell_position_csv_file(self, csv_file):
        """ Adds cell positions from the csv file
        """
        self._cell_position_csv_files.append(csv_file)
        if len(self._cell_positions) == 0:
            self._initialise_cell_positions()
        else:
            self._add_to_cell_positions(csv_file)

    def pop_cell_position_csv_file(self, idx):
        """ Remove the csv cell positions from the csv file at index idx
        """
        self._cell_position_csv_files.pop(idx)
        if len(self._cell_positions) == 0:
            self._initialise_cell_positions()
        else:
            pass

    def remove_cell_position_csv_file(self, csv_file):
        """ Remove the csv cell positions from the csv file at index idx
        """
        self._cell_position_csv_files.remove(csv_file)
        if len(self._cell_positions) == 0:
            self._initialise_cell_positions()
        else:
            pass

    @property
    def cell_positions(self):
        """ A dictionary with {frame index -> Nx2 x,y cell positions}.

        Returns the positions of the blood cells as a dictionary indexed by the frame index as is in the csv file
        but 0 indexed instead!. To get the first frame do  session.positions[0] instead of session.positions[1].

        To access ith frame's cell positions do:
        self.positions[i - 1]
        """
        if len(self._cell_position_csv_files) == 0:
            raise Exception(f"No csv found with cell positions for video session {basename(self.video_oa790_file)}")

        if len(self._cell_positions) == 0:
            self._initialise_cell_positions()

        return self._cell_positions

    @staticmethod
    def _vessel_mask_from_file(file):
        vessel_mask = plt.imread(file)
        if len(vessel_mask.shape) == 3:
            vessel_mask = vessel_mask[..., 0]

        return np.bool8(vessel_mask)

    @property
    def vessel_mask_oa790(self):
        if self._vessel_mask_oa790 is None:
            if self.vessel_mask_oa790_file == '':
                vessel_image = create_vessel_image(
                    self.frames_oa790, self.mask_frames_oa790, sigma=0, method='j_tam', adapt_hist=True)
                self._vessel_mask_oa790 = binarize_vessel_image(vessel_image)
            else:
                self._vessel_mask_oa790 = VideoSession._vessel_mask_from_file(self.vessel_mask_oa790_file)
        return self._vessel_mask_oa790

    @vessel_mask_oa790.setter
    def vessel_mask_oa790(self, val):
        self._vessel_mask_oa790 = val

    @property
    def vessel_mask_oa850(self):
        if self._vessel_mask_oa850 is None:
            if self.vessel_mask_oa850_file == '':
                vessel_image = create_vessel_image(
                    self.frames_oa850, self.mask_frames_oa850, sigma=0, method='j_tam', adapt_hist=True)
                self._vessel_mask_oa850 = create_vessel_mask_from_vessel_image(vessel_image)
            else:
                self._vessel_mask_oa850 = VideoSession._vessel_mask_from_file(self.vessel_mask_oa850_file)
        return self._vessel_mask_oa850

    @vessel_mask_oa850.setter
    def vessel_mask_oa850(self, val):
        self._vessel_mask_oa850 = val
        self._vessel_masked_frames_oa850 = None
        self._fully_masked_frames_oa850 = None

        self._registered_mask_frames_oa850 = None
        self._registered_vessel_mask_oa850 = None

    @property
    def vessel_mask_confocal(self):
        if self._vessel_mask_confocal is None:
            if not self.vessel_mask_confocal_file:
                vessel_image = create_vessel_image(
                    self.frames_confocal, self.mask_frames_confocal, sigma=0, method='j_tam', adapt_hist=True)
                self._vessel_mask_confocal = create_vessel_mask_from_vessel_image(vessel_image)
            else:
                self._vessel_mask_confocal = VideoSession._vessel_mask_from_file(self.vessel_mask_confocal_file)
        return self._vessel_mask_confocal

    @vessel_mask_confocal.setter
    def vessel_mask_confocal(self, val):
        self._vessel_mask_confocal = val
        self._vessel_masked_frames_confocal = None
        self._fully_masked_frames_confocal = None
        self._registered_mask_frames_oa850 = None
        self._registered_mask_frames_oa850 = None

    @staticmethod
    def _std_image_from_file(file):
        from imageprosessing import normalize_data
        std_image = plt.imread(file)
        if len(std_image.shape) == 3:
            std_image = std_image[..., 0]

        # if image is 16 bit scale back to uint8 (assuming data range is from 0 to 2^16 -1)
        if std_image.dtype == np.uint16:
            uint16_max_val = np.iinfo(np.uint16).max
            std_image = np.uint8(normalize_data(std_image, target_range=(0, 255), data_range=(0, uint16_max_val)))

        return std_image

    @property
    def std_image_oa790(self):
        if self._std_image_oa790 is None:
            if self.std_image_oa790_file == '':
                raise Exception(f"No standard deviation image oa790 found for session '{self.video_oa790_file}'")
            self._std_image_oa790 = VideoSession._std_image_from_file(self.std_image_oa790_file)
        return self._std_image_oa790

    @property
    def std_image_oa850(self):
        if self._std_image_oa850 is None:
            if self.std_image_oa850_file == '':
                raise Exception(f"No standard deviation image oa850 found for session '{self.video_oa790_file}'")
            self._std_image_oa850 = VideoSession._std_image_from_file(self.std_image_oa850_file)
        return self._std_image_oa850

    @property
    def std_image_confocal(self):
        if self._std_image_confocal is None:
            if self.std_image_confocal_file == '':
                raise Exception(f"No standard deviation image confocal found for session '{self.video_oa790_file}'")
            self._std_image_confocal = VideoSession._std_image_from_file(self.std_image_confocal_file)
        return self._std_image_confocal


class SessionPreprocessor(object):
    preprocess_functions: List[Any]
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
        return masked_frames

    def apply_preprocessing_to_oa790(self):
        masked_frames = self._apply_preprocessing(self.session.masked_frames_oa790)
        self.session.frames_oa790 = masked_frames.filled(masked_frames.mean())
        self.session.mask_frames_oa790 = ~masked_frames.mask

    def apply_preprocessing_to_oa850(self):
        masked_frames = self._apply_preprocessing(self.session.masked_frames_oa850)
        self.session.frames_oa850 = masked_frames.filled(masked_frames.mean())
        self.session.mask_frames_oa850 = ~masked_frames.mask

    def apply_preprocessing_to_confocal(self):
        masked_frames = self._apply_preprocessing(self.session.masked_frames_confocal)
        self.session.frames_confocal = masked_frames.filled(masked_frames.mean())
        self.session.mask_frames_confocal = masked_frames.mask

    def apply_preprocessing(self):
        self.apply_preprocessing_to_confocal()
        self.apply_preprocessing_to_oa790()
        self.apply_preprocessing_to_oa850()


if __name__ == '__main__':
    from sharedvariables import get_video_sessions
    vs = get_video_sessions(should_be_registered=True)[0]
    vs.mask_frames_oa790
