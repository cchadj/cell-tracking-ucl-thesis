import os
import sys
from os.path import basename
import re
import glob
from videoutils import get_frames_from_video
import numpy as np
import matplotlib.pyplot as plt


def files_of_same_source(f1, f2):
    f1, f2 = os.path.basename(f1), os.path.basename(f2)
    f1_split = f1.split('_')
    f2_split = f2.split('_')

    are_files_of_same_source = True
    for i, (s1, s2) in enumerate(zip(f1_split, f2_split)):
        if i > 4:
            break

        if s1 != s2:
            are_files_of_same_source = False
            break

    return are_files_of_same_source


DATA_FOLDER = os.path.join('.', 'data')
CACHE_FOLDER = os.path.join('.', 'cache')
CACHED_MODELS_FOLDER = os.path.join(CACHE_FOLDER, 'models')
CACHED_DATASETS_FOLDER = os.path.join(CACHE_FOLDER, 'datasets')
CACHED_DICE = os.path.join(CACHE_FOLDER, 'dice')

OUTPUT_FOLDER = os.path.join(DATA_FOLDER, 'output')
OUTPUT_ESTIMATED_POSITIONS_FOLDER = os.path.join(OUTPUT_FOLDER, 'estimated-positions')

# Set up file extensions here. All extensions must be lowercase.
csv_file_extension = ('.csv', '.txt')
video_file_extensions = ('.avi', '.webm', '.mp4')
image_file_extensions = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp')

# find all files in data folder
all_files = glob.glob(os.path.join(DATA_FOLDER, '**', '*.*'), recursive=True)

# # Remove duplicates based on the file basename.
all_files_basenames = [basename(f) for f in all_files]
seen = set()
all_files_uniq = []
all_files_basenames_uniq = []
for i, file in enumerate(all_files_basenames):
    if file not in seen:
        all_files_basenames_uniq.append(file)
        all_files_uniq.append(all_files[i])
        seen.add(file)
all_files = all_files_uniq
# sort based on basename
all_files = [f for _, f in sorted(zip(all_files_basenames_uniq, all_files))]
## end remove duplicates ##

all_video_files = [f for f in all_files if f.lower().endswith(video_file_extensions)]
all_csv_files = [f for f in all_files if f.lower().endswith(csv_file_extension)]
all_image_files = [f for f in all_files if f.lower().endswith(image_file_extensions)]

# unmarked videos. Must not have '_marked' or 'mask in them.
all_video_files_unmarked = [f for f in all_video_files if
                            '_marked' not in basename(f).lower() and 'mask' not in basename(f)]
unmarked_video_oa790_filenames = [f for f in all_video_files_unmarked if 'oa790' in basename(f).lower()]
unmarked_video_oa850_filenames = [f for f in all_video_files_unmarked if 'oa850' in basename(f).lower()]

# marked videos. Must not have 'mask'. Must have '_marked'.
all_video_files_marked = [f for f in all_video_files if '_marked' in basename(f).lower() and 'mask' not in basename(f)]
marked_video_oa790_files = [f for f in all_video_files_marked if 'oa790' in basename(f).lower()]
marked_video_oa850_files = [f for f in all_video_files_marked if 'oa850' in basename(f).lower()]

# mask videos. Must have '_mask.' in them.
all_mask_video_files = [f for f in all_video_files if '_mask.' in basename(f).lower()]
mask_video_oa790_files = [f for f in all_mask_video_files if 'oa790' in basename(f).lower()]
mask_video_oa850_files = [f for f in all_mask_video_files if 'oa850' in basename(f).lower()]
mask_video_confocal_files = [f for f in all_mask_video_files if 'confocal' in basename(f).lower().lower()]

# Csv files with blood-cell coordinate files.
all_csv_cell_cords_filenames = [f for f in all_csv_files if 'coords' in basename(f).lower() or 'cords' in basename(f)]
csv_cell_cords_oa790_filenames = [f for f in all_csv_cell_cords_filenames if 'oa790nm' in basename(f).lower()]
csv_cell_cords_oa850_filenames = [f for f in all_csv_cell_cords_filenames if 'oa850nm' in basename(f).lower()]

# mask images. must end with 'vessel_mask.<file_extension>'
all_vessel_mask_files = [f for f in all_image_files if '_vessel_mask.' in basename(f).lower()]
vessel_mask_oa790_files = [f for f in all_vessel_mask_files if 'oa790' in basename(f).lower()]
vessel_mask_oa850_files = [f for f in all_vessel_mask_files if 'oa850' in basename(f).lower()]
vessel_mask_confocal_files = [f for f in all_vessel_mask_files if 'confocal' in basename(f).lower()]

# standard deviation images. must end with '_std.<file_extension>
all_std_image_files = [f for f in all_image_files if '_std.' in basename(f).lower()]
std_image_oa790_files = [f for f in all_std_image_files if 'oa790' in basename(f).lower()]
std_image_oa850_files = [f for f in all_std_image_files if 'oa850' in basename(f).lower()]
std_image_confocal_files = [f for f in all_std_image_files if 'confocal' in basename(f).lower()]


def find_filename_of_same_source(target_filename, filenames):
    """ Find the file name in filenames that is of the same source of target filename.

    Args:
        target_filename:
        filenames:

    Returns:
        The filename in filenames that is of same source of target file name.
        If not found returns an empty string

    """
    for filename in filenames:
        if files_of_same_source(target_filename, filename):
            return filename
    return ''


class VideoSession(object):
    def __init__(self, video_filename):
        channel_type = ''
        if 'confocal' in video_filename.lower():
            channel_type = 'confocal'
        elif 'oa790' in video_filename.lower():
            channel_type = 'oa790'
        elif 'oa850' in video_filename.lower():
            channel_type = 'oa790'

        vid_marked_790_filename = find_filename_of_same_source(video_filename, marked_video_oa790_files)
        vid_marked_850_filename = find_filename_of_same_source(video_filename, marked_video_oa850_files)

        has_marked_video = vid_marked_790_filename != '' or vid_marked_850_filename != ''

        subject_number = -1
        session_number = -1
        for string in video_filename.split('_'):
            if 'Subject' in string:
                subject_number = int(re.search(r'\d+', string).group())
            if 'Session' in string:
                session_number = int(re.search(r'\d+', string).group())

        self.type = channel_type
        self.has_marked_video = has_marked_video
        self.subject_number = subject_number
        self.session_number = session_number
        self.video_file = video_filename
        self.video_oa790_file = find_filename_of_same_source(video_filename, unmarked_video_oa790_filenames)
        self.video_850_file = find_filename_of_same_source(video_filename, unmarked_video_oa850_filenames)
        self.marked_video_oa790_file = vid_marked_790_filename
        self.marked_video_oa850_file = vid_marked_850_filename
        self.mask_video_oa790_file = find_filename_of_same_source(video_filename, mask_video_oa790_files)
        self.mask_video_oa850_file = find_filename_of_same_source(video_filename, mask_video_oa850_files)
        self.mask_video_confocal_file = find_filename_of_same_source(video_filename, mask_video_confocal_files)
        self.vessel_mask_confocal_file = find_filename_of_same_source(video_filename, vessel_mask_confocal_files)
        self.vessel_mask_oa790_file = find_filename_of_same_source(video_filename, vessel_mask_oa790_files)
        self.vessel_mask_oa850_file = find_filename_of_same_source(video_filename, vessel_mask_oa850_files)
        self.std_image_confocal_file = find_filename_of_same_source(video_filename, std_image_confocal_files)
        self.std_image_oa790_file = find_filename_of_same_source(video_filename, std_image_oa790_files)
        self.std_image_oa850_file = find_filename_of_same_source(video_filename, std_image_oa850_files)
        self.cell_position_csv_files = [csv_file for csv_file
                                        in csv_cell_cords_oa790_filenames
                                        if files_of_same_source(csv_file, video_filename)]

        self._frames_oa790 = None
        self._frames_oa850 = None
        self._mask_frames_oa790 = None
        self._mask_frames_oa850 = None
        self._marked_frames_oa790 = None
        self._marked_frames_oa850 = None
        self._cell_positions = None
        self._std_image_oa790 = None
        self._std_image_oa850 = None
        self._std_image_confocal = None
        self._vessel_mask_oa790 = None
        self._vessel_mask_oa850 = None
        self._vessel_mask_confocal = None

    @property
    def frames_oa790(self):
        if self._frames_oa790 is None:
            self._frames_oa790 = get_frames_from_video(self.video_oa790_file)[..., 0]
        return self._frames_oa790

    @property
    def frames_oa850(self):
        if self._frames_oa850 is None:
            self._frames_oa850 = get_frames_from_video(self.video_850_file)[..., 0]
        return self._frames_oa850

    @property
    def mask_frames_oa790(self):
        if self._mask_frames_oa790 is None:
            if self.mask_video_oa790_file == '':
                raise Exception(f"Video session '{basename(self.video_oa790_file)}' has no mask video.")
            self._mask_frames_oa790 = get_frames_from_video(self.mask_video_oa790_file)
        return self._mask_frames_oa790

    @property
    def mask_frames_oa850(self):
        if self._mask_frames_oa850 is None:
            if self.mask_video_oa850_file == '':
                raise Exception(f"Video session '{basename(self.video_oa790_file)}' has no mask video.")
            self._mask_frames_oa850 = get_frames_from_video(self.mask_video_oa850_file)
        return self._mask_frames_oa850

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
    def cell_positions(self):
        """ A dictionary with {frame index -> Nx2 x,y cell positions}.

        Returns the positions of the blood cells as a dictionary indexed by the frame index as is in the csv file
        but 0 indexed instead!. To get the first frame do  session.cell_positions[0] instead of session.cell_positions[1].

        To access ith frame's cell positions do:
        self.cell_positions[i - 1]
        """
        import pandas as pd
        import numpy as np
        if len(self.cell_position_csv_files) == 0:
            raise Exception(f"No csv found with cell positions for video session {basename(self.video_oa790_file)}")

        cell_positions = {}
        if self._cell_positions is None:
            for csv_file in self.cell_position_csv_files:
                csv_cell_positions_df = pd.read_csv(csv_file, delimiter=',')

                csv_cell_positions_coordinates = np.int32(csv_cell_positions_df[['X', 'Y']].to_numpy())
                csv_cell_positions_frame_indices = np.int32(csv_cell_positions_df[['Slice']].to_numpy())

                frame_indices = np.unique(csv_cell_positions_frame_indices)

                # Number of cells in videos is the same as the number of entries in the csv_file
                for frame_idx in frame_indices:
                    curr_coordinates = csv_cell_positions_coordinates[
                        np.where(csv_cell_positions_frame_indices == frame_idx)[0]]
                    # The csv file is 1 indexed but python is 0 indexed so we -1.
                    cell_positions[frame_idx - 1] = curr_coordinates
        return cell_positions

    @property
    def vessel_mask_oa790(self):
        if self._vessel_mask_oa790 is None:
            if self.vessel_mask_oa790_file == '':
                raise Exception(f"No vessel mask found fr session {self.video_oa790_file}")
            vessel_mask = plt.imread(self.vessel_mask_confocal_file)
            if len(vessel_mask.shape) == 3:
                vessel_mask = vessel_mask[..., 0]
            self._vessel_mask_oa790 = np.bool8(vessel_mask)
        return self._vessel_mask_oa790

    @property
    def vessel_mask_oa850(self):
        if self._vessel_mask_oa850 is None:
            if self.vessel_mask_oa850_file == '':
                raise Exception(f"No vessel mask found fr session {self.video_oa790_file}")
            vessel_mask = plt.imread(self.vessel_mask_oa850_file)
            if len(vessel_mask.shape) == 3:
                vessel_mask = vessel_mask[..., 0]
            self._vessel_mask_oa850 = np.bool8(vessel_mask)
        return self._vessel_mask_oa850

    @property
    def vessel_mask_confocal(self):
        if self._vessel_mask_confocal is None:
            if self.vessel_mask_confocal_file == '':
                raise Exception(f"No vessel mask found for session {self.video_oa790_file}")
            vessel_mask = plt.imread(self.vessel_mask_confocal_file)
            if len(vessel_mask.shape) == 3:
                vessel_mask = vessel_mask[..., 0]
            self._vessel_mask_confocal = np.bool8(vessel_mask)
        return self._vessel_mask_confocal

    @property
    def std_image_oa790(self):
        if self._std_image_oa790 is None:
            if self.std_image_oa790_file == '':
                raise Exception(f"No standard deviation image oa790 found for session '{self.video_oa790_file}'")
            self._std_image_oa790 = plt.imread(self.std_image_oa790_file)
        return self._std_image_oa790

    @property
    def std_image_oa850(self):
        if self._std_image_oa850 is None:
            if self.std_image_oa850_file == '':
                raise Exception(f"No standard deviation image oa850 found for session '{self.video_oa790_file}'")
            self._std_image_oa850 = plt.imread(self.std_image_oa850_file)
        return self._std_image_oa850

    @property
    def std_image_confocal(self):
        if self._std_image_confocal is None:
            if self.std_image_confocal_file == '':
                raise Exception(f"No standard deviation image confocal found for session '{self.video_oa790_file}'")
            self._std_image_confocal = plt.imread(self.std_image_confocal_file)
        return self._std_image_confocal


def get_video_sessions(should_have_marked_video=False):
    available_channel_type = ['oa790', 'oa850', 'confocal']

    video_sessions = []
    for video_filename in unmarked_video_oa790_filenames:

        vid_session = VideoSession(video_filename)
        if should_have_marked_video and not vid_session.has_marked_video:
            continue
        else:
            video_sessions.append(vid_session)

    return video_sessions


def get_video_file_dictionaries(channel_type, should_have_marked_video=False):
    available_channel_type = ['oa790', 'oa850', 'confocal']
    assert channel_type.lower() in available_channel_type, f'Channel type must be one of {available_channel_type}'

    video_file_dictionaries = []
    for video_filename in unmarked_video_oa790_filenames:
        cur_channel_type = ''
        if 'confocal' in video_filename.lower():
            cur_channel_type = 'confocal'
        elif 'oa790' in video_filename.lower():
            cur_channel_type = 'oa790'
        elif 'oa850' in video_filename.lower():
            cur_channel_type = 'oa790'

        if cur_channel_type != channel_type.lower():
            continue

        vid_marked_790_filename = find_filename_of_same_source(video_filename, marked_video_oa790_files)
        vid_marked_850_filename = find_filename_of_same_source(video_filename, marked_video_oa850_files)

        has_marked_video = vid_marked_790_filename != '' or vid_marked_850_filename != ''

        if should_have_marked_video and not has_marked_video:
            continue

        subject_number = -1
        session_number = -1
        for string in video_filename.split('_'):
            if 'Subject' in string:
                subject_number = int(re.search(r'\d+', string).group())
            if 'Session' in string:
                session_number = int(re.search(r'\d+', string).group())

        video_file_dict = {
            'type': channel_type,
            'has_marked_video': has_marked_video,
            'subject_number': subject_number,
            'session_number': session_number,
            'video_file': video_filename,
            'video_790_file': find_filename_of_same_source(video_filename, unmarked_video_oa790_filenames),
            'video_850_file': find_filename_of_same_source(video_filename, unmarked_video_oa850_filenames),
            'marked_video_oa790_file': vid_marked_790_filename,
            'marked_video_oa850_file': vid_marked_850_filename,
            'mask_video_oa790_file': find_filename_of_same_source(video_filename, mask_video_oa790_files),
            'mask_video_oa850_file': find_filename_of_same_source(video_filename, mask_video_oa850_files),
            'mask_video_confocal_file': find_filename_of_same_source(video_filename, mask_video_confocal_files),
            'vessel_mask_confocal_file': find_filename_of_same_source(video_filename, vessel_mask_confocal_files),
            'vessel_mask_oa790_file': find_filename_of_same_source(video_filename, vessel_mask_oa790_files),
            'vessel_mask_oa850_file': find_filename_of_same_source(video_filename, vessel_mask_oa850_files),
            'std_image_confocal_file': find_filename_of_same_source(video_filename, std_image_confocal_files),
            'std_image_oa790_file': find_filename_of_same_source(video_filename, std_image_oa790_files),
            'std_image_oa850_file': find_filename_of_same_source(video_filename, std_image_oa850_files),
            'cell_position_csv_files': [csv_file for csv_file
                                        in csv_cell_cords_oa790_filenames if
                                        files_of_same_source(csv_file, video_filename)]
        }
        video_file_dictionaries.append(video_file_dict)
    return video_file_dictionaries


if __name__ == '__main__':
    # Make sure that all_files is unique based on the basename of the file (being at different path doesn't matter)
    all_files_basenames_unique = list(set([basename(f) for f in all_files]))
    assert len(all_files_basenames_unique) == len(all_files) and \
        [basename(f1) == f2 for f1, f2 in zip(sorted(all_files), sorted(all_files_basenames_unique))], \
        'all_files is not unique, there is one or more duplicates.'

    sys.exit(0)
