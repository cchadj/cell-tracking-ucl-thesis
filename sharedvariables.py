import os
import sys
from os.path import basename
import re
import glob


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
all_video_files = [f for f in all_files if f.lower().endswith(video_file_extensions)]
all_csv_files = [f for f in all_files if f.lower().endswith(csv_file_extension)]
all_image_files = [f for f in all_files if f.lower().endswith(image_file_extensions)]

# unmarked videos. Must not have '_marked' or 'mask in them.
all_video_files_unmarked = [f for f in all_video_files if '_marked' not in basename(f).lower() and 'mask' not in basename(f)]
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
            'coordinate_csv_file': [csv_file for csv_file
                                    in csv_cell_cords_oa790_filenames if files_of_same_source(csv_file, video_filename)]
        }
        video_file_dictionaries.append(video_file_dict)
    return video_file_dictionaries


if __name__ == '__main__':
    sys.exit(0)
