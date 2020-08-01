import os
import sys
from os.path import basename


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
CACHE_FOLDER = os.path.join(DATA_FOLDER, 'cache')
CACHED_MODELS_FOLDER = os.path.join(CACHE_FOLDER, 'models')
CACHED_DATASETS_FOLDER = os.path.join(CACHE_FOLDER, 'datasets')
CACHED_DICE = os.path.join(CACHE_FOLDER, 'dice')
SHARED_VIDEOS_PATH = os.path.join(DATA_FOLDER, 'shared-videos')

OUTPUT_FOLDER = os.path.join(DATA_FOLDER, 'output')
OUTPUT_ESTIMATED_POSITIONS_FOLDER = os.path.join(OUTPUT_FOLDER, 'estimated-positions')

REGISTERED_VIDEOS_FOLDER = os.path.join(DATA_FOLDER, 'registered-videos')
REGISTERED_VIDEOS_2_FOLDER = os.path.join(DATA_FOLDER, 'registered-videos-2')
REGISTERED_VIDEOS_2_MARKED_FOLDER = os.path.join(REGISTERED_VIDEOS_2_FOLDER, 'marked')
TRAINED_MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, 'trained_models')
STDEV_FOLDER = os.path.join(DATA_FOLDER, 'stdev-images')

all_video_filenames = [os.path.join(SHARED_VIDEOS_PATH, f) for f in os.listdir(SHARED_VIDEOS_PATH) if f.endswith('.avi')]
all_video_filenames.extend([
    os.path.join(REGISTERED_VIDEOS_2_FOLDER, f) for f in os.listdir(REGISTERED_VIDEOS_2_FOLDER) if f.endswith('.avi')
])
all_video_filenames.extend([
    os.path.join(REGISTERED_VIDEOS_2_MARKED_FOLDER, f) for f in os.listdir(REGISTERED_VIDEOS_2_MARKED_FOLDER) if f.endswith('.avi')
])

marked_video_OA790_filenames = [f for f in all_video_filenames if 'marked' in basename(f) and 'OA790' in f]
marked_video_OA850_filenames = [f for f in all_video_filenames if 'marked' in basename(f) and 'OA850' in f]

csv_cell_cords_filenames = [os.path.join(SHARED_VIDEOS_PATH, f) for f in os.listdir(SHARED_VIDEOS_PATH) if f.endswith('.csv')]
csv_cell_cords_filenames.extend([
    os.path.join(REGISTERED_VIDEOS_2_MARKED_FOLDER, f) for
    f in os.listdir(REGISTERED_VIDEOS_2_MARKED_FOLDER) if f.endswith('.csv')
])
csv_cell_cords_OA790_filenames = [file for file in csv_cell_cords_filenames if 'OA790nm' in file]
csv_cell_cords_OA850_filenames = [file for file in csv_cell_cords_filenames if 'OA850nm' in file]

_unmarked_video_OA790_filenames_all = [
    f for f in all_video_filenames if 'OA790nm' in basename(f) and 'marked' not in basename(f) and 'mask' not in basename(f)
]
# Video files that correspond to each csv cell coordinate file
unmarked_labeled_video_OA790_filenames = []
for csv_cord_file in csv_cell_cords_OA790_filenames:
    for labeled_video_file in _unmarked_video_OA790_filenames_all:
        if files_of_same_source(csv_cord_file, labeled_video_file):
            unmarked_labeled_video_OA790_filenames.append(labeled_video_file)

_unmarked_video_OA850_filenames_all = [
    f for f in all_video_filenames if 'marked' not in basename(f) and 'OA850nm' in basename(f) and 'mask' not in basename(f)
]
# Video files that correspond to each unmarked labeled OA790nm channel video
unmarked_video_OA850_filenames = []
for file_OA790nm in unmarked_labeled_video_OA790_filenames:
    for file_OA850nm in _unmarked_video_OA850_filenames_all:
        if files_of_same_source(file_OA790nm, file_OA850nm):
            unmarked_video_OA850_filenames.append(file_OA850nm)

mask_video_filenames = [
    os.path.join(REGISTERED_VIDEOS_2_FOLDER, file) for file in os.listdir(REGISTERED_VIDEOS_2_FOLDER)
    if file.endswith('.avi') and 'mask' in file
]

std_images = [
    os.path.join(REGISTERED_VIDEOS_2_FOLDER, f) for
    f in os.listdir(REGISTERED_VIDEOS_2_FOLDER) if f.endswith('.tif') and 'std' in basename(f)
]
std_images_OA790 = [f for f in std_images if 'OA790' in f]
std_images_OA850 = [f for f in std_images if 'OA850' in f]
std_images_confocal = [f for f in std_images if 'Confocal' in f]

# Make a list of std confocal images that correspond to the labeled videos
# TODO: Currently adding two empty strings for the two videos without std images in shared_video
std_confocal_images_for_labeled_OA790 = []
for labeled_video_file in unmarked_labeled_video_OA790_filenames:
    file_found = False
    for std_image in std_images_confocal:
        if files_of_same_source(std_image, labeled_video_file):
            file_found = True
            std_confocal_images_for_labeled_OA790.append(std_image)
            break
    if not file_found:
        std_confocal_images_for_labeled_OA790.append('')

# TODO: Currently adding two empty strings for the two videos without mask videos in shared_video
mask_video_filenames_for_labeled_O790 = []
for labeled_video_file in unmarked_labeled_video_OA790_filenames:
    file_found = False
    for mask_video in mask_video_filenames:
        if 'OA790' in mask_video and files_of_same_source(mask_video, labeled_video_file):
            file_found = True
            mask_video_filenames_for_labeled_O790.append(mask_video)
            break
    if not file_found:
        mask_video_filenames_for_labeled_O790.append('')

# Create a list of std images that correspond to the labeled videos
# TODO: Currently adding two empty strings for the two videos without std image in shared_video
std_OA850_images_for_labeled_OA790 = []
for labeled_video_file in unmarked_labeled_video_OA790_filenames:
    file_found = False
    for std_image in std_images_OA850:
        if files_of_same_source(std_image, labeled_video_file):
            file_found = True
            std_OA850_images_for_labeled_OA790.append(std_image)
            break
    if not file_found:
        std_OA850_images_for_labeled_OA790.append('')

if __name__ == '__main__':
    assert len(unmarked_labeled_video_OA790_filenames) == len(marked_video_OA790_filenames)
    for i, (unmarked_video, marked_video) in enumerate(zip(unmarked_labeled_video_OA790_filenames, marked_video_OA790_filenames)):
        assert files_of_same_source(unmarked_video, marked_video)

    assert len(unmarked_labeled_video_OA790_filenames) == len(csv_cell_cords_OA790_filenames)
    for i, (unmarked_video, csv) in enumerate(zip(unmarked_labeled_video_OA790_filenames, csv_cell_cords_OA790_filenames)):
        assert files_of_same_source(unmarked_video, csv)

    assert len(unmarked_video_OA850_filenames) == len(unmarked_labeled_video_OA790_filenames)
    for vidOA790, vidO850 in zip(unmarked_labeled_video_OA790_filenames, unmarked_video_OA850_filenames):
        assert files_of_same_source(vidOA790, vidO850)

    assert len(unmarked_labeled_video_OA790_filenames) == len(std_confocal_images_for_labeled_OA790)
    for i, (unmarked_video, marked_video) in enumerate(zip(unmarked_labeled_video_OA790_filenames, marked_video_OA790_filenames)):
        assert files_of_same_source(unmarked_video, marked_video) or unmarked_video == ''

    assert len(unmarked_labeled_video_OA790_filenames) == len(mask_video_filenames_for_labeled_O790)
    for video, mask_video in zip(unmarked_labeled_video_OA790_filenames, mask_video_filenames_for_labeled_O790):
        assert files_of_same_source(video, mask_video) or mask_video == ''

    assert len(unmarked_labeled_video_OA790_filenames) == len(std_OA850_images_for_labeled_OA790)
    for vid_OA790, std_OA850 in zip(unmarked_labeled_video_OA790_filenames, std_OA850_images_for_labeled_OA790):
        assert files_of_same_source(vid_OA790, std_OA850) or std_OA850 == ''

    assert len(unmarked_labeled_video_OA790_filenames) == len(mask_video_filenames_for_labeled_O790)
    for vid_OA790, mask_vid in zip(unmarked_labeled_video_OA790_filenames, mask_video_filenames_for_labeled_O790):
        assert files_of_same_source(vid_OA790, mask_vid) or mask_vid == ''

    sys.exit(0)

