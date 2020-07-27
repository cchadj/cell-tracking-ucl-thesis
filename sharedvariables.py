import os
DATA_FOLDER = os.path.join('.', 'data')
CACHE_FOLDER = os.path.join(DATA_FOLDER, 'cache')
CACHED_MODELS_FOLDER = os.path.join(CACHE_FOLDER, 'models')
CACHED_DATASETS_FOLDER = os.path.join(CACHE_FOLDER, 'datasets')
SHARED_VIDEOS_PATH = os.path.join(DATA_FOLDER, 'shared-videos')

OUTPUT_FOLDER = os.path.join(DATA_FOLDER, 'output')
OUTPUT_ESTIMATED_POSITIONS_FOLDER = os.path.join(OUTPUT_FOLDER, 'estimated-positions')

REGISTERED_VIDEOS_FOLDER = os.path.join(DATA_FOLDER, 'registered-videos')
REGISTERED_VIDEOS_2_FOLDER = os.path.join(DATA_FOLDER, 'registered-videos-2')
REGISTERED_VIDEOS_2_MARKED_FOLDER = os.path.join(REGISTERED_VIDEOS_2_FOLDER, 'marked')
TRAINED_MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, 'trained_models')
STDEV_FOLDER = os.path.join(DATA_FOLDER, 'stdev-images')

video_filenames = [
    os.path.join(SHARED_VIDEOS_PATH, file) for file in
    [f for f in os.listdir(SHARED_VIDEOS_PATH) if f.endswith('.avi')]
]

video_filenames.extend([
    os.path.join(REGISTERED_VIDEOS_2_MARKED_FOLDER, file) for file in
    [f for f in os.listdir(REGISTERED_VIDEOS_2_MARKED_FOLDER) if f.endswith('.avi')]
])

marked_video_OA790_filenames = [f for f in video_filenames if 'marked' in os.path.basename(f) and 'OA790' in f]
marked_video_OA850_filenames = [f for f in video_filenames if 'marked' in os.path.basename(f) and 'OA850' in f]
unmarked_video_OA790_filenames = [f for f in video_filenames if 'marked' not in os.path.basename(f) and 'OA790nm' in f]
unmarked_video_OA850_filenames = [f for f in video_filenames if 'marked' not in os.path.basename(f) and 'OA850nm' in f]

csv_filenames = [os.path.join(SHARED_VIDEOS_PATH, f) for f in os.listdir(SHARED_VIDEOS_PATH) if f.endswith('.csv')]
csv_filenames.extend([
    os.path.join(REGISTERED_VIDEOS_2_MARKED_FOLDER, f) for
    f in os.listdir(REGISTERED_VIDEOS_2_MARKED_FOLDER) if f.endswith('.csv')
])
csv_OA790_filenames = [file for file in csv_filenames if 'OA790nm' in file]
csv_OA850_filenames = [file for file in csv_filenames if 'OA850nm' in file]

registered_videos_filenames = [os.path.join(REGISTERED_VIDEOS_FOLDER, file)
                               for file in os.listdir(REGISTERED_VIDEOS_FOLDER) if file.endswith('.avi')]

registered_videos_stdev_images = [os.path.join(REGISTERED_VIDEOS_FOLDER, file)
                               for file in os.listdir(REGISTERED_VIDEOS_FOLDER) if file.endswith('.tif')]

registered_videos_registration_csv = [os.path.join(REGISTERED_VIDEOS_FOLDER, file)
                               for file in os.listdir(REGISTERED_VIDEOS_FOLDER) if file.endswith('.csv')]

registered_videos_2_filenames = [
    os.path.join(REGISTERED_VIDEOS_2_FOLDER, file) for file in os.listdir(REGISTERED_VIDEOS_2_FOLDER)
    if file.endswith('.avi') and 'mask' not in file
]

registered_videos_2_mask_filenames = [
    os.path.join(REGISTERED_VIDEOS_2_FOLDER, file) for file in os.listdir(REGISTERED_VIDEOS_2_FOLDER)
    if file.endswith('.avi') and 'mask' in file
]

registered_videos_2_stdev_images = [
    os.path.join(REGISTERED_VIDEOS_2_FOLDER, file) for file in
    os.listdir(REGISTERED_VIDEOS_2_FOLDER) if file.endswith('.tif')
]

registered_videos_2_stdev_images_masks = [
    os.path.join(REGISTERED_VIDEOS_2_FOLDER, file) for file in
    os.listdir(REGISTERED_VIDEOS_2_FOLDER) if file.endswith('.jpg') and 'mask' in file
]

registered_videos_2_registration_csv = [
    os.path.join(REGISTERED_VIDEOS_2_FOLDER, file) for file in
    os.listdir(REGISTERED_VIDEOS_2_FOLDER) if file.endswith('.csv')
]

stdev_images = [os.path.join(STDEV_FOLDER, file) for file in os.listdir(STDEV_FOLDER)]
