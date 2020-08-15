import os
from os.path import basename
import shutil
import pathlib

import numpy as np
import torch
import tqdm

from imageprosessing import hist_match_images, normalize_data
from learningutils import LabeledImageDataset
from sharedvariables import get_video_sessions, CACHED_DATASETS_FOLDER
from patchextraction import SessionPatchExtractor
from PIL import Image


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            raise Exception('Failed to delete %s. Reason: %s' % (file_path, e))


def create_cell_and_no_cell_patches(
        patch_size=(21, 21),
        do_hist_match=False,
        n_negatives_per_positive=1,
        v=False,
        vv=False):
    assert type(patch_size) == int or type(patch_size) == tuple or (type(patch_size) == np.ndarray)
    if type(patch_size) == int:
        patch_size = patch_size, patch_size

    cell_images = np.zeros([0, *patch_size], dtype=np.uint8)
    non_cell_images = np.zeros_like(cell_images)

    cell_images_marked = np.zeros_like(cell_images, dtype=np.uint8)
    non_cell_images_marked = np.zeros_like(cell_images, dtype=np.uint8)

    video_sessions = get_video_sessions(should_have_marked_video=True)
    if v:
        print('Creating cell and no cell images from videos...')
    for session in tqdm.tqdm(video_sessions):
        assert session.has_marked_video, 'Something went wrong.' \
                                         ' get_video_sessions() should have ' \
                                         ' returned that have corresponding marked videos.'

        video_file = session.video_file
        marked_video_file = session.marked_video_oa790_file
        csv_cell_coord_files = session.cell_position_csv_files

        patch_extractor = SessionPatchExtractor(session, patch_size, n_negatives_per_positive)

        if vv:
            print('Unmarked', basename(video_file), '<->')
            print(*[basename(f) for f in csv_cell_coord_files], sep='\n')

        cur_session_cell_images = patch_extractor.cell_patches_oa790
        cur_session_non_cell_images = patch_extractor.non_cell_patches_oa790

        cell_images = np.concatenate((cell_images, cur_session_cell_images), axis=0)
        non_cell_images = np.concatenate((non_cell_images, cur_session_non_cell_images), axis=0)

        if vv:
            print('Marked', basename(marked_video_file), '<->')
            print(*[basename(f) for f in csv_cell_coord_files], sep='\n')

        cur_session_marked_cell_images = patch_extractor.marked_cell_patches_oa790
        cur_session_non_marked_cell_images = patch_extractor.marked_non_cell_patches_oa790

        cell_images_marked = np.concatenate((cell_images_marked, cur_session_marked_cell_images), axis=0)
        non_cell_images_marked = np.concatenate((non_cell_images_marked, cur_session_non_marked_cell_images),
                                                axis=0)

    if do_hist_match:
        if v:
            print('Doing histogram matching...')
        hist_match_template = np.float32(video_sessions[2].frames_oa790[0])
        if vv:
            print('Doing histogram matching on cell images')
        cell_images = hist_match_images(cell_images, hist_match_template)
        if vv:
            print('Doing histogram matching on non cell images')
        non_cell_images = hist_match_images(non_cell_images, hist_match_template)

    return cell_images, non_cell_images, cell_images_marked, non_cell_images_marked


def create_dataset_from_cell_and_no_cell_images(
        cell_images,
        non_cell_images,
        standardize=False,
        to_grayscale=False,
        v=False):
    if v:
        print('Creating dataset from cell and non cell patches')
        print('-----------------------------------------------')
    dataset = LabeledImageDataset(
        np.concatenate((cell_images[:len(cell_images), ...], non_cell_images[:len(non_cell_images), ...]), axis=0),
        np.concatenate((np.ones(len(cell_images)).astype(np.int), np.zeros(len(non_cell_images)).astype(np.int)),
                       axis=0),
        standardize=standardize,
        to_grayscale=to_grayscale,
    )

    if v:
        print('Splitting into training set and validation set')

    trainset_size = int(len(dataset) * 0.80)
    validset_size = len(dataset) - trainset_size
    # noinspection PyUnresolvedReferences
    trainset, validset = torch.utils.data.random_split(dataset, (trainset_size, validset_size))

    return trainset, validset


def get_cell_and_no_cell_patches(patch_size=(21, 21),
                                 n_negatives_per_positive=1,
                                 do_hist_match=False,
                                 standardize_dataset=False,
                                 dataset_to_grayscale=False,
                                 overwrite_cache=False,
                                 v=False,
                                 vv=False):
    """ Convenience function to get cell and no cell patches and their corresponding marked(for debugging),
        the torch Datasets, and the template image for histogram matching.

        Firstly checks the cache folder with the datasets to see if the dataset with the exact parameters was already
        created and if not then it creates it and saves it in cache.

    Args:
        patch_size (int, tuple): The patch size (height, width) or int for square.
        n_negatives_per_positive (int):  How many non cells per cell patch.
        do_hist_match (bool):  Whether to histogram matching or not.
        standardize_dataset: Standardizes the values of the datasets to have mean and std -.5, (values [-1, 1])
        dataset_to_grayscale: Makes the datasets output to have 1 channel.
        overwrite_cache (bool): Set to true to skip attempting to read from cache and create new dataset overwriting
                                the old data.
        v (bool):  Verbose description of what is currently happening.
        vv (bool):  Very Verbose description of what is currently happening.

    Returns:
        (Dataset, Dataset, np array NxHxW, np array,      np array,           np.array,             , np.array h x w of template)
        trainset, validset, cell_images, non_cell_images, cell_images_marked, non_cell_images_marked, hist_match_template

        Training set and validation set can be used by torch DataLoader.
        Cell images and non cell images are numpy arrays n x patch_height x patch_width and of type uint8.
        Cell images marked and non marked images are the same but from the marked videos for debugging.
        Histogram match template is the template used for histogram matching. If do_hist_match is False then
        None is returned.
    """
    assert type(patch_size) == int or type(patch_size) == tuple
    if type(patch_size) == int:
        height, width = patch_size, patch_size
    elif type(patch_size) == tuple:
        height, width = patch_size

    if v:
        print(f'patch size {(height, width)}')
        print(f'do hist match: {do_hist_match}')
        print(f'Negatives per positive: {n_negatives_per_positive}')
        print()
    if vv:
        v = True

    if not do_hist_match:
        hist_match_template = None

    patch_size = (height, width)

    dataset_folder = os.path.join(
        CACHED_DATASETS_FOLDER,
        f'dataset_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}')
    pathlib.Path(dataset_folder).mkdir(parents=True, exist_ok=True)

    trainset_filename = os.path.join(
        dataset_folder,
        f'trainset_bloodcells_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}.pt')
    validset_filename = os.path.join(
        dataset_folder,
        f'validset_bloodcells_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}.pt')

    cell_images_filename = os.path.join(
        dataset_folder,
        f'bloodcells_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}.npy')
    non_cell_images_filename = os.path.join(
        dataset_folder,
        f'non_bloodcells_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}.npy')
    cell_images_marked_filename = os.path.join(
        dataset_folder,
        f'bloodcells_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}_marked.npy')
    non_cell_images_marked_filename = os.path.join(
        dataset_folder,
        f'non_bloodcells_ps_{patch_size[0]}_hm_{str(do_hist_match).lower()}_npp_{n_negatives_per_positive}_marked.npy')
    template_image_filename = os.path.join(
        dataset_folder,
        f'hist_match_template_image'
    )
    # normalisation_data_range_filename = os.path.join(
    #     dataset_folder,
    #     f'normalisation_data_range_image.npy'
    # )

    try:
        if overwrite_cache:
            if v:
                print('Not checking cache. Overwriting any old data in the cache.')
            clean_folder(dataset_folder)
            # raise exception to go to catch scope
            raise FileNotFoundError

        if v:
            print('Dataset loading from cache')
            print('--------------------------')
            print(f"loading training set from '{trainset_filename}'...")
        trainset = torch.load(trainset_filename)
        if v:
            print(f"loading validation set from '{validset_filename}'...")
        validset = torch.load(validset_filename)

        if v:
            print(f"loading bloodcell patches from '{cell_images_filename}'...")
        cell_images = np.load(cell_images_filename)
        if v:
            print(f"loading non bloodcell patches from '{non_cell_images_filename}'...")
        non_cell_images = np.load(non_cell_images_filename)

        if v:
            print(f"loading marked bloodcell patches from '{cell_images_marked_filename}'...")
        cell_images_marked = np.load(cell_images_marked_filename)
        if v:
            print(f"loading marked non bloodcell patches from '{non_cell_images_marked_filename}'")
        non_cell_images_marked = np.load(non_cell_images_marked_filename)

        # if v:
        #     print(f"loading normalisation range '{normalisation_data_range_filename}'")
        # normalisation_data_range = np.load(normalisation_data_range_filename)

        if do_hist_match:
            if v:
                print(f"loading histogram matching template image (npy array)")
            hist_match_template = np.load(template_image_filename + '.npy')

        if v:
            print('Done')
            print()
    except FileNotFoundError:
        print('Not all data found fom cache. Creating datasets...')

        cell_images, non_cell_images, cell_images_marked, non_cell_images_marked =\
            create_cell_and_no_cell_patches(patch_size=patch_size,
                                            do_hist_match=False,
                                            n_negatives_per_positive=1,
                                            v=v, vv=vv)

        hist_match_template = cell_images[0]
        if do_hist_match:
            cell_images = hist_match_images(cell_images, hist_match_template)
            non_cell_images = hist_match_images(cell_images, hist_match_template)

        trainset, validset = create_dataset_from_cell_and_no_cell_images(
            cell_images, non_cell_images, standardize=standardize_dataset, to_grayscale=dataset_to_grayscale, v=v
        )

        if v:
            print()
            print('Saving datasets')
            print('---------------')

        torch.save(trainset, os.path.join(trainset_filename))
        torch.save(validset, os.path.join(validset_filename))
        np.save(cell_images_filename, cell_images)
        np.save(non_cell_images_filename, non_cell_images)
        np.save(cell_images_marked_filename, cell_images_marked)
        np.save(non_cell_images_marked_filename, non_cell_images_marked)
        # np.save(normalisation_data_range_filename, normalisation_data_range)
        if do_hist_match:
            Image.fromarray(np.uint8(hist_match_template * 255)).save(template_image_filename + '.png')
            np.save(template_image_filename + '.npy', hist_match_template)

        if v:
            print(f"Saved training set as: '{trainset_filename}'")
            print(f"Saved validation set as: '{validset_filename}'")
            print('Saving cell and non cell images')
            print(f"Saved cell images as: '{cell_images_filename}'")
            print(f"Saved non cell images as: '{non_cell_images_filename}'")
            print(f"Saved marked cell images (for debugging) as: '{cell_images_marked_filename}'")
            print(f"Saved marked non cell images (for debugging) as: '{non_cell_images_marked_filename}'")
            if do_hist_match:
                print(f"Saved histogram matching template as: {template_image_filename}.png")
                print(f"Saved histogram matching template (npy array) as: {template_image_filename}.npy")
            # print(f"Saved normalisation data range as: {normalisation_data_range}.npy")
            print("Cell images array shape:", cell_images.shape)
            print("Non cell images array shape:", non_cell_images.shape)

    return trainset, validset,\
        cell_images, non_cell_images, \
        cell_images_marked, non_cell_images_marked, \
        hist_match_template


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch-size', default=21, type=int, help='Patch size')
    parser.add_argument('--hist-match', action='store_true',
                        help='Set this flag to do histogram match.')
    parser.add_argument('-n', '--n-negatives-per-positive', default=3, type=int)
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='Set to overwrite existing datasets in cache')
    parser.add_argument('-v', default=False, action='store_true',
                        help='Verbose output.')
    parser.add_argument('-vv', default=False, action='store_true',
                        help='Very verbose output.')
    args = parser.parse_args()

    patch_size = args.patch_size, args.patch_size
    hist_match = args.hist_match
    npp = args.n_negatives_per_positive
    overwrite = args.overwrite
    v = args.v
    vv = args.vv

    print('---------------------------------------')
    print('Patch size:', patch_size)
    print('hist match:', hist_match)
    print('Negatives per positive:', npp)
    print('---------------------------------------')

    get_cell_and_no_cell_patches(patch_size=patch_size,
                                 n_negatives_per_positive=npp,
                                 do_hist_match=hist_match,
                                 overwrite_cache=overwrite,
                                 v=v,
                                 vv=vv)


if __name__ == '__main__':
    main()
