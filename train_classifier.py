from cell_no_cell import get_cell_and_no_cell_patches
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch-size', type=int, default=21)
    parser.add_argument('-h', '--do-hist-match', type=bool, default=False)
    parser.add_argument('-n', '--negatives-per-positive', type=int, default=3)

    args = parser.parse_args()
    patch_size = args.patch_size

    trainset, validset,\
    cell_images, non_cell_images,\
    cell_images_marked, non_cell_images_marked = get_cell_and_no_cell_patches(
        patch_size=patch_size,
        n_negatives_per_positive=args.negatives_per_positive,
        do_hist_match=args.do_hist_match,
    )



    pass


if __name__ == '__main__':
    trainset, validset, cell_images, non_cell_images,
    pass