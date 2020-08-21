import numpy as np
from nearest_neighbors import get_nearest_neighbor_distances
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def get_positions_too_close_to_border(patch_positions, image_shape, patch_size=(33, 33)):
    half_patch_height, half_patch_width = np.uint8(np.ceil(patch_size[0] / 2)), np.uint8(np.ceil(patch_size[1] / 2)),

    invalid = np.zeros([len(patch_positions), 1])
    invalid[patch_positions[:, 0] < (1+half_patch_height)] = 1
    invalid[patch_positions[:, 0] > (image_shape[1]-half_patch_height)] = 1
    invalid[patch_positions[:, 1] < (1+half_patch_height)] = 1
    invalid[patch_positions[:, 1] > (image_shape[0]-half_patch_height)] = 1

    return np.where(invalid)[0]


def evaluate_results(ground_truth_positions,
                     estimated_positions,
                     image,
                     patch_size=(21, 21),
                     visualise_position_comparison=False
                     ):
    """ Evaluate how good are the estimated position in relation to the ground truth.

    Get dice's coefficient, true positive rate and false discovery rate.
    A true positive is when the distance of an estimated position to it's closest ground truth position is less
    than d where d is .75 * median spacing of the ground truth positions.

    Args:
        ground_truth_positions (np.array): (Nx2( The ground truth positions.
        estimated_positions (np.array): The estimated positions.
        image (np.array): The image the cells are in. Used to get the shape for pruning points that are too close to the border.
        patch_size (tuple):  The patch size that the cells are in. Used for pruning points that are too close to the border.
        visualise_position_comparison (bool): True to plot estimated positions superimposed on ground truth positions
            over the image.

    Returns:
        (float, float, float):
        Dice's coefficient, true positive rate, false discovery rate.
    """
    assert len(ground_truth_positions) > 0
    assert len(estimated_positions) > 0

    ground_truth_positions_pruned = np.delete(
        ground_truth_positions,
        get_positions_too_close_to_border(ground_truth_positions, image.shape[:2], patch_size),
        axis=0
    )
    estimated_positions_pruned = np.delete(
        estimated_positions,
        get_positions_too_close_to_border(estimated_positions, image.shape[:2], patch_size),
        axis=0
    )

    if len(ground_truth_positions_pruned) == 0 or len(estimated_positions_pruned) == 0:
        # return -1, -1, -1 for unexpected output. This happens when all positions are near borders.
        return -1, -1, -1

    median_spacing = np.mean(get_nearest_neighbor_distances(ground_truth_positions))
    distance_for_true_positive = .75 * median_spacing

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ground_truth_positions_pruned)
    distances, closest_manual_position_indices = nbrs.kneighbors(estimated_positions_pruned)
    distances, closest_manual_position_indices = distances.squeeze(), closest_manual_position_indices.squeeze()

    # Each prediction is assigned to it's closest manual position.
    # True Positive:
    # If it's distance to the manual position is < distance_for_true_positive then it's considered true positive
    # False Positive:.
    # If it's distance is > distance_for_true_positive then it's considered a  false positive.
    # If multiple cells are assigned to the same manual position, then the closest is selected and the rest are counted
    # as false positive.
    # False Negatives:
    # Manually marked cones that did not have a matching automatically detected cone were considered as false negatives.

    # The indices of the manual positions that are assigned to each predicted point
    closest_manual_position_indices = list(closest_manual_position_indices.flatten())
    # print(closest_manual_position_indices)
    # The distance of each predicted point to it's closed manual position
    distances = list(distances.flatten())
    # The predicted point indices

    predicted_point_indices = list(np.arange(len(closest_manual_position_indices)).flatten())

    # Sorting by closest_assigned_manual_indices to identify duplicates
    closest_manual_position_indices, distances, predicted_point_indices = zip(*sorted(
        zip(closest_manual_position_indices, distances, predicted_point_indices)))
    # Create dictionary with the manual positions and it's matched predicted positions.
    # The entries are of type match_dict[manual_position_idx] = ([distance_1,      distance_2, ...],
    #                                                            [predicted_idx_1, predicted_idx_2, ...])

    match_dict = {}
    for i in range(1, len(closest_manual_position_indices)):
        closest_assigned_manual_position_idx = closest_manual_position_indices[i]
        dist = distances[i]
        predicted_point_idx = predicted_point_indices[i]

        if closest_assigned_manual_position_idx in match_dict.keys():
            match_dict[closest_assigned_manual_position_idx][0].append(dist)
            match_dict[closest_assigned_manual_position_idx][1].append(predicted_point_idx)
        else:
            match_dict[closest_assigned_manual_position_idx] = ([dist], [predicted_point_idx])

    # By now match_dict, can have many predicted positions for each manual position.
    # We keep the predicted position with the smallest distance.
    n_false_positives_duplicates = 0
    for key in match_dict.keys():
        dists = match_dict[key][0]
        predicted_indices = match_dict[key][1]
        minimum_dist_idx = np.argmin(dists)

        # if predicted positions that are matched to manual position are more than 1,
        # then we increase the number of false positives.
        n_false_positives_duplicates += len(predicted_indices) - 1

        # Match dict will contain a single distance and the predicted position index that is assigned to that
        # manual position
        match_dict[key] = (dists[minimum_dist_idx], predicted_indices[minimum_dist_idx])

    # Remove false positives where distance between manual position and automatic position is bigger than median_spacing
    keys_to_delete = []
    for key in match_dict.keys():
        dist = match_dict[key][0]
        if dist >= distance_for_true_positive:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del match_dict[key]

    # The remaining entries are true positives.
    n_true_positives = len(match_dict)

    n_false_positives_unmatched_predictions = len(estimated_positions_pruned) - len(match_dict)
    n_false_positives = n_false_positives_duplicates + n_false_positives_unmatched_predictions

    n_false_negatives = len(ground_truth_positions_pruned) - len(match_dict)

    n_manual = len(ground_truth_positions_pruned)
    n_automatic = len(estimated_positions_pruned)

    assert n_manual == n_true_positives + n_false_negatives
    assert n_automatic == n_true_positives + n_false_positives_unmatched_predictions

    true_positive_rate = n_true_positives / n_manual
    false_discovery_rate = n_false_positives / n_automatic
    dices_coefficient = (2 * n_true_positives) / (n_manual + n_automatic)

    if visualise_position_comparison:
        plt.imshow(image, cmap='gray')

        plt.scatter(ground_truth_positions_pruned[:, 0],
                    ground_truth_positions_pruned[:, 1],
                    c='blue',
                    s=100,
                    label='Manual Positions')
        plt.scatter(estimated_positions_pruned[:, 0],
                    estimated_positions_pruned[:, 1],
                    c='yellow',
                    label='Predicted Positions')
        plt.title(f"Dice's Coefficient {dices_coefficient:.3f}.\n"
                  f'Distance between ground truth and predicted position must be less than {median_spacing:.3f} to be TP.\n'
                  f'True positive Rate {true_positive_rate:3f}.\n'
                  f'False positive Rate {false_discovery_rate:3f}.\n')
        plt.legend()

        fig = plt.gcf()
        fig_size = fig.get_size_inches()
        fig.set_size_inches((fig_size[0] * 5, fig_size[1] * 5))

    return dices_coefficient, true_positive_rate, false_discovery_rate


# https://gist.github.com/JDWarner/6730747
def dice(mask1, mask2):
    """ Computes the Dice coefficient, a measure of set similarity.

    Parameters
    ----------
    mask1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    mask2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    mask1 = np.asarray(mask1).astype(np.bool)
    mask2 = np.asarray(mask2).astype(np.bool)

    if mask1.shape != mask2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(mask1, mask2)

    return 2 * intersection.sum() / (mask1.sum() + mask2.sum())
