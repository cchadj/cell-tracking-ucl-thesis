from typing import Any, Tuple

import numpy as np
from nearest_neighbors import get_nearest_neighbor_distances
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def get_positions_too_close_to_border(patch_positions, image_shape, patch_size):
    half_patch_height, half_patch_width = np.uint8(np.ceil(patch_size / 2)), np.uint8(np.ceil(patch_size / 2)),

    invalid = np.zeros([len(patch_positions), 1])
    invalid[patch_positions[:, 0] < (1 + half_patch_height)] = 1
    invalid[patch_positions[:, 0] > (image_shape[1] - half_patch_height)] = 1
    invalid[patch_positions[:, 1] < (1 + half_patch_height)] = 1
    invalid[patch_positions[:, 1] > (image_shape[0] - half_patch_height)] = 1

    return np.where(invalid)[0]


def evaluate_results(ground_truth_positions,
                     estimated_positions,
                     image,
                     mask=None,
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
        visualise_position_comparison (bool): If True  plots estimated positions superimposed on ground truth positions
            over the image.

    Returns:
        (EvaluationResults):
        Dice's coefficient, true positive rate, false discovery rate and other results.
    """
    assert len(ground_truth_positions) > 0
    assert len(estimated_positions) > 0
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=np.bool8)

    ground_truth_positions = ground_truth_positions.astype(np.int32)
    ground_truth_positions_pruned = np.delete(
        ground_truth_positions, np.where(~mask[ground_truth_positions[:, 1], ground_truth_positions[:, 0]])[0], axis=0
    )
    # np.delete(
    #     ground_truth_positions,
    #     get_positions_too_close_to_border(ground_truth_positions, image.shape[:2], patch_size),
    #     axis=0
    # )

    estimated_positions = estimated_positions.astype(np.int32)
    estimated_positions_pruned = np.delete(
        estimated_positions, np.where(~mask[estimated_positions[:, 1], estimated_positions[:, 0]])[0], axis=0
    )
    # np.delete(
    #     estimated_positions,
    #     get_positions_too_close_to_border(estimated_positions, image.shape[:2], patch_size),
    #     axis=0
    # )

    if len(ground_truth_positions_pruned) == 0 or len(estimated_positions_pruned) == 0:
        # return -1, -1, -1 for unexpected output. This happens when all positions are near borders.
        return -1, -1, -1

    median_spacing = np.mean(get_nearest_neighbor_distances(ground_truth_positions_pruned))
    distance_for_true_positive = .75 * median_spacing

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(ground_truth_positions_pruned)
    distances, closest_ground_truth_position_indices = nbrs.kneighbors(estimated_positions_pruned)
    distances, closest_ground_truth_position_indices = distances.squeeze(), closest_ground_truth_position_indices.squeeze()
    estimated_to_ground_distances = distances.copy()

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
    closest_ground_truth_position_indices = list(closest_ground_truth_position_indices.flatten())

    # The distance of each predicted point to it's closed manual position
    distances = list(distances.flatten())

    # The predicted point indices
    predicted_point_indices = list(np.arange(len(closest_ground_truth_position_indices)).flatten())

    # Sorting by closest_assigned_manual_indices to identify duplicates
    closest_ground_truth_position_indices, distances, predicted_point_indices = zip(*sorted(
        zip(closest_ground_truth_position_indices, distances, predicted_point_indices)))

    # Create dictionary with the manual positions and it's matched predicted positions.
    # The entries are of type match_dict[manual_position_idx] => ([distance_to_point_1, distance_to_point_2, ...],
    #                                                             [predicted_idx_1,     predicted_idx_2, ...])

    estimated_to_ground_truth = {}
    ground_truth_to_estimated = {}

    match_dict = {}
    for i in range(len(closest_ground_truth_position_indices)):
        closest_ground_truth_point_idx = closest_ground_truth_position_indices[i]

        distance_to_estimated_point = distances[i]
        estimated_point_idx = predicted_point_indices[i]

        estimated_to_ground_truth[estimated_point_idx] = closest_ground_truth_point_idx

        if closest_ground_truth_point_idx in match_dict.keys():
            match_dict[closest_ground_truth_point_idx][0].append(distance_to_estimated_point)
            match_dict[closest_ground_truth_point_idx][1].append(estimated_point_idx)

            ground_truth_to_estimated[closest_ground_truth_point_idx].append(estimated_point_idx)
        else:
            match_dict[closest_ground_truth_point_idx] = ([distance_to_estimated_point], [estimated_point_idx])

            ground_truth_to_estimated[closest_ground_truth_point_idx] = [estimated_point_idx]

    true_positive_points = np.empty((0, 2), dtype=np.int32)
    true_positive_dists = np.empty(0, dtype=np.float32)

    false_positive_points = np.empty((0, 2), dtype=np.int32)
    false_positive_dists = np.empty(0, dtype=np.float32)

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

        ground_truth_point = ground_truth_positions[key]
        minimum_dist_point = estimated_positions_pruned[predicted_indices[minimum_dist_idx]]
        mininum_dist = dists[minimum_dist_idx]

        false_positive_indices = np.delete(predicted_indices, minimum_dist_idx)
        cur_false_positive_points = estimated_positions_pruned[false_positive_indices]
        cur_false_positive_dists = np.delete(dists, minimum_dist_idx)

        false_positive_points = np.concatenate((false_positive_points, cur_false_positive_points), axis=0)
        false_positive_dists = np.concatenate((false_positive_dists, cur_false_positive_dists), axis=0)

        if mininum_dist <= distance_for_true_positive:
            true_positive_points = np.concatenate((true_positive_points, minimum_dist_point[None, ...]), axis=0)
            true_positive_dists = np.append(true_positive_dists, mininum_dist)
        else:
            false_positive_points = np.concatenate((false_positive_points, minimum_dist_point[None, ...]), axis=0)
            false_positive_dists = np.append(false_positive_dists, mininum_dist)

    # Remove false positives where distance between manual position and automatic position is bigger than median_spacing
    keys_to_delete = []
    for key in match_dict.keys():
        distance_to_estimated_point = match_dict[key][0]
        if distance_to_estimated_point >= distance_for_true_positive:
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

    results = EvaluationResults(
        dice=dices_coefficient,
        distance_for_true_positive=distance_for_true_positive,

        image=image,
        mask=mask,

        ground_truth_positions=ground_truth_positions_pruned,
        estimated_positions=estimated_positions_pruned,

        estimated_to_ground_truth=estimated_to_ground_truth,
        ground_truth_to_estimated=ground_truth_to_estimated,
        estimated_to_ground_distances=estimated_to_ground_distances,

        true_positive_rate=true_positive_rate,
        false_discovery_rate=false_discovery_rate,

        true_positive_points=true_positive_points,
        false_positive_points=false_positive_points,

        true_positive_dists=true_positive_dists,
        false_positive_dists=false_positive_dists,
    )

    if visualise_position_comparison:
        results.visualize()
        plt.show()

    return results


class EvaluationResults:
    true_positive_dists: np.ndarray
    distance_for_true_positive: np.ndarray

    def __init__(self,
                 dice, distance_for_true_positive, ground_truth_positions, true_positive_rate,
                 estimated_to_ground_distances, image, mask,
                 ground_truth_to_estimated, estimated_to_ground_truth,
                 false_discovery_rate, true_positive_dists,
                 false_positive_dists, true_positive_points,
                 false_positive_points, estimated_positions):
        self.extended_maxima_h = None
        self.region_max_threshold = None
        self.sigma = None
        self.probability_map = None

        self.image = image
        self.dice = dice
        self.mask = mask
        self.distance_for_true_positive = distance_for_true_positive

        self.ground_truth_to_estimated = ground_truth_to_estimated
        self.estimated_to_ground_truth = estimated_to_ground_truth

        self.ground_truth_positions = ground_truth_positions
        self.estimated_positions = estimated_positions

        self.estimated_to_ground_distances = estimated_to_ground_distances

        self.true_positive_rate = true_positive_rate
        self.false_discovery_rate = false_discovery_rate

        self.true_positive_points = true_positive_points
        self.false_positive_points = false_positive_points

        self.true_positive_dists = true_positive_dists
        self.false_positive_dists = false_positive_dists

        self.all_sigmas = None
        self.all_extended_maxima_hs = None
        self.all_dice_coefficients = None
        self.region_max_thresholds = None

    def visualize(self, show_probability_map=False):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, ConnectionPatch
        _, ax = plt.subplots(figsize=(60, 60))

        ax.imshow(self.image * self.mask, cmap='gray', vmin=0, vmax=255)
        if show_probability_map and self.probability_map is not None:
            ax.imshow(self.probability_map * self.mask, cmap='hot', vmin=0, vmax=255)

        ax.scatter(self.ground_truth_positions[:, 0], self.ground_truth_positions[:, 1],
                   c='blue', s=230, label='Ground truth positions')

        for point in self.ground_truth_positions:
            circ = Circle(point, self.distance_for_true_positive, fill=False, linestyle='--', color='blue')
            ax.add_artist(circ)

        ax.scatter(self.estimated_positions[:, 0], self.estimated_positions[:, 1],
                   c='yellow', label='Predicted Positions')

        for estimated_idx, ground_truth_idx in self.estimated_to_ground_truth.items():
            estimated_point = self.estimated_positions[estimated_idx]
            ground_truth_point = self.ground_truth_positions[ground_truth_idx]
            dist_to_ground_truth = self.estimated_to_ground_distances[estimated_idx]

            if dist_to_ground_truth <= self.distance_for_true_positive:
                con = ConnectionPatch(estimated_point, ground_truth_point, 'data', 'data',
                                      arrowstyle='-', shrinkA=5, shrinkB=5, mutation_scale=20, fc="w")
                ax.add_artist(con)

        ax.scatter(self.false_positive_points[:, 0], self.false_positive_points[:, 1],
                   c='red', s=150, label='False positive points')

        ax.scatter(self.true_positive_points[:, 0], self.true_positive_points[:, 1],
                   c='green', s=200, label='True positive points')

        ax.set_title(f"Dice's Coefficient {self.dice:.3f}.\n"
                     f'Distance between ground truth point and estimated point must be less than {self.distance_for_true_positive:.3f} to be TP.\n'
                     f'Mean true positive distance {self.true_positive_dists.mean():3f}\n'
                     f'True positive Rate {self.true_positive_rate:3f}.\n'
                     f'False positive Rate {self.false_discovery_rate:3f}.\n')
        ax.legend()


def dice(mask1, mask2):
    """ Computes the Dice coefficient, a measure of set similarity.

    Implementation thanks to: https://gist.github.com/JDWarner/6730747

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
