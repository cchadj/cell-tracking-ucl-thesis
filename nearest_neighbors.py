from sklearn.neighbors import kd_tree


def get_nearest_neighbor(points):
    kdtree = kd_tree.KDTree(points, metric='euclidean')
    distances, nearest_points = kdtree.query(points, k=2)

    # points = ground_truth_cell_positions
    # nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(points)
    # distances, indices = nbrs.kneighbors(points)

    return distances[:, 1], nearest_points[: 1]


def get_nearest_neighbor_distances(points):
    distances, _ = get_nearest_neighbor(points)

    return distances
