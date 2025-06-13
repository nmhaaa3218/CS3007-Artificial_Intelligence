# kdtree implementation for CS3007 Assignment 3
STUDENT_ID = "a1840406"
DEGREE = "UG"

import sys
import numpy as np
import pandas as pd

# global vars 
first_split_left_pts_count  = -1
first_split_right_pts_count = -1

class KdNode:
    def __init__(self, point, d, val):
        self.point = point
        self.d = d
        self.val = val
        self.left = None
        self.right = None

def calculate_median(values):
    if not values:
        return None
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if n % 2 == 1:
        # odd -> middle element
        return sorted_values[n // 2]
    else:
        # even -> avg of two middle elements
        mid1 = sorted_values[n // 2 - 1]
        mid2 = sorted_values[n // 2]
        return (mid1 + mid2) / 2

def build_kd_tree(points, depth=0, initial_depth_for_output=0):
    global first_split_left_pts_count, first_split_right_pts_count

    if not points:
        return None

    n_features = len(points[0]) - 1
    axis = depth % n_features

    if len(points) == 1:
        p = points[0]
        return KdNode(point=p, d=axis, val=p[axis])

    ax_values = [p[axis] for p in points]
    split_val = calculate_median(ax_values)

    points.sort(key=lambda p: p[axis])
    median_idx = (len(points) - 1) // 2
    median_point = points[median_idx]

    left_pts  = [p for i, p in enumerate(points)
                 if i != median_idx and p[axis] < split_val]
    right_pts = [p for i, p in enumerate(points)
                 if i != median_idx and p[axis] >= split_val]

    if depth == initial_depth_for_output and first_split_left_pts_count == -1:
        first_split_left_pts_count  = len(left_pts)
        first_split_right_pts_count = len(right_pts)

    node = KdNode(point=median_point, d=axis, val=split_val)
    node.left = build_kd_tree(left_pts,  depth + 1, initial_depth_for_output)
    node.right = build_kd_tree(right_pts, depth + 1, initial_depth_for_output)
    return node

# L2 distance my man =)))))))) (please don't judge me)
def calculate_l2_distance(point1_features, point2_features):
    # return np.sum((point1_features - point2_features) ** 2)
    return np.linalg.norm(point1_features - point2_features)

def find_nearest_neighbor_recursive_search(root, query):
    best_point = None
    best_dist_sq = float("inf")

    def recurse(node):
        nonlocal best_point, best_dist_sq
        if node is None:
            return

        # Calculate distance to current node (excluding label)
        current_dist_sq = calculate_l2_distance(query, node.point[:-1])
        if current_dist_sq < best_dist_sq:
            best_dist_sq = current_dist_sq
            best_point = node.point

        # Get the splitting axis
        axis = node.d
        if axis >= len(query):
            return

        # Determine which child to search first
        diff = query[axis] - node.val
        if diff <= 0:
            # Search left subtree first
            recurse(node.left)
            # If we might find a closer point in right subtree, search it too
            if diff ** 2 < best_dist_sq:
                recurse(node.right)
        else:
            # Search right subtree first
            recurse(node.right)
            if diff ** 2 < best_dist_sq:
                recurse(node.left)

    recurse(root)
    return best_point

def standardize_features(arr2d):
    means = arr2d.mean(axis=0)
    stds = arr2d.std(axis=0, ddof=0)
    stds[stds == 0] = 1
    return (arr2d - means) / stds, means, stds

def apply_standardization(arr2d, means, stds):
    return (arr2d - means) / stds

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    first_axis = int(sys.argv[3])

    # train_df = pd.read_csv(train_path, sep="\s+")
    # X_train = train_df.iloc[:, :-1].values.astype(float)
    # y_train = train_df.iloc[:, -1].values.astype(float)

    # X_train_s, means, stds = standardize_features(X_train)
    # train_scaled_with_lbl = np.hstack([X_train_s, y_train.reshape(-1, 1)])
    # training_rows = [row for row in train_scaled_with_lbl]

    # kd_root = build_kd_tree(training_rows, depth=first_axis, initial_depth_for_output=first_axis)

    # print("." * first_axis + f"l{first_split_left_pts_count}")
    # print("." * first_axis + f"r{first_split_right_pts_count}")

    # test_df = pd.read_csv(test_path, sep="\s+")
    # X_test_s = apply_standardization(test_df.values.astype(float), means, stds)

    # for q in X_test_s:
    #     nn_point = find_nearest_neighbor_recursive_search(kd_root, q)
    #     predicted_label = int(nn_point[-1]) if nn_point is not None else 0
    #     print(predicted_label)
    
    
    train_df = pd.read_csv(train_path, sep="\s+")
    X_train = train_df.iloc[:, :-1].values.astype(float)
    y_train = train_df.iloc[:, -1].values.astype(float)

    mask = (y_train >= 5) & (y_train <= 7)
    X_train = X_train[mask]
    y_train = y_train[mask]

    train_with_lbl = np.hstack([X_train, y_train.reshape(-1, 1)])
    training_rows = [row for row in train_with_lbl]

    kd_root = build_kd_tree(training_rows, depth=first_axis, initial_depth_for_output=first_axis)

    print("." * first_axis + f"l{first_split_left_pts_count}")
    print("." * first_axis + f"r{first_split_right_pts_count}")

    test_df = pd.read_csv(test_path, sep="\s+")
    X_test = test_df.values.astype(float)

    for q in X_test:
        nn_point = find_nearest_neighbor_recursive_search(kd_root, q)
        predicted_label = int(nn_point[-1]) if nn_point is not None else 0
        print(predicted_label)