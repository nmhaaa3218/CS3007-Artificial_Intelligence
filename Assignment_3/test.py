# kdtree implementation for CS3007 Assignment 3

STUDENT_ID = 'a1840406'
DEGREE = 'UG'

import sys
import pandas as pd
import numpy as np

# global vars 
first_split_left_pts_count = -1
first_split_right_pts_count = -1

class KdNode:
    def __init__(self, point, axis, median_val, left_child=None, right_child=None):
        self.point = point
        self.axis = axis
        self.median_val = median_val  
        self.left = left_child
        self.right = right_child
        
def calculate_median(values):
    if not values:
        return None
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if n % 2 == 1:
        # Odd number of elements - return middle element
        return sorted_values[n // 2]
    else:
        # Even number of elements - return average of two middle elements
        mid1 = sorted_values[n // 2 - 1]
        mid2 = sorted_values[n // 2]
        return (mid1 + mid2) / 2.0


def build_kdtree(points_list, current_depth=0, initial_call_depth_for_output=0):
    global first_split_left_pts_count, first_split_right_pts_count

    if not points_list:
        return None

    num_features = len(points_list[0]) - 1
    if num_features <= 0:
        return None

    axis_to_split_on = current_depth % num_features

    if len(points_list) == 1:
        p = points_list[0]
        return KdNode(point=p,
                      axis=axis_to_split_on,
                      median_val=p[axis_to_split_on])

    # Extract values for the splitting axis
    axis_values = [pt[axis_to_split_on] for pt in points_list]
    
    # Calculate the median using the calculate_median function
    splitting_value = calculate_median(axis_values)

    # Find the point closest to the median value
    points_list.sort(key=lambda pt: pt[axis_to_split_on])
    median_idx = (len(points_list) - 1) // 2
    median_point = points_list[median_idx]

    left_points  = [pt for i, pt in enumerate(points_list)
                    if i != median_idx and pt[axis_to_split_on] < splitting_value]
    right_points = [pt for i, pt in enumerate(points_list)
                    if i != median_idx and pt[axis_to_split_on] >= splitting_value]

    if current_depth == initial_call_depth_for_output and first_split_left_pts_count == -1:
        first_split_left_pts_count  = len(left_points)
        first_split_right_pts_count = len(right_points)

    node = KdNode(point=median_point,
                  axis=axis_to_split_on,
                  median_val=splitting_value)
    node.left  = build_kdtree(left_points,  current_depth + 1, initial_call_depth_for_output)
    node.right = build_kdtree(right_points, current_depth + 1, initial_call_depth_for_output)
    return node


# L2 distance my man =)))))))) (please don't judge me)
def calculate_squared_euclidean_distance(point1_features, point2_features):
    distance_squared = 0
    for i in range(len(point1_features)): 
        diff = point1_features[i] - point2_features[i]
        distance_squared += diff * diff
    return distance_squared

def find_nearest_neighbor_recursive_search(node, query_point_features, best_nn_info_dict):
    if node is None:
        return best_nn_info_dict

    current_node_point_features = node.point[:-1]
    dist_sq_to_this_node_point = calculate_squared_euclidean_distance(query_point_features, current_node_point_features)

    if dist_sq_to_this_node_point < best_nn_info_dict['dist_sq']:
        best_nn_info_dict['point'] = node.point
        best_nn_info_dict['dist_sq'] = dist_sq_to_this_node_point

    splitting_axis_of_current_node = node.axis
    
    # Check if the splitting axis is within bounds of query features
    if splitting_axis_of_current_node >= len(query_point_features):
        return best_nn_info_dict
        
    value_of_query_at_splitting_axis = query_point_features[splitting_axis_of_current_node]
    node_split_val_on_axis = node.median_val 

    primary_search_child = None
    secondary_search_child = None

    if value_of_query_at_splitting_axis < node_split_val_on_axis:
        primary_search_child = node.left
        secondary_search_child = node.right
    else:
        primary_search_child = node.right
        secondary_search_child = node.left
    
    best_nn_info_dict = find_nearest_neighbor_recursive_search(primary_search_child, 
                                                               query_point_features, 
                                                               best_nn_info_dict)

    # efficient search
    diff_axis_sq = (value_of_query_at_splitting_axis - node_split_val_on_axis) * \
                   (value_of_query_at_splitting_axis - node_split_val_on_axis)

    if diff_axis_sq < best_nn_info_dict['dist_sq']:
        # might be closer point on other side
        best_nn_info_dict = find_nearest_neighbor_recursive_search(secondary_search_child, 
                                                                   query_point_features, 
                                                                   best_nn_info_dict)
    
    return best_nn_info_dict

def find_1nn_for_query(kdtree_root_node, query_point_features_list):
    if kdtree_root_node is None:
        return 0

    initial_best_nn_info = {
        'point': None,
        'dist_sq': float('inf')
    }
    
    final_best_nn_info = find_nearest_neighbor_recursive_search(kdtree_root_node, 
                                                                query_point_features_list, 
                                                                initial_best_nn_info)
    
    if final_best_nn_info['point'] is not None:
        predicted_quality = final_best_nn_info['point'][-1]
        return int(predicted_quality)
    else:
        return 0

if __name__ == "__main__":
    train_file_path_arg = sys.argv[1]
    test_file_path_arg = sys.argv[2]
    initial_dimension_for_build = int(sys.argv[3])

    train_data_frame = pd.read_fwf(train_file_path_arg)
    training_data_points = train_data_frame.values.tolist()

    # please work :<
    kdtree_root = build_kdtree(training_data_points, 
                               initial_dimension_for_build, 
                               initial_dimension_for_build)

    # dot dot dot
    dots_prefix_str = '.' * initial_dimension_for_build
    print(f"{dots_prefix_str}l{first_split_left_pts_count}")
    print(f"{dots_prefix_str}r{first_split_right_pts_count}")

    test_data_frame = pd.read_fwf(test_file_path_arg)
    test_queries_list = test_data_frame.values.tolist()

    if kdtree_root is None and test_queries_list:
        # no training data edge case
        for _ in test_queries_list:
            print("0")
    elif test_queries_list:
        for single_test_query_features in test_queries_list:
            # Extract only feature columns 
            query_features = single_test_query_features[:-1]
            predicted_wine_quality = find_1nn_for_query(kdtree_root, query_features)
            print(predicted_wine_quality)