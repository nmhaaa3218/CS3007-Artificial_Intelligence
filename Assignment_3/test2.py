import numpy as np
import pandas as pd
import argparse

def get_args():
    '''
    Get arguments from command line

    :return argparse.Namespace
    '''
    parser = argparse.ArgumentParser(description='1NN using KD tree')
    parser.add_argument('train', type=str, help='Path to data file')
    parser.add_argument('test', type=str, help='Path to query file')
    parser.add_argument('dimension', type=int, default=0, help='Dimension to choose')
    return parser

def print_kd_tree(node):
    if node is None:
        return
    print("Point:", node.point, " Dimension:", node.dimension, " Value:", node.value)
    print_kd_tree(node.left)
    print_kd_tree(node.right)

class KDP:
    def __init__(self, data: list, d=None, val=None, l=None, r=None)->None:
        '''
        Initialize a pts in the KD tree

        :param point: list of coordinates
        :param d: int
        :param val: float
        :param l: KDP left child
        :param r: KDP right child
        :return None
        '''
        self.point = data
        self.d = d
        self.val = val
        self.l = l
        self.r = r

class KDT:
    def __init__(self, P: np.array, dim_choose: int)->None:
        '''
        Initialize a KD tree

        :param P: np.array
        :return None
        '''
        self.root = self.BuildKdTree(P, dim_choose)

    def Find_1NN(self, query_point: list)->list:
        '''
        Find the nearest neighbor in the KD tree

        :param query_point: list
        :return list
        '''
        return self.NNSearch(self.root, query_point)

    def BuildKdTree(self, P: np.array, D: int)->KDP:
        '''
        Build a KD tree from a list of points

        :param P: np.array
        :param D: int
        :return KDP
        '''

        if len(P) == 0:
            return None
        elif len(P) == 1:
            return KDP(P[0], d=D)
        else:
            M = len(P[0])  # Retrieve number of dimensions
            d = D % M  # Select d based on depth
            sorted_indices = np.argsort(P[:, d])
            P = P[sorted_indices] # Sort points along d d
            median_idx = len(P) // 2 # Find median index
            val = P[median_idx][d] 
            pts = KDP(P[median_idx], d=d, val=val) # Create pts
            pts.l = self.BuildKdTree(P[:median_idx], D + 1) # Recursively build l subtree
            pts.r = self.BuildKdTree(P[median_idx + 1:], D + 1) # Recursively build r subtree
            return pts

    def NNSearch(self, pts: KDP, query_point: list)->list:
        '''
        Find the nearest neighbor in the KD tree

        :param pts: KDP
        :param query_point: list
        :return list
        '''
        
        if pts is None:
            return None
        if pts.l is None and pts.r is None:
            return pts.point
        dim = pts.d
        if query_point[dim] <= pts.val:
            nxt = pts.l
            ops = pts.r
        else:
            nxt = pts.r
            ops = pts.l
        result = self.NNSearch(nxt, query_point)
        if (result is None) or (np.linalg.norm(query_point - result[:-1]) > abs(query_point[dim] - pts.val)):
            result = self.NNSearch(ops, query_point)
        return result
    
def main():
    args = get_args()
    args = args.parse_args()
    train_data = pd.read_csv(args.train, sep="\s+")
    test_data = pd.read_csv(args.test, sep="\s+") 
    X_train = train_data
    y_train = train_data.iloc[:, -1] 
    X_train = X_train[(y_train >= 5) & (y_train <= 7)].values
    dim = args.dimension
    root = KDT(X_train, dim)

    # Perform 1-nearest neighbor search for each test sample
    predictions = []
    # loop thru each row in test data
    for i in range(len(test_data)):
        row = test_data.iloc[i].values
        nearest_neighbor = root.Find_1NN(row)
        # get the quality rating
        quality_rating = nearest_neighbor[-1]
        predictions.append(quality_rating)
        
    for prediction in predictions:
        print(int(prediction))

if __name__ == '__main__':
    main()