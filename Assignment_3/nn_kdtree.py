# kdtree implementation for CS3007 Assignment 3
STUDENT_ID = 'a1840406'
DEGREE = 'UG'

import numpy as np
import pandas as pd
import argparse

class Node:
    def __init__(self, point, label, split_dim, split_val, left=None, right=None):
        self.point = point      
        self.label = label      
        self.split_dim = split_dim
        self.split_val = split_val
        self.left = left
        self.right = right

def build_kdtree(points, current_depth=0):
    if len(points) == 0:
        return None
    
    elif len(points) == 1:
        point = points[0]
        label = point[-1]
        split_dim = current_depth % len(point)
        split_val = point[split_dim]
        return Node(point, label, split_dim, split_val)
    

        

        