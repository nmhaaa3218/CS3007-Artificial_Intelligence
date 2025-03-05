# Assignment 1: Search Algorithms
# Author: Manh Ha Nguyen - a1840406 - University of Adelaide
# Date: 4-Mar-2025
#============================================================#

STUDENT_ID = 'a1840406' 
DEGREE = 'UG' 

import sys
import argparse
import os
import math

class Solution:
    def __init__(self, path, visit_count_grid, first_visit, last_visit):
        self.path = path 
        self.visit_count_grid = visit_count_grid 
        self.first_visit = first_visit
        self.last_visit = last_visit
        
class Node:
    def __init__(self, state, parent, action, height, path_cost, depth, number_of_visits, first_visit, last_visit):
        self.state = state
        self.parent = parent
        self.action = action
        self.height = height
        self.path_cost = path_cost
        self.depth = depth
        self.number_of_visits = number_of_visits
        self.first_visit = first_visit
        self.last_visit = last_visit
        
        
# Formulalize the problem
class Problem:
    def __init__(self, initial_coordinate, target_coordinate, grid):
        self.initial_coordinate = initial_coordinate
        self.target_coordinate = target_coordinate
        self.grid = grid

    def check_completeness(self, current_coordinate):
        return current_coordinate == self.target_coordinate

    def actions(self, current_coordinate):
        available_actions = []
        if current_coordinate == self.target_coordinate:
            return []
        else:
            # check up down left right with boundary checks
            if current_coordinate[0] > 0 and self.grid[current_coordinate[0] - 1][current_coordinate[1]] != 'X':
                available_actions.append('left')
            if current_coordinate[0] < len(self.grid) - 1 and self.grid[current_coordinate[0] + 1][current_coordinate[1]] != 'X':
                available_actions.append('right')
            if current_coordinate[1] > 0 and self.grid[current_coordinate[0]][current_coordinate[1] - 1] != 'X':
                available_actions.append('up')
            if current_coordinate[1] < len(self.grid[0]) - 1 and self.grid[current_coordinate[0]][current_coordinate[1] + 1] != 'X':
                available_actions.append('down')
            return available_actions

    def result(self, state, action):
        direction_map = {
            'left': (-1, 0),
            'right': (1, 0),
            'up': (0, -1),
            'down': (0, 1)
        }
        delta = direction_map[action]
        return (state[0] + delta[0], state[1] + delta[1])

    def step_cost(self, current_coordinate, next_coordinate):
        current_height = int(self.grid[current_coordinate[0]][current_coordinate[1]])
        next_height = int(self.grid[next_coordinate[0]][next_coordinate[1]])
        if next_height > current_height:
            return 1 + next_height - current_height
        else:
            return 1

# Search algorithms 
def backtracking(node: Node):
    path = []
    while node.parent is not None:
        path.insert(0, node.action)
        node = node.parent
    return path

def breadth_first_search(problem: Problem) -> Solution:
    # initial node queue + closed set
    node_queue = []
    initial_node = Node(problem.initial_coordinate, None, None, 0, 0)
    node_queue.append(initial_node)
    closed_set = set()
    
    # initialize result grids
    visit_count_grid = []
    first_visit_grid = []
    last_visit_grid = []
    rows = len(problem.grid)
    cols = len(problem.grid[0])
    for i in range(rows):
        row = []
        for j in range(cols):
            if problem.grid[i][j] == "X":
                row.append("X")
            else:
                row.append(0)
        visit_count_grid.append(row)
        first_visit_grid.append(row)
        last_visit_grid.append(row)
            
    while node_queue:
        current_node = node_queue.pop(0)
        current_coordinate = current_node.state
        
        if problem.check_completeness(current_coordinate):
            return backtracking(current_node)
        
        if current_coordinate not in closed_set:
            closed_set.add(current_coordinate)
            for action in problem.actions(current_coordinate):
                next_coordinate = problem.result(current_coordinate, action)
                path_cost = current_node.path_cost + problem.step_cost(current_coordinate, next_coordinate)
                next_node = Node(next_coordinate, current_node, action, path_cost, current_node.depth + 1)
                node_queue.append(next_node)
    
    return None


def uniform_cost_search(problem: Problem) -> Solution:
    """
    Uniform Cost Search algorithm.
    Use a priority queue keyed on path_cost.
    """
    return None

def a_star_search(problem: Problem, heuristic: function) -> Solution:
    """
    A* Search algorithm.
    Use a priority queue where priority = path_cost + heuristic.
    """
    return None

# CALCULATE DISTANCE 
def l2_distance(current_coor, target_coor):
    (x1, y1) = current_coor
    (x2, y2) = target_coor
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def l1_distance(current_coor, target_coor):
    (x1, y1) = current_coor
    (x2, y2) = target_coor
    return abs(x1 - x2) + abs(y1 - y2)


# INPUT FUNCTIONS
def read_map(file_path: str):
    data = open(file_path, "r")
    lines = data.readlines()

    # Map size
    size = lines[0].split()
    rows = int(size[0])
    cols = int(size[1])
    
    # start and target coordinates
    start = lines[1].split()
    start_row = int(start[0])
    start_col = int(start[1])
    
    target = lines[2].split()
    target_row = int(target[0])
    target_col = int(target[1])
    
    # Read the map
    grid = []
    for line in lines[3:]:
        grid.append([x for x in line.split()]) 
        
    return Problem((start_row, start_col), (target_row, target_col), grid)

# OUTPUT FUNCTIONS
def format_release_output(problem: Problem, solution: Solution):
    solution_grid = problem.grid
    initial_coordinate = problem.initial_coordinate
    target_coordinate = problem.target_coordinate
    
    steps = solution.path
    curr_row = initial_coordinate[0]
    curr_col = initial_coordinate[1]
    
    # mark the start and target
    solution_grid[initial_coordinate[0]][initial_coordinate[1]] = '*'
    solution_grid[target_coordinate[0]][target_coordinate[1]] = '*'
    
    # mark the path
    for step in steps:
        if step == "up":
            curr_row -= 1
            solution_grid[curr_row][curr_col] = '*'
        elif step == "down":
            curr_row += 1
            solution_grid[curr_row][curr_col] = '*'
        elif step == "left":
            curr_col -= 1
            solution_grid[curr_row][curr_col] = '*'
        elif step == "right":
            curr_col += 1
            solution_grid[curr_row][curr_col] = '*'
            
    return solution_grid

def format_debug_output(problem: Problem, solution: Solution):
    formatted_solution = format_release_output(problem, solution)
    return formatted_solution, solution.visits_grid, solution.first_visit, solution.last_visit
            

# MAIN FUNCTION
def main():
    parser = argparse.ArgumentParser(description="Pathfinding using BFS, UCS, or A*")
    parser.add_argument("mode", choices=["debug", "release"], help="Mode: debug or release")
    parser.add_argument("map", help="Path to map file")
    parser.add_argument("algorithm", choices=["bfs", "ucs", "astar"], help="Algorithm to use")
    parser.add_argument("heuristic", nargs="?", default=None, choices=["euclidean", "manhattan"], help="Heuristic for A* (ignored for BFS and UCS)")
    args = parser.parse_args()

    # Read the map file and create the Problem instance
    problem = read_map(args.map)
    if problem is None:
        print("Error reading map file.")
        sys.exit(1)

    # Choose search algorithm based on command-line argument
    if args.algorithm == "bfs":
        solution = breadth_first_search(problem)
    elif args.algorithm == "ucs":
        solution = uniform_cost_search(problem)
    elif args.algorithm == "astar":
        if args.heuristic == "euclidean":
            heuristic = l2_distance
        elif args.heuristic == "manhattan":
            heuristic = l1_distance
        else:
            print("Heuristic must be specified for A* search.")
            sys.exit(1)
        solution = a_star_search(problem, heuristic)
    else:
        print("Invalid algorithm specified.")
        sys.exit(1)

    # MUST BE ABLE TO HANDLE WHEN NO SOLUTION IS FOUND
    if args.mode == "debug":
        format_debug_output(problem, solution)
    else:
        format_release_output(problem, solution)


if __name__ == "__main__":
    main()
