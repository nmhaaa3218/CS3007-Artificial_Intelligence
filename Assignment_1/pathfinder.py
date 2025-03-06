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
import random

class Solution:
    def __init__(self, path, visit_count_grid, first_visit, last_visit):
        self.path = path # list of actions ex: ['up', 'down', 'left', 'right']
        self.visit_count_grid = visit_count_grid # same as input grid but with number of visits
        self.first_visit = first_visit # same as input grid but with turn of first visit
        self.last_visit = last_visit # same as input grid but with turn of last visit
        
class Node:
    def __init__(self, state, parent, action, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost # for UCS and A* search, default as 0 for BFS to not throw error
        
# Formulalize the problem
class Problem:
    def __init__(self, initial_coordinate, target_coordinate, grid):
        self.initial_coordinate = initial_coordinate
        self.target_coordinate = target_coordinate
        self.grid = grid

    def check_completeness(self, current_coordinate):
        return current_coordinate == self.target_coordinate

    def actions(self, current_coordinate, randomized=True):
        available_actions = []
        if current_coordinate == self.target_coordinate:
            return []
        else:
            # check up down left right with boundary checks
            if current_coordinate[1] > 0 and self.grid[current_coordinate[0]][current_coordinate[1]-1] != 'X':
                available_actions.append('left')
            if current_coordinate[1] < len(self.grid) - 1 and self.grid[current_coordinate[0]][current_coordinate[1]+1] != 'X':
                available_actions.append('right')
            if current_coordinate[0] > 0 and self.grid[current_coordinate[0]-1][current_coordinate[1]] != 'X':
                available_actions.append('up')
            if current_coordinate[0] < len(self.grid[0]) - 1 and self.grid[current_coordinate[0]+1][current_coordinate[1]] != 'X':
                available_actions.append('down')
                
            # add randomization since observe that order of actions of final result tends to prefer left -> right -> up -> down 
            # observe that random sometimes solve in less steps
            if randomized:
                random.shuffle(available_actions)
                
            return available_actions

    # got mixed up between [r,c] and [c,r] 
    def result(self, state, action):
        direction_map = {
            'left': (0, -1),
            'right': (0, 1),
            'up': (-1, 0),
            'down': (1, 0)
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
    # initial node queue 
    node_queue = []
    initial_node = Node(problem.initial_coordinate, None, None)
    node_queue.append(initial_node)
    
    # initial closed set for visited nodes
    closed_set = set()
    
    # step counter
    step_counter = 0
    
    # initialize result grids - have the same structure as the input grid
    visit_count_grid = []
    first_visit_grid = []
    last_visit_grid = []
    rows = len(problem.grid)
    cols = len(problem.grid[0])
    for i in range(rows):
        visit_count_row = []
        first_visit_row = []
        last_visit_row = []
        for j in range(cols):
            if problem.grid[i][j] == "X":
                visit_count_row.append("X")
                first_visit_row.append("X") 
                last_visit_row.append("X")
            else:
                visit_count_row.append(0)
                first_visit_row.append(0)
                last_visit_row.append(0)
        visit_count_grid.append(visit_count_row)
        first_visit_grid.append(first_visit_row)
        last_visit_grid.append(last_visit_row)
        
    # begin searching        
    while node_queue:
        current_node = node_queue.pop(0) # get the first node in the queue
        current_coordinate = current_node.state # get current coordinate
        step_counter += 1 # update step counter for each loop
        
        # check if not obstacle
        if problem.grid[current_coordinate[0]][current_coordinate[1]] != "X":
            # update visit count
            visit_count_grid[current_coordinate[0]][current_coordinate[1]] += 1
            
            # check if first visit
            if first_visit_grid[current_coordinate[0]][current_coordinate[1]] == 0:
                # update first visit
                first_visit_grid[current_coordinate[0]][current_coordinate[1]] = step_counter
                
            # update last visit
            last_visit_grid[current_coordinate[0]][current_coordinate[1]] = step_counter
            
        # DEBUG
        # print("Step: ", step_counter)     
        # print("Current Coordinate: ", current_coordinate)
        
        # check if target is reached
        if problem.check_completeness(current_coordinate):
            path = backtracking(current_node)
            # print(path)
            return Solution(path, visit_count_grid, first_visit_grid, last_visit_grid)
        
        # check if current coordinate is not in closed set
        if current_coordinate not in closed_set:
            # add current coordinate to closed set
            closed_set.add(current_coordinate)
            
            # explore all possible actions from current coordinate
            for action in problem.actions(current_coordinate):
                # get next coordinate
                next_coordinate = problem.result(current_coordinate, action)
                # craete a new node
                next_node = Node(next_coordinate, current_node, action)
                # add new node to the queue
                node_queue.append(next_node)
    
    # return None if no solution 
    return Solution([], visit_count_grid, first_visit_grid, last_visit_grid)

# the more you know, it's Djiikstra on steroid =)))))))) (if you are reading this, you are a legend)
def uniform_cost_search(problem: Problem) -> Solution:
    # initial node queue 
    node_queue = []
    initial_node = Node(problem.initial_coordinate, None, None)
    node_queue.append(initial_node)
    
    # initial closed set for visited nodes
    closed_set = set()
    
    # step counter
    step_counter = 0
    
    # initialize result grids - have the same structure as the input grid
    visit_count_grid = []
    first_visit_grid = []
    last_visit_grid = []
    rows = len(problem.grid)
    cols = len(problem.grid[0])
    for i in range(rows):
        visit_count_row = []
        first_visit_row = []
        last_visit_row = []
        for j in range(cols):
            if problem.grid[i][j] == "X":
                visit_count_row.append("X")
                first_visit_row.append("X") 
                last_visit_row.append("X")
            else:
                visit_count_row.append(0)
                first_visit_row.append(0)
                last_visit_row.append(0)
        visit_count_grid.append(visit_count_row)
        first_visit_grid.append(first_visit_row)
        last_visit_grid.append(last_visit_row)
    
    # begin searching        
    while node_queue:
        current_node = node_queue.pop(0) # get the first node in the queue
        current_coordinate = current_node.state # get current coordinate
        step_counter += 1 # update step counter for each loop
        
        # check if not obstacle
        if problem.grid[current_coordinate[0]][current_coordinate[1]] != "X":
            # update visit count
            visit_count_grid[current_coordinate[0]][current_coordinate[1]] += 1
            
            # check if first visit
            if first_visit_grid[current_coordinate[0]][current_coordinate[1]] == 0:
                # update first visit
                first_visit_grid[current_coordinate[0]][current_coordinate[1]] = step_counter
                
            # update last visit
            last_visit_grid[current_coordinate[0]][current_coordinate[1]] = step_counter
            
        # DEBUG
        # print("Step: ", step_counter)     
        # print("Current Coordinate: ", current_coordinate)
        
        # check if target is reached
        if problem.check_completeness(current_coordinate):
            path = backtracking(current_node)
            # print(path)
            return Solution(path, visit_count_grid, first_visit_grid, last_visit_grid)
        
        # check if current coordinate is not in closed set
        if current_coordinate not in closed_set:
            # add current coordinate to closed set
            closed_set.add(current_coordinate)
            
            # explore all possible actions from current coordinate
            for action in problem.actions(current_coordinate):
                # get next coordinate
                next_coordinate = problem.result(current_coordinate, action)
                
                # calculate the cost
                cost = problem.step_cost(current_coordinate, next_coordinate)
                
                # craete a new node
                next_node = Node(next_coordinate, current_node, action, cost)
                
                # add new node to the queue
                node_queue.append(next_node)
                
                # sort the queue based on the cost
                node_queue = sorted(node_queue, key=lambda x: x.cost)
                
                # DEBUG
                # print("Next Coordinate: ", next_coordinate)
                # print("Cost: ", cost)
                
    # return None if no solution 
    return Solution([], visit_count_grid, first_visit_grid, last_visit_grid)
    

def a_star_search(problem: Problem, heuristic) -> Solution:
    # initial node queue
    node_queue = []
    initial_node = Node(problem.initial_coordinate, None, None)
    node_queue.append(initial_node)
    
    # initial closed set for visited nodes
    closed_set = set()
    
    # step counter
    step_counter = 0
    
    # initialize result grids - have the same structure as the input grid
    visit_count_grid = []
    first_visit_grid = []
    last_visit_grid = []
    rows = len(problem.grid)
    cols = len(problem.grid[0])
    for i in range(rows):
        visit_count_row = []
        first_visit_row = []
        last_visit_row = []
        for j in range(cols):
            if problem.grid[i][j] == "X":
                visit_count_row.append("X")
                first_visit_row.append("X") 
                last_visit_row.append("X")
            else:
                visit_count_row.append(0)
                first_visit_row.append(0)
                last_visit_row.append(0)
        visit_count_grid.append(visit_count_row)
        first_visit_grid.append(first_visit_row)
        last_visit_grid.append(last_visit_row)
    
    # begin searching        
    while node_queue:
        current_node = node_queue.pop(0) # get the first node in the queue
        current_coordinate = current_node.state # get current coordinate
        step_counter += 1 # update step counter for each loop
        
        # check if not obstacle
        if problem.grid[current_coordinate[0]][current_coordinate[1]] != "X":
            # update visit count
            visit_count_grid[current_coordinate[0]][current_coordinate[1]] += 1
            
            # check if first visit
            if first_visit_grid[current_coordinate[0]][current_coordinate[1]] == 0:
                # update first visit
                first_visit_grid[current_coordinate[0]][current_coordinate[1]] = step_counter
                
            # update last visit
            last_visit_grid[current_coordinate[0]][current_coordinate[1]] = step_counter
            
        # DEBUG
        # print("Step: ", step_counter)     
        # print("Current Coordinate: ", current_coordinate)
        
        # check if target is reached
        if problem.check_completeness(current_coordinate):
            path = backtracking(current_node)
            # print(path)
            return Solution(path, visit_count_grid, first_visit_grid, last_visit_grid)
        
        # check if current coordinate is not in closed set
        if current_coordinate not in closed_set:
            # add current coordinate to closed set
            closed_set.add(current_coordinate)
            
            # explore all possible actions from current coordinate
            for action in problem.actions(current_coordinate):
                # get next coordinate
                next_coordinate = problem.result(current_coordinate, action)
                
                # calculate the cost
                step_cost = problem.step_cost(current_coordinate, next_coordinate)
                heuristic_cost = heuristic(next_coordinate, problem.target_coordinate)
                cost = step_cost + heuristic_cost
                
                # craete a new node
                next_node = Node(next_coordinate, current_node, action, cost)
                
                # add new node to the queue
                node_queue.append(next_node)
                
                # sort the queue based on the cost
                node_queue = sorted(node_queue, key=lambda x: x.cost)
                
                # DEBUG
                # print("Next Coordinate: ", next_coordinate)
                # print("Cost: ", cost)
                
    # return None if no solution 
    return Solution([], visit_count_grid, first_visit_grid, last_visit_grid)

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
    
    # Transform coordinate to 0-based
    start_row -= 1
    start_col -= 1
    target_row -= 1
    target_col -= 1
    
    # Read the map
    grid = []
    for line in lines[3:]:
        grid.append([x for x in line.split()]) 
        
    return Problem((start_row, start_col), (target_row, target_col), grid)

# OUTPUT FUNCTIONS
def calculate_path_cost(problem: Problem, path: list):
    cost = 0
    current_coordinate = problem.initial_coordinate
    for action in path:
        next_coordinate = problem.result(current_coordinate, action)
        cost += problem.step_cost(current_coordinate, next_coordinate)
        current_coordinate = next_coordinate
    return cost

def format_release_output(problem: Problem, solution: Solution):
    solution_grid = problem.grid
    initial_coordinate = problem.initial_coordinate
    target_coordinate = problem.target_coordinate
    
    steps = solution.path
    curr_row = initial_coordinate[0]
    curr_col = initial_coordinate[1]
    
    # check if steps is empty
    if not steps:
        return "no solution"
    
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
    visit_count_grid = solution.visit_count_grid
    first_visit = solution.first_visit
    last_visit = solution.last_visit
    
    # change never visit cell to "."
    for i in range(len(visit_count_grid)):
        for j in range(len(visit_count_grid[0])):
            if visit_count_grid[i][j] == 0:
                visit_count_grid[i][j] = "."
                first_visit[i][j] = "."
                last_visit[i][j] = "."
    
    return formatted_solution, visit_count_grid, first_visit, last_visit
            

# MAIN FUNCTION
def main():
    parser = argparse.ArgumentParser(description="Pathfinding using BFS, UCS, or A*")
    parser.add_argument("mode", choices=["debug", "release"])
    parser.add_argument("map", help="Path to map file")
    parser.add_argument("algorithm", choices=["bfs", "ucs", "astar"])
    parser.add_argument("heuristic", nargs="?", default=None, choices=["euclidean", "manhattan"])
    args = parser.parse_args()

    # Read the map file and create the Problem instance
    # problem = read_map(args.map)
    problem = read_map("sample.txt")
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
    print("Total cost: ", calculate_path_cost(problem, solution.path)) # FOR CURIOUSITY PURPOSES
    if args.mode == "debug":
        formatted_solution, visit_count_grid, first_visit, last_visit = format_debug_output(problem, solution)
        print("path:")
        if formatted_solution == "no solution":
            print("no solution")
        else:
            for row in formatted_solution:
                print(" ".join(row))
        print("#visits: ")
        for row in visit_count_grid:
            print(" ".join([str(x) for x in row]))
        print("first visit: ")
        for row in first_visit:
            print(" ".join([str(x) for x in row]))
        print("last visit: ")
        for row in last_visit:
            print(" ".join([str(x) for x in row]))
    else:
        formatted_solution = format_release_output(problem, solution)
        if formatted_solution == "no solution":
            print("no solution")
        else:
            for row in formatted_solution:
                print(" ".join(row))


if __name__ == "__main__":
    main()
