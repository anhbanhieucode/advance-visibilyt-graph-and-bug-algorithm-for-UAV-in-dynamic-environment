"""
RRT_CONNECT_2D
@author: huiming zhou
"""
import pandas as pd
import os
import sys
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

import env, plotting, utils


def calculate_path_length(path):
    """
    Calculate the length of a path.

    :param path: List of tuples representing the path [(x1, y1), (x2, y2), ...]
    :return: Total length of the path
    """
    if not path or len(path) < 2:
        return 0  # No valid path or single point in path

    path_length = 0
    for i in range(1, len(path)):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        path_length += math.hypot(x2 - x1, y2 - y1)

    return path_length


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class RrtConnect:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.V1 = [self.s_start]
        self.V2 = [self.s_goal]

        self.env = env.Env()
        self.plotting = plotting.Plotting(s_start, s_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def planning(self):
        
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.s_goal, self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.V1, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.V1.append(node_new)
                node_near_prim = self.nearest_neighbor(self.V2, node_new)
                node_new_prim = self.new_state(node_near_prim, node_new)

                if node_new_prim and not self.utils.is_collision(node_new_prim, node_near_prim):
                    self.V2.append(node_new_prim)

                    while True:
                        node_new_prim2 = self.new_state(node_new_prim, node_new)
                        if node_new_prim2 and not self.utils.is_collision(node_new_prim2, node_new_prim):
                            self.V2.append(node_new_prim2)
                            node_new_prim = self.change_node(node_new_prim, node_new_prim2)
                        else:
                            break

                        if self.is_node_same(node_new_prim, node_new):
                            break

                if self.is_node_same(node_new_prim, node_new):
                    return self.extract_path(node_new, node_new_prim)

            if len(self.V2) < len(self.V1):
                list_mid = self.V2
                self.V2 = self.V1
                self.V1 = list_mid

        return None

    @staticmethod
    def change_node(node_new_prim, node_new_prim2):
        node_new = Node((node_new_prim2.x, node_new_prim2.y))
        node_new.parent = node_new_prim

        return node_new

    @staticmethod
    def is_node_same(node_new_prim, node_new):
        if node_new_prim.x == node_new.x and \
                node_new_prim.y == node_new.y:
            return True

        return False

    def generate_random_node(self, sample_goal, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return sample_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    @staticmethod
    def extract_path(node_new, node_new_prim):
        path1 = [(node_new.x, node_new.y)]
        node_now = node_new

        while node_now.parent is not None:
            node_now = node_now.parent
            path1.append((node_now.x, node_now.y))

        path2 = [(node_new_prim.x, node_new_prim.y)]
        node_now = node_new_prim

        while node_now.parent is not None:
            node_now = node_now.parent
            path2.append((node_now.x, node_now.y))

        return list(list(reversed(path1)) + path2)

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    x_start = (0, 50)  # Starting node
    x_goal = (50, 50)  # Goal node

    rrt_conn = RrtConnect(x_start, x_goal, 5, 0.05, 5000)
    start_t = time.time()
    path = rrt_conn.planning()
    end_t = time.time()
    elapsed_time = end_t-start_t
    print("elapsed_time:", end_t - start_t)
    if path:
        path_length = calculate_path_length(path)
        print(f"Path length: {path_length}")
    else:
        print("No valid path found.")
    file_path = 'comparison_map_3.xlsx'
    # Check if the file exists
    try:
        # Load existing data from the Excel file
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty DataFrame with the correct columns
        df = pd.DataFrame(columns=['Elapsed_time_RRT_Connect', 'Path_length_RRT_Connect', 'Column3', 'Column4', 'Elapsed Time_T', 'Value A_B'])

    # Run the process and get the elapsed time and value A

    # Create a new DataFrame for the new entry starting from the second row
    new_entry = pd.DataFrame({'Elapsed_time_RRT_Connect': [elapsed_time], 'Path_length_RRT_Connect': [path_length]})

    # Append the new entry to the DataFrame
    # If the DataFrame already has data, we want to ensure it starts from the second row
    # Concatenate the new entry to the existing DataFrame
    df = pd.concat([df, new_entry], ignore_index=True)

    # Write the updated DataFrame back to the Excel file
    df.to_excel(file_path, index=False, sheet_name='Sheet2')

    print(f"Data appended to {file_path} successfully.")
    
    rrt_conn.plotting.animation_connect(rrt_conn.V1, rrt_conn.V2, path, "RRT_CONNECT")



if __name__ == '__main__':
    main()
