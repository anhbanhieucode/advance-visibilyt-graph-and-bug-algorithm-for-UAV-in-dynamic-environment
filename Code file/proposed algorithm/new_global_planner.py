import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
import time
from shapely.geometry import LineString, box

def rect_distance(rect1, rect2):
    dx = max(rect2[0] - rect1[2], rect1[0] - rect2[2], 0)
    dy = max(rect2[1] - rect1[3], rect1[1] - rect2[3], 0)
    return np.hypot(dx, dy)

def get_rectangle_vertices(rect, p):
    x_min, y_min, x_max, y_max = rect
    return [(x_min - p, y_min - p), (x_max + p, y_min - p),
            (x_max + p, y_max + p), (x_min - p, y_max + p)]

def line_intersects_rectangles(line, rectangles, p):
    line_geom = LineString(line)
    return any(box(rect[0]-p, rect[1]-p, rect[2]+p, rect[3]+p).intersects(line_geom) for rect in rectangles)

def add_edges(nodes, rectangles, r):
    edges = []
    for i, node1 in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            node2 = nodes[j]
            if not line_intersects_rectangles([node1, node2], rectangles, r):
                edges.append((node1, node2))
    return edges

def dijkstra(start, goal, edges):
    pq = [(0, start)]  # (distance, node)
    distances = {start: 0}
    predecessors = {}
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_node == goal:
            path = []
            while current_node:
                path.append(current_node)
                current_node = predecessors.get(current_node)
            return path[::-1]
        
        for neighbor1, neighbor2 in edges:
            neighbor = neighbor2 if current_node == neighbor1 else neighbor1 if current_node == neighbor2 else None
            if neighbor is None:
                continue
            
            new_distance = current_distance + np.linalg.norm(np.array(current_node) - np.array(neighbor))
            if new_distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_distance
                predecessors[neighbor] = current_node
                heapq.heappush(pq, (new_distance, neighbor))
    return None

def plot_rectangles(start, goal, rectangles, p ,r, m):
    fig, ax = plt.subplots()
    random.seed(42)
    colors = ["#" + ''.join(random.choices('0123456789ABCDEF', k=6)) for _ in range(len(rectangles))]
    rect_colors = list(range(len(rectangles)))
    grouped_flag = [False] * len(rectangles)
    
    start_time = time.time()

    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            if rect_distance(rectangles[i], rectangles[j]) < m:
                if not grouped_flag[j]:
                    rect_colors[j] = rect_colors[i]
                    grouped_flag[j] = True
                else:
                    rect_colors[i] = rect_colors[j]
                    grouped_flag[i] = True
    
    point_list = []
    line = LineString([start, goal])
    
    for i, rect in enumerate(rectangles):
        rect_box = box(rect[0]-p, rect[1]-p, rect[2]+p, rect[3]+p)
        
        if rect_box.intersects(line):
            for j, gr in enumerate(rect_colors):
                if gr == rect_colors[i]:
                    point_list.extend(get_rectangle_vertices(rectangles[j], p))
    
    unique_colors = set(rect_colors)
    color_map = {group: colors[idx % len(colors)] for idx, group in enumerate(unique_colors)}
    for i, rect in enumerate(rectangles):
        rect_patch = plt.Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1],
                                   color=color_map[rect_colors[i]], linewidth=2)
        ax.add_patch(rect_patch)
    
    chosen_nodes = [start] + point_list + [goal]
    edges = add_edges(chosen_nodes, rectangles, r)
    path = dijkstra(tuple(start), tuple(goal), edges)
    
    end_time = time.time()
    
    print(f"Execution Time: {end_time - start_time:.5f} seconds")
    ax.plot(*start, 'ro', label="Start")
    ax.plot(*goal, 'go', label="Goal")
    ax.plot([start[0], goal[0]], [start[1], goal[1]], 'b--', label="Path Line")
    for point in point_list:
        ax.plot(*point, 'bo')
    
    for edge in edges:
        x_values, y_values = zip(*edge)
        ax.plot(x_values, y_values, 'grey', linewidth=1)
    
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, 'b-', linewidth=2, label="Shortest Path")
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.legend()
    plt.grid()
    plt.show()
'''
rectangles = [
    (3, 5, 13, 15), (20, 10, 32, 22), (40, 8, 53, 21), (60, 2, 73, 15), (80, 5, 89, 14),
    (5, 25, 17, 37), (25, 28, 39, 42), (45, 30, 59, 44), (65, 25, 78, 38), (82, 30, 90, 38),
    (7, 50, 20, 63), (27, 55, 41, 69), (50, 50, 63, 63), (68, 55, 82, 69), (85, 50, 90, 60),
    (10, 75, 23, 88), (30, 78, 45, 90), (55, 75, 70, 90), (75, 75, 88, 88), (5, 80, 15, 90)
]
'''
rectangles = [
    #[2, 2, 4, 3],       # Medium rectangle in bottom-left
    #[10, 2, 6, 2],      # Long horizontal rectangle in lower-middle
    #[20, 2, 2, 4],      # Small vertical rectangle in bottom-right
    #[2, 10, 3, 6],      # Long vertical rectangle in middle-left
    #[10, 10, 4, 4],     # Square in the center
    #[22, 10, 5, 2],     # Thin horizontal rectangle in middle-right
    #[2, 22, 3, 2],      # Small rectangle in top-left
    #[10, 22, 8, 1],     # Long horizontal rectangle in top-middle
    #[22, 22, 4, 3],     # Medium rectangle in top-right
    #[16,8, 2 , 12],
    #[10, 15, 2, 4],
    [34, 1, 5, 12], [46, 1, 9, 7], [58, 1, 8, 8],
    [70, 1, 7, 6], [82, 1, 8, 10], [94, 1, 6, 5],
    [2, 15, 7, 8], [10, 15, 8, 5], [22, 15, 6, 8],
    [34, 15, 12, 3], [50, 15, 5, 7], [58, 15, 7, 6],
    [70, 15, 9, 4], [82, 15, 8, 5], [94, 15, 6, 8],
    [2, 30, 5, 6], [10, 30, 10, 3], [25, 30, 8, 6],
    [34, 30, 4, 10], [46, 30, 5, 7], [58, 30, 6, 5],
    [70, 30, 9, 4], [82, 30, 6, 5], [94, 30, 5, 10],
    [2, 45, 10, 9], [14, 45, 5, 4], [25, 45, 6, 8],
    [34, 45, 8, 5], [46, 45, 7, 6], [58, 45, 5, 7],
    [70, 45, 6, 4], [82, 45, 9, 5], [94, 45, 6, 5],
    [2, 60, 5, 6], [12, 60, 5, 12], [26, 60, 7, 5],
    [34, 60, 4, 9], [46, 60, 11, 3], [58, 60, 6, 8],
    [70, 60, 9, 4], [82, 60, 5, 5], [94, 60, 7, 8],
    [2, 75, 8, 5], [12, 75, 6, 10], [22, 75, 5, 7],
    [34, 75, 7, 4], [46, 75, 10, 5], [58, 75, 5, 8],
    [70, 75, 6, 7], [82, 75, 8, 5], [94, 75, 10, 4],
    [2, 90, 6, 5], [10, 90, 8, 10], [20, 90, 4, 8],
    [30, 90, 12, 3], [45, 90, 5, 6], [58, 90, 7, 4],
    [70, 90, 8, 5], [82, 90, 10, 4], [94, 90, 6, 5]
]

def convert_rectangles(rectangles):
    return [(x, y, x + w, y + h) for x, y, w, h in rectangles]

rectangles = convert_rectangles(rectangles)

start = (1, 1)
goal = (99, 99)
p = 0.6
m = 2*p
r = 0.5
plot_rectangles(start, goal, rectangles, p, r, m)
#print(f"Execution Time: {end_time - start_time:.5f} seconds")