import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import time
import pandas as pd
# Map limits and elements
map_limits = [0, 100, 0, 100]
start = [1, 1]
goal = [99, 99]

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


'''
rectangles =  [
      [14,10,50,5],
   [65,10,5,50],
   [14,55,50,5]
]
'''
'''
rectangles =  [
   #[4,30,3,30],
   #[14,27,3,35],
   #[24,28,3,35],
   #[34,20,3,38],
   #[44,30,3,36],
   #[54,26,3,30],
   #[64,30,3,30],
   #[74,40,3,40],
   #[84,30,3,30],
   #[94,30,3,35],

   [10,10,10,5],
   [60,5,10,7],
   [20,80,10,8],
   [50,85,15,7],
   [80,14,15,8],
   [4,70,10,5]
   
]
'''
robot_radius = 0.5
boundary_offset = 0.6

# Function to check if a line intersects a rectangle
def line_intersects_rectangle(line, rect, bound=robot_radius):
    x1, y1, x2, y2 = line
    rx, ry, rw, rh = rect

    # Rectangle edges
    edges = [
        [rx - bound, ry - bound, rx + rw + bound, ry - bound],  # Bottom edge
        [rx + rw + bound, ry - bound, rx + rw + bound, ry + rh + bound],  # Right edge
        [rx - bound, ry + rh + bound, rx + rw + bound, ry + rh + bound],  # Top edge
        [rx - bound, ry - bound, rx - bound, ry + rh + bound]  # Left edge
    ]

    # Check if line intersects any edge
    for edge in edges:
        ex1, ey1, ex2, ey2 = edge
        if lines_intersect(x1, y1, x2, y2, ex1, ey1, ex2, ey2):
            return True
    return False


# Helper function to check if two line segments intersect
def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    def ccw(ax, ay, bx, by, cx, cy):
        return (cy - ay) * (bx - ax) >= (by - ay) * (cx - ax)

    return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4)


# Function to construct a graph
def construct_graph(nodes, edges):
    graph = {i: [] for i in range(len(nodes))}
    for edge in edges:
        i, j = edge
        dist = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))  # Euclidean distance
        graph[i].append((j, dist))
        graph[j].append((i, dist))  # Undirected graph
    return graph


# Dijkstra's algorithm
def dijkstra(graph, start, goal):
    queue = [(0, start)]  # Priority queue with (distance, node)
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    parent = {start: None}

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        # If we reached the goal, stop
        if current_node == goal:
            break

        # Visit neighbors
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                parent[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    # Reconstruct the shortest path
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    return path, distances[goal]

start_time = time.time()
# Generate nodes (start, goal, and boundary corners)
nodes = [start, goal]
for rect in rectangles:
    x, y, width, height = rect
    corners = [
        [x - boundary_offset, y - boundary_offset],
        [x + width + boundary_offset, y - boundary_offset],
        [x + width + boundary_offset, y + height + boundary_offset],
        [x - boundary_offset, y + height + boundary_offset]
    ]
    nodes.extend(corners)


# Generate edges, avoiding obstacle intersections
edges = []
for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        if i >= j:  # Avoid duplicate edges
            continue
        line = [*node1, *node2]
        if not any(line_intersects_rectangle(line, rect) for rect in rectangles):
            edges.append((i, j))

# Construct graph and run Dijkstra
graph = construct_graph(nodes, edges)
start_index, goal_index = 0, 1
shortest_path, total_distance = dijkstra(graph, start_index, goal_index)

end_time = time.time()
elapsed_time = end_time-start_time
print("elapsed time: ", end_time - start_time)
# Plot the graph with the shortest path
plt.figure(figsize=(10, 10))
plt.xlim(map_limits[0:2])
plt.ylim(map_limits[2:4])

# Plot obstacles
for rect in rectangles:
    x, y, width, height = rect
    plt.gca().add_patch(plt.Rectangle((x, y), width, height, color='black', alpha=0.7))

# Plot nodes
nodes = np.array(nodes)
plt.scatter(nodes[:, 0], nodes[:, 1], color='blue', zorder=5, label='Nodes')
plt.scatter(*start, color='red', zorder=6, label='Start')
plt.scatter(*goal, color='green', zorder=6, label='Goal')

# Plot edges
for edge in edges:
    node1, node2 = nodes[edge[0]], nodes[edge[1]]
    plt.plot([node1[0], node2[0]], [node1[1], node2[1]], color='black', alpha=0.5)

# Plot the shortest path
path_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
for edge in path_edges:
    node1, node2 = nodes[edge[0]], nodes[edge[1]]
    plt.plot([node1[0], node2[0]], [node1[1], node2[1]], color='orange', linewidth=2, zorder=10, label='Shortest Path' if edge == path_edges[0] else None)

# Plot contour around the shortest path
for edge in path_edges:
    node1, node2 = nodes[edge[0]], nodes[edge[1]]
    x1, y1 = node1
    x2, y2 = node2

    # Vector perpendicular to the line segment
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    offset_x = -dy / length * robot_radius
    offset_y = dx / length * robot_radius

    # Define the contour as a polygon
    contour_points = [
        [x1 + offset_x, y1 + offset_y],
        [x1 - offset_x, y1 - offset_y],
        [x2 - offset_x, y2 - offset_y],
        [x2 + offset_x, y2 + offset_y]
    ]
    polygon = Polygon(contour_points, closed=True, color='blue', alpha=0.3)
    plt.gca().add_patch(polygon)


file_path = 'comparison_map_3.xlsx'
# Check if the file exists
try:
    # Load existing data from the Excel file
    df = pd.read_excel(file_path)
except FileNotFoundError:
    # If the file doesn't exist, create an empty DataFrame with the correct columns
    df = pd.DataFrame(columns=['Elapsed_time_RRT_Connect', 'Path_length_RRT_Connect', 'Time_Old_Visibility_Graph', 'Path_length_Visibility_Graph', 'Elapsed Time_T', 'Value A_B'])

# Run the process and get the elapsed time and value A

# Create a new DataFrame for the new entry starting from the second row
new_entry = pd.DataFrame({'Time_Old_Visibility_Graph': [elapsed_time], 'Path_length_Visibility_Graph': [total_distance]})

# Append the new entry to the DataFrame
# If the DataFrame already has data, we want to ensure it starts from the second row
# Concatenate the new entry to the existing DataFrame
df = pd.concat([df, new_entry], ignore_index=True)

# Write the updated DataFrame back to the Excel file
df.to_excel(file_path, index=False, sheet_name='Sheet2')

print(f"Data appended to {file_path} successfully.")


# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Shortest Path with Contour (Distance: {total_distance:.2f})')
plt.legend()
plt.show()
